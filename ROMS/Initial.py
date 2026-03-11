"""
ROMS Initial Conditions Generator - FIXED FILL VALUE HANDLING
==============================================================
Creates initial conditions from CMEMS 3D data with robust fill value handling.

IMPROVEMENTS:
1. Handles masked arrays (common in CMEMS)
2. Variable-specific physical range checks
3. Tolerant fill value comparison
4. Catches extreme values like 1e20, 9.96921e+36
5. Better validation and reporting
"""

import numpy as np
import netCDF4 as nc
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator, interp1d
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_FILE = './ocean_grd_fine.nc'
CMEMS_FILE = './cmems_data/cmems_phy_P1D_snap.nc'
OUTPUT_FILE = './roms_ini_fine.nc'

# Vertical grid parameters (must match roms.in)
N_LEVELS = 30
THETA_S = 7.0
THETA_B = 2.0
TCLINE = 200.0
VTRANSFORM = 2
VSTRETCHING = 4

# Initial time (days since 1900-01-01 for 2020-01-02)
INIT_TIME = 43830.0

# =============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_var_safe(nc_var, varname):
    """
    Read NetCDF variable and replace fill values with NaN
    
    ROBUST VERSION - handles all common NetCDF fill value patterns
    """
    data = nc_var[:]
    
    # Handle masked arrays (CMEMS uses these heavily)
    if hasattr(data, 'mask'):
        n_masked = np.sum(data.mask)
        if n_masked > 0:
            log(f"    Masked array: {n_masked:,} masked values found")
        data = np.ma.filled(data, np.nan)
    
    # Convert to float if needed
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    
    initial_size = data.size
    n_bad = 0
    
    # Check for NetCDF _FillValue attribute
    if hasattr(nc_var, '_FillValue'):
        fill = nc_var._FillValue
        # Use tolerance for float comparison
        n_fill = np.sum(np.abs(data - fill) < 1e-5)
        if n_fill > 0:
            log(f"    _FillValue={fill}: replacing {n_fill:,} values")
            data = np.where(np.abs(data - fill) < 1e-5, np.nan, data)
            n_bad += n_fill
    
    # Check for missing_value attribute
    if hasattr(nc_var, 'missing_value'):
        missing = nc_var.missing_value
        n_miss = np.sum(np.abs(data - missing) < 1e-5)
        if n_miss > 0:
            log(f"    missing_value={missing}: replacing {n_miss:,} values")
            data = np.where(np.abs(data - missing) < 1e-5, np.nan, data)
            n_bad += n_miss
    
    # Variable-specific physical range checks
    var_lower = varname.lower()
    if 'temp' in var_lower or 'thetao' in var_lower:
        # Temperature: -10 to 50°C
        bad = (data < -10) | (data > 50)
        var_type = "temperature"
    elif 'sal' in var_lower or var_lower == 'so':
        # Salinity: 0 to 50 PSU
        bad = (data < 0) | (data > 50)
        var_type = "salinity"
    elif 'vel' in var_lower or var_lower in ['uo', 'vo', 'u', 'v', 'usi', 'vsi']:
        # Velocities: -20 to 20 m/s
        bad = (data < -20) | (data > 20)
        var_type = "velocity"
    elif 'ssh' in var_lower or 'zos' in var_lower or 'zeta' in var_lower:
        # SSH: -50 to 50 m
        bad = (data < -50) | (data > 50)
        var_type = "SSH"
    elif 'ice' in var_lower or 'si' in var_lower:
        # Ice variables
        if 'conc' in var_lower:
            bad = (data < 0) | (data > 1)
            var_type = "ice concentration"
        elif 'thick' in var_lower:
            bad = (data < 0) | (data > 50)
            var_type = "ice thickness"
        else:
            bad = (data < -100) | (data > 100)
            var_type = "ice variable"
    else:
        # Generic: catch extreme values
        bad = (data < -1e6) | (data > 1e6)
        var_type = "generic"
    
    n_range = np.sum(bad)
    if n_range > 0:
        log(f"    Physical range ({var_type}): replacing {n_range:,} out-of-range values")
        data = np.where(bad, np.nan, data)
        n_bad += n_range
    
    # Catch common extreme fill values (9.96921e+36, 1e20, etc.)
    extreme_bad = (np.abs(data) > 1e8)  # Lowered threshold to catch more
    n_extreme = np.sum(extreme_bad)
    if n_extreme > 0:
        log(f"    Extreme values (>1e8): replacing {n_extreme:,} values")
        data = np.where(extreme_bad, np.nan, data)
        n_bad += n_extreme
    
    # Summary
    pct = 100 * n_bad / initial_size if initial_size > 0 else 0
    
    if n_bad > 0:
        log(f"    Total replaced: {n_bad:,} ({pct:.1f}%) bad values with NaN")
    
    # Report actual data range
    if not np.all(np.isnan(data)):
        log(f"    Valid range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f}")
    else:
        log(f"    ⚠ WARNING: All values are NaN!")
    
    return data


def compute_s_levels(N, theta_s, theta_b, Vstretching):
    """Compute ROMS S-coordinate vertical levels."""
    s_rho = (np.arange(1, N + 1) - N - 0.5) / N
    
    if Vstretching == 4:
        if theta_s > 0:
            Csur = (1 - np.cosh(theta_s * s_rho)) / (np.cosh(theta_s) - 1)
        else:
            Csur = -s_rho**2
        
        if theta_b > 0:
            Cbot = (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
        else:
            Cbot = Csur
        Cs_r = Cbot
    else:
        Cs_r = (1 - np.cosh(theta_s * s_rho)) / (np.cosh(theta_s) - 1)
    
    return s_rho, Cs_r


def compute_z_levels(s_rho, Cs_r, h, Vtransform, Tcline, zeta=0):
    """Compute actual z depths at each grid point (negative values = below surface)."""
    N = len(s_rho)
    eta, xi = h.shape
    z = np.zeros((N, eta, xi))
    
    hc = min(Tcline, np.nanmin(h[h > 0]))
    
    for k in range(N):
        if Vtransform == 1:
            z0 = hc * s_rho[k] + (h - hc) * Cs_r[k]
            z[k] = z0 + zeta * (1 + z0 / h)
        elif Vtransform == 2:
            z0 = (hc * s_rho[k] + h * Cs_r[k]) / (hc + h)
            z[k] = zeta + (zeta + h) * z0
    
    return z


class FastInterpolator:
    """Fast horizontal interpolation using RegularGridInterpolator."""
    
    def __init__(self, src_lon, src_lat, dst_lon, dst_lat):
        self.src_lat = np.asarray(src_lat).flatten()
        self.src_lon = np.asarray(src_lon).flatten()
        self.dst_shape = dst_lon.shape
        self.dst_lon = dst_lon
        self.dst_lat = dst_lat
        
        # Check if latitude needs flipping
        self.lat_flip = self.src_lat[0] > self.src_lat[-1]
        if self.lat_flip:
            self.src_lat = self.src_lat[::-1]
        
        # Handle longitude wrapping
        dst_lon_adj = np.where(dst_lon > 180, dst_lon - 360, dst_lon)
        src_lon_adj = np.where(self.src_lon > 180, self.src_lon - 360, self.src_lon)
        
        self.lon_sort = np.argsort(src_lon_adj)
        self.src_lon_sorted = src_lon_adj[self.lon_sort]
        self.points = np.column_stack([dst_lat.ravel(), dst_lon_adj.ravel()])
    
    def __call__(self, src_data):
        """Interpolate 2D field."""
        data = np.asarray(src_data)
        if self.lat_flip:
            data = data[::-1, :]
        data = data[:, self.lon_sort]
        
        # Fill NaN with nearest neighbor mean
        if np.any(np.isnan(data)):
            mean_val = np.nanmean(data)
            if np.isnan(mean_val):
                mean_val = 0.0
            data = np.nan_to_num(data, nan=mean_val)
        
        interp = RegularGridInterpolator(
            (self.src_lat, self.src_lon_sorted), data,
            method='linear', bounds_error=False, fill_value=None
        )
        result = interp(self.points).reshape(self.dst_shape)
        return result


def interpolate_3d_field(src_data, src_depth, src_lon, src_lat, 
                          dst_lon, dst_lat, dst_z, lat_flip=False, lon_sort=None):
    """
    Interpolate 3D field from source grid to ROMS grid.
    
    src_data: (depth, lat, lon) - MUST HAVE BAD VALUES AS NaN
    src_depth: (depth,) - positive down
    dst_z: (N, eta, xi) - negative (depth below surface)
    """
    n_src_depth, n_lat, n_lon = src_data.shape
    N, eta, xi = dst_z.shape
    
    log(f"    3D interpolation: ({n_src_depth}, {n_lat}, {n_lon}) -> ({N}, {eta}, {xi})")
    
    # Prepare source coordinates
    src_lat_arr = np.asarray(src_lat).flatten()
    src_lon_arr = np.asarray(src_lon).flatten()
    
    if lat_flip:
        src_lat_arr = src_lat_arr[::-1]
        src_data = src_data[:, ::-1, :]
    
    if lon_sort is not None:
        src_lon_arr = src_lon_arr[lon_sort]
        src_data = src_data[:, :, lon_sort]
    
    # Handle longitude adjustment
    dst_lon_adj = np.where(dst_lon > 180, dst_lon - 360, dst_lon)
    
    # First: Interpolate horizontally at each source depth level
    log(f"    Horizontal interpolation at {n_src_depth} depth levels...")
    src_on_dst = np.zeros((n_src_depth, eta, xi), dtype=np.float32)
    
    for k in range(n_src_depth):
        data_k = src_data[k]
        
        # Fill NaN with mean
        if np.any(np.isnan(data_k)):
            mean_k = np.nanmean(data_k)
            if np.isnan(mean_k):
                mean_k = 0.0
            data_k = np.nan_to_num(data_k, nan=mean_k)
        
        interp = RegularGridInterpolator(
            (src_lat_arr, src_lon_arr), data_k,
            method='linear', bounds_error=False, fill_value=None
        )
        points = np.column_stack([dst_lat.ravel(), dst_lon_adj.ravel()])
        src_on_dst[k] = interp(points).reshape(eta, xi)
    
    # Second: Interpolate vertically to ROMS levels
    log(f"    Vertical interpolation to {N} ROMS levels...")
    result = np.zeros((N, eta, xi), dtype=np.float32)
    
    # Convert ROMS z (negative) to positive depth
    dst_depth = -dst_z
    
    # For each column, interpolate vertically
    for j in range(eta):
        if j % 50 == 0:
            log(f"      Row {j}/{eta}...")
        for i in range(xi):
            col = src_on_dst[:, j, i]
            target_depths = dst_depth[:, j, i]
            
            # Skip if all NaN
            if np.all(np.isnan(col)):
                result[:, j, i] = np.nan
                continue
            
            # Find valid depth levels
            valid = ~np.isnan(col)
            if np.sum(valid) < 2:
                result[:, j, i] = col[valid][0] if np.any(valid) else np.nan
                continue
            
            # Vertical interpolation with extrapolation
            try:
                f = interp1d(
                    src_depth[valid], col[valid],
                    kind='linear', 
                    bounds_error=False,
                    fill_value=(col[valid][0], col[valid][-1])
                )
                result[:, j, i] = f(target_depths)
            except:
                result[:, j, i] = col[valid][0]
    
    return result


def validate_field(data, name, min_val, max_val, ocean_mask=None):
    """Validate a field for bad values."""
    issues = []
    
    # Check for NaN, Inf
    n_nan = np.sum(np.isnan(data))
    n_inf = np.sum(np.isinf(data))
    
    if n_nan > 0:
        issues.append(f"NaN: {n_nan:,}")
    if n_inf > 0:
        issues.append(f"Inf: {n_inf:,}")
    
    # Check for extreme values
    n_extreme = np.sum(np.abs(data) > 1e6)
    if n_extreme > 0:
        issues.append(f"Extreme (>1e6): {n_extreme:,}")
    
    # Check range
    actual_min = np.nanmin(data)
    actual_max = np.nanmax(data)
    
    if actual_min < min_val or actual_max > max_val:
        issues.append(f"Out of range [{min_val}, {max_val}]: [{actual_min:.3f}, {actual_max:.3f}]")
    
    # Check ocean points if mask provided
    if ocean_mask is not None and len(data.shape) == 2:
        ocean_data = data[ocean_mask]
        n_bad_ocean = np.sum((np.abs(ocean_data) > 1e6) | np.isnan(ocean_data))
        if n_bad_ocean > 0:
            issues.append(f"Bad values in ocean: {n_bad_ocean:,}")
    
    if issues:
        log(f"  ✗ {name}: {', '.join(issues)}")
        return False
    else:
        log(f"  ✓ {name}: OK [{actual_min:.3f}, {actual_max:.3f}]")
        return True


def main():
    log("=" * 70)
    log("ROMS Initial Conditions Generator - IMPROVED VERSION")
    log("=" * 70)
    
    # Read ROMS grid
    log(f"\nReading ROMS grid: {GRID_FILE}")
    try:
        with nc.Dataset(GRID_FILE, 'r') as grd:
            lon_rho = grd.variables['lon_rho'][:]
            lat_rho = grd.variables['lat_rho'][:]
            h = grd.variables['h'][:]
            mask_rho = grd.variables['mask_rho'][:]
    except FileNotFoundError:
        log(f"ERROR: Grid file not found: {GRID_FILE}")
        sys.exit(1)
    
    eta_rho, xi_rho = lon_rho.shape
    xi_u = xi_rho - 1
    eta_v = eta_rho - 1
    log(f"  Grid: {eta_rho} x {xi_rho}, N={N_LEVELS}")
    log(f"  Bathymetry: {h.min():.1f} to {h.max():.1f} m")
    log(f"  Ocean points: {np.sum(mask_rho == 1):,}")
    log(f"  Land points: {np.sum(mask_rho == 0):,}")
    
    # Compute S-coordinates
    log("\nComputing S-coordinates...")
    s_rho, Cs_r = compute_s_levels(N_LEVELS, THETA_S, THETA_B, VSTRETCHING)
    z_rho = compute_z_levels(s_rho, Cs_r, h, VTRANSFORM, TCLINE)
    log(f"  S-coord: {s_rho.min():.4f} to {s_rho.max():.4f}")
    log(f"  Z range: {z_rho.min():.1f} to {z_rho.max():.1f} m")
    
    # Read CMEMS data WITH IMPROVED FILL VALUE HANDLING
    log(f"\nReading CMEMS: {CMEMS_FILE}")
    try:
        with nc.Dataset(CMEMS_FILE, 'r') as src:
            src_lon = src.variables['longitude'][:]
            src_lat = src.variables['latitude'][:]
            src_depth = src.variables['depth'][:]
            
            log(f"  Grid: {len(src_lat)} x {len(src_lon)}")
            log(f"  Depth: {len(src_depth)} levels ({src_depth[0]:.1f} to {src_depth[-1]:.1f} m)")
            
            # Read 3D variables with improved fill value handling
            log("  Reading temperature (3D)...")
            thetao = read_var_safe(src.variables['thetao'][0], 'thetao')
            
            log("  Reading salinity (3D)...")
            so = read_var_safe(src.variables['so'][0], 'so')
            
            log("  Reading u-velocity (3D)...")
            uo = read_var_safe(src.variables['uo'][0], 'uo')
            
            log("  Reading v-velocity (3D)...")
            vo = read_var_safe(src.variables['vo'][0], 'vo')
            
            # Read 2D variables
            log("  Reading SSH...")
            zos = read_var_safe(src.variables['zos'][0], 'zos')
            
            # Ice variables (may not exist)
            log("  Reading ice variables...")
            try:
                siconc = read_var_safe(src.variables['siconc'][0], 'siconc')
                sithick = read_var_safe(src.variables['sithick'][0], 'sithick')
                usi = read_var_safe(src.variables['usi'][0], 'usi')
                vsi = read_var_safe(src.variables['vsi'][0], 'vsi')
            except KeyError:
                log("    Ice variables not found, using zeros")
                siconc = np.zeros_like(zos)
                sithick = np.zeros_like(zos)
                usi = np.zeros_like(zos)
                vsi = np.zeros_like(zos)
                
    except FileNotFoundError:
        log(f"ERROR: CMEMS file not found: {CMEMS_FILE}")
        sys.exit(1)
    
    # Setup interpolation
    log("\nSetting up interpolation...")
    lat_flip = src_lat[0] > src_lat[-1]
    if lat_flip:
        log("  Flipping latitude (N->S to S->N)")
        src_lat_sorted = src_lat[::-1]
    else:
        src_lat_sorted = src_lat
    
    src_lon_adj = np.where(src_lon > 180, src_lon - 360, src_lon)
    lon_sort = np.argsort(src_lon_adj)
    src_lon_sorted = src_lon_adj[lon_sort]
    
    interp_2d = FastInterpolator(src_lon, src_lat, lon_rho, lat_rho)
    
    # Interpolate 3D fields
    log("\nInterpolating 3D fields...")
    log("  Temperature...")
    temp = interpolate_3d_field(
        thetao, src_depth, src_lon_sorted, src_lat_sorted,
        lon_rho, lat_rho, z_rho, lat_flip=lat_flip, lon_sort=lon_sort
    )
    
    log("  Salinity...")
    salt = interpolate_3d_field(
        so, src_depth, src_lon_sorted, src_lat_sorted,
        lon_rho, lat_rho, z_rho, lat_flip=lat_flip, lon_sort=lon_sort
    )
    
    log("  U-velocity...")
    u_3d = interpolate_3d_field(
        uo, src_depth, src_lon_sorted, src_lat_sorted,
        lon_rho, lat_rho, z_rho, lat_flip=lat_flip, lon_sort=lon_sort
    )
    
    log("  V-velocity...")
    v_3d = interpolate_3d_field(
        vo, src_depth, src_lon_sorted, src_lat_sorted,
        lon_rho, lat_rho, z_rho, lat_flip=lat_flip, lon_sort=lon_sort
    )
    
    # Interpolate 2D fields
    log("\nInterpolating 2D fields...")
    zeta = interp_2d(zos)
    aice = interp_2d(siconc)
    hice = interp_2d(sithick)
    uice = interp_2d(usi)
    vice = interp_2d(vsi)
    
    # Compute barotropic velocities (FIXED: depth-weighted integral)
    log("\nComputing barotropic velocities...")
    # Calculate layer thicknesses
    dz = np.abs(np.diff(z_rho, axis=0))
    dz = np.vstack([dz, dz[-1:]])  # Pad to N levels
    H = np.sum(dz, axis=0)
    H = np.where(H == 0, 1, H)  # Avoid division by zero
    # Depth-weighted barotropic velocities
    ubar = np.nansum(u_3d * dz, axis=0) / H
    vbar = np.nansum(v_3d * dz, axis=0) / H
    
    # Fill and mask
    log("\nFilling NaN and applying land mask...")
    land = mask_rho == 0
    ocean = mask_rho == 1
    
    def fill_and_mask_3d(data, default_ocean, land_mask):
        """Fill NaN and mask land"""
        result = np.copy(data)
        result = np.where(np.isnan(result), default_ocean, result)
        result = np.where(np.abs(result) > 1e6, default_ocean, result)
        for k in range(result.shape[0]):
            result[k, land_mask] = 0.0
        return result
    
    def fill_and_mask_2d(data, default_ocean, land_mask):
        """Fill NaN and mask land"""
        result = np.copy(data)
        result = np.where(np.isnan(result), default_ocean, result)
        result = np.where(np.abs(result) > 1e6, default_ocean, result)
        result[land_mask] = 0.0
        return result
    
    temp = fill_and_mask_3d(temp, 5.0, land)
    salt = fill_and_mask_3d(salt, 34.5, land)
    u_3d = fill_and_mask_3d(u_3d, 0.0, land)
    v_3d = fill_and_mask_3d(v_3d, 0.0, land)
    
    ubar = fill_and_mask_2d(ubar, 0.0, land)
    vbar = fill_and_mask_2d(vbar, 0.0, land)
    zeta = fill_and_mask_2d(zeta, 0.0, land)
    aice = fill_and_mask_2d(aice, 0.0, land)
    hice = fill_and_mask_2d(hice, 0.0, land)
    uice = fill_and_mask_2d(uice, 0.0, land)
    vice = fill_and_mask_2d(vice, 0.0, land)
    
    # Clip to physical ranges
    log("Clipping to physical ranges...")
    temp = np.clip(temp, -3.0, 35.0)
    salt = np.clip(salt, 25.0, 42.0)
    aice = np.clip(aice, 0.0, 1.0)
    hice = np.clip(hice, 0.0, 20.0)
    
    # Validate before writing
    log("\nValidating fields...")
    all_valid = True
    all_valid &= validate_field(temp[0], "temp", -5, 40, ocean)
    all_valid &= validate_field(salt[0], "salt", 20, 45, ocean)
    all_valid &= validate_field(zeta, "zeta", -10, 10, ocean)
    all_valid &= validate_field(ubar, "ubar", -5, 5, ocean)
    all_valid &= validate_field(vbar, "vbar", -5, 5, ocean)
    
    if not all_valid:
        log("\n✗ VALIDATION FAILED - Stopping")
        sys.exit(1)
    
    # Interpolate to U/V grids
    log("\nInterpolating to U/V points...")
    u_on_u = 0.5 * (u_3d[:, :, :-1] + u_3d[:, :, 1:])
    v_on_v = 0.5 * (v_3d[:, :-1, :] + v_3d[:, 1:, :])
    ubar_on_u = 0.5 * (ubar[:, :-1] + ubar[:, 1:])
    vbar_on_v = 0.5 * (vbar[:-1, :] + vbar[1:, :])
    uice_on_u = 0.5 * (uice[:, :-1] + uice[:, 1:])
    vice_on_v = 0.5 * (vice[:-1, :] + vice[1:, :])
    
    # Write output
    log(f"\nWriting output: {OUTPUT_FILE}")
    with nc.Dataset(OUTPUT_FILE, 'w', format='NETCDF4') as out:
        # Dimensions
        out.createDimension('xi_rho', xi_rho)
        out.createDimension('eta_rho', eta_rho)
        out.createDimension('xi_u', xi_u)
        out.createDimension('eta_v', eta_v)
        out.createDimension('s_rho', N_LEVELS)
        out.createDimension('s_w', N_LEVELS + 1)
        out.createDimension('ocean_time', None)
        
        # Time
        tvar = out.createVariable('ocean_time', 'f8', ('ocean_time',))
        tvar.long_name = 'time since initialization'
        tvar.units = 'days since 1900-01-01 00:00:00'
        tvar[0] = INIT_TIME
        
        # S-coordinates
        svar = out.createVariable('s_rho', 'f8', ('s_rho',))
        svar.long_name = 's-coordinate at RHO-points'
        svar[:] = s_rho
        
        swvar = out.createVariable('s_w', 'f8', ('s_w',))
        swvar.long_name = 's-coordinate at W-points'
        s_w = (np.arange(N_LEVELS + 1) - N_LEVELS) / N_LEVELS
        swvar[:] = s_w
        
        csvar = out.createVariable('Cs_r', 'f8', ('s_rho',))
        csvar.long_name = 'S-coordinate stretching at RHO-points'
        csvar[:] = Cs_r
        
        # Temperature
        var = out.createVariable('temp', 'f4', ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'potential temperature'
        var.units = 'Celsius'
        var[0] = temp
        
        # Salinity
        var = out.createVariable('salt', 'f4', ('ocean_time', 's_rho', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'salinity'
        var.units = 'PSU'
        var[0] = salt
        
        # U velocity
        var = out.createVariable('u', 'f4', ('ocean_time', 's_rho', 'eta_rho', 'xi_u'),
                                 zlib=True, complevel=4)
        var.long_name = 'u-momentum component'
        var.units = 'm/s'
        var[0] = u_on_u
        
        # V velocity
        var = out.createVariable('v', 'f4', ('ocean_time', 's_rho', 'eta_v', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'v-momentum component'
        var.units = 'm/s'
        var[0] = v_on_v
        
        # Ubar
        var = out.createVariable('ubar', 'f4', ('ocean_time', 'eta_rho', 'xi_u'),
                                 zlib=True, complevel=4)
        var.long_name = 'vertically integrated u-momentum'
        var.units = 'm/s'
        var[0] = ubar_on_u
        
        # Vbar
        var = out.createVariable('vbar', 'f4', ('ocean_time', 'eta_v', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'vertically integrated v-momentum'
        var.units = 'm/s'
        var[0] = vbar_on_v
        
        # Zeta
        var = out.createVariable('zeta', 'f4', ('ocean_time', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'free-surface'
        var.units = 'meter'
        var[0] = zeta
        
        # Ice concentration
        var = out.createVariable('aice', 'f4', ('ocean_time', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'sea ice concentration'
        var.units = 'fraction'
        var[0] = aice
        
        # Ice thickness
        var = out.createVariable('hice', 'f4', ('ocean_time', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'sea ice thickness'
        var.units = 'meter'
        var[0] = hice
        
        # Ice U velocity
        var = out.createVariable('uice', 'f4', ('ocean_time', 'eta_rho', 'xi_u'),
                                 zlib=True, complevel=4)
        var.long_name = 'sea ice u-velocity'
        var.units = 'm/s'
        var[0] = uice_on_u
        
        # Ice V velocity
        var = out.createVariable('vice', 'f4', ('ocean_time', 'eta_v', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = 'sea ice v-velocity'
        var.units = 'm/s'
        var[0] = vice_on_v
        
        # Global attributes
        out.title = 'ROMS initial conditions for Southern Ocean'
        out.history = f'Created {datetime.now().isoformat()} - IMPROVED FILL VALUE HANDLING'
        out.source = f'CMEMS: {os.path.basename(CMEMS_FILE)}'
        out.grid_file = os.path.basename(GRID_FILE)
        out.Vtransform = VTRANSFORM
        out.Vstretching = VSTRETCHING
        out.theta_s = THETA_S
        out.theta_b = THETA_B
        out.Tcline = TCLINE
    
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    log(f"\n✓ Output: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    
    log("\n" + "=" * 70)
    log("✓ DONE!")
    log("=" * 70)
    log("\nNext steps:")
    log(f"1. Verify output with: ncdump -h {OUTPUT_FILE}")
    log(f"2. Check ranges match expectations")
    log(f"3. Test ROMS with short run (ntimes=100)")
    log("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)