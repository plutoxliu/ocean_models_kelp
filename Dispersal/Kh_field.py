#!/usr/bin/env python3
"""
Generate Spatially-Varying Horizontal Diffusivity (Kh) Field from EKE
=====================================================================

Calculates Kh based on Eddy Kinetic Energy (EKE) from velocity variance.
Output can be used in Parcels for more realistic particle dispersion.

MEMORY-EFFICIENT VERSION: Processes data in chunks.

Methods:
1. EKE = 0.5 * (u'^2 + v'^2) where u' = u - u_mean
2. Kh = gamma * sqrt(2*EKE) * L_mix  (mixing length formulation)

References:
- Smagorinsky (1963) for sub-grid diffusivity
- Klocker & Abernathey (2014) for Southern Ocean eddy diffusivity
"""

import numpy as np
import netCDF4 as nc
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

CURRENTS_FILE = './roms_forcing_fine/roms_frc_currents.nc'
GRID_FILE = './ocean_grd_fine.nc'
OUTPUT_FILE = './roms_forcing_fine/Kh_field.nc'

# Diffusivity parameters
GAMMA = 0.15  # Mixing efficiency coefficient (0.1-0.2 typical)
L_MIX_KM = 30.0  # Mixing length scale in km (typically 20-50 km for mesoscale)
KH_MIN = 10.0  # Minimum Kh (m²/s) - prevents zero diffusivity
KH_MAX = 2000.0  # Maximum Kh (m²/s) - caps extreme values

# Chunk size for memory-efficient processing
CHUNK_SIZE = 100  # Number of timesteps to process at once

# =============================================================================
# MAIN
# =============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("="*70)
    log("GENERATING SPATIALLY-VARYING Kh FIELD FROM EKE")
    log("(Memory-efficient chunked version)")
    log("="*70)
    
    # Load grid
    log(f"\nLoading grid: {GRID_FILE}")
    with nc.Dataset(GRID_FILE, 'r') as f:
        lon_rho = f.variables['lon_rho'][:]
        lat_rho = f.variables['lat_rho'][:]
        mask_rho = f.variables['mask_rho'][:]
    
    eta_rho, xi_rho = lon_rho.shape
    log(f"  Grid shape: {lon_rho.shape}")
    log(f"  Lon range: {lon_rho.min():.1f} to {lon_rho.max():.1f}")
    log(f"  Lat range: {lat_rho.min():.1f} to {lat_rho.max():.1f}")
    
    # Get dimensions from currents file
    log(f"\nOpening currents file: {CURRENTS_FILE}")
    with nc.Dataset(CURRENTS_FILE, 'r') as f:
        n_times = len(f.dimensions['ocean_time'])
        ubar_shape = f.variables['ubar'].shape
        vbar_shape = f.variables['vbar'].shape
        ocean_time = f.variables['ocean_time'][0]  # Just get first time for output
    
    log(f"  Total timesteps: {n_times}")
    log(f"  ubar shape: {ubar_shape}")
    log(f"  vbar shape: {vbar_shape}")
    
    # Determine if we need to interpolate to rho grid
    need_interp_u = (ubar_shape[2] != xi_rho)
    need_interp_v = (vbar_shape[1] != eta_rho)
    
    if need_interp_u:
        log("  Will interpolate ubar from u-grid to rho-grid")
    if need_interp_v:
        log("  Will interpolate vbar from v-grid to rho-grid")
    
    # First pass: Calculate mean velocities (chunked)
    log("\nPass 1: Calculating mean velocities (chunked)...")
    
    u_sum = np.zeros((eta_rho, xi_rho), dtype=np.float64)
    v_sum = np.zeros((eta_rho, xi_rho), dtype=np.float64)
    n_valid = np.zeros((eta_rho, xi_rho), dtype=np.int32)
    
    n_chunks = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    with nc.Dataset(CURRENTS_FILE, 'r') as f:
        for chunk_idx in range(n_chunks):
            t_start = chunk_idx * CHUNK_SIZE
            t_end = min((chunk_idx + 1) * CHUNK_SIZE, n_times)
            
            log(f"  Chunk {chunk_idx+1}/{n_chunks}: timesteps {t_start}-{t_end-1}")
            
            # Read chunk
            ubar_chunk = f.variables['ubar'][t_start:t_end, :, :]
            vbar_chunk = f.variables['vbar'][t_start:t_end, :, :]
            
            # Handle masked arrays
            if np.ma.isMaskedArray(ubar_chunk):
                ubar_chunk = np.ma.filled(ubar_chunk.astype(np.float64), np.nan)
            else:
                ubar_chunk = ubar_chunk.astype(np.float64)
            if np.ma.isMaskedArray(vbar_chunk):
                vbar_chunk = np.ma.filled(vbar_chunk.astype(np.float64), np.nan)
            else:
                vbar_chunk = vbar_chunk.astype(np.float64)
            
            # Interpolate to rho grid if needed
            if need_interp_u:
                ubar_rho = np.zeros((t_end - t_start, eta_rho, xi_rho), dtype=np.float64)
                ubar_rho[:, :, 1:-1] = 0.5 * (ubar_chunk[:, :, :-1] + ubar_chunk[:, :, 1:])
                ubar_rho[:, :, 0] = ubar_chunk[:, :, 0]
                ubar_rho[:, :, -1] = ubar_chunk[:, :, -1]
                ubar_chunk = ubar_rho
            
            if need_interp_v:
                vbar_rho = np.zeros((t_end - t_start, eta_rho, xi_rho), dtype=np.float64)
                vbar_rho[:, 1:-1, :] = 0.5 * (vbar_chunk[:, :-1, :] + vbar_chunk[:, 1:, :])
                vbar_rho[:, 0, :] = vbar_chunk[:, 0, :]
                vbar_rho[:, -1, :] = vbar_chunk[:, -1, :]
                vbar_chunk = vbar_rho
            
            # Accumulate sums (ignoring NaN)
            for t in range(ubar_chunk.shape[0]):
                valid_mask = ~np.isnan(ubar_chunk[t]) & ~np.isnan(vbar_chunk[t])
                u_sum[valid_mask] += ubar_chunk[t, valid_mask]
                v_sum[valid_mask] += vbar_chunk[t, valid_mask]
                n_valid[valid_mask] += 1
    
    # Calculate means
    n_valid = np.where(n_valid == 0, 1, n_valid)  # Avoid division by zero
    u_mean = u_sum / n_valid
    v_mean = v_sum / n_valid
    
    log(f"  Mean u range: {np.nanmin(u_mean):.4f} to {np.nanmax(u_mean):.4f} m/s")
    log(f"  Mean v range: {np.nanmin(v_mean):.4f} to {np.nanmax(v_mean):.4f} m/s")
    
    # Second pass: Calculate variance (EKE)
    log("\nPass 2: Calculating EKE (chunked)...")
    
    u_var_sum = np.zeros((eta_rho, xi_rho), dtype=np.float64)
    v_var_sum = np.zeros((eta_rho, xi_rho), dtype=np.float64)
    n_valid2 = np.zeros((eta_rho, xi_rho), dtype=np.int32)
    
    with nc.Dataset(CURRENTS_FILE, 'r') as f:
        for chunk_idx in range(n_chunks):
            t_start = chunk_idx * CHUNK_SIZE
            t_end = min((chunk_idx + 1) * CHUNK_SIZE, n_times)
            
            log(f"  Chunk {chunk_idx+1}/{n_chunks}: timesteps {t_start}-{t_end-1}")
            
            # Read chunk
            ubar_chunk = f.variables['ubar'][t_start:t_end, :, :]
            vbar_chunk = f.variables['vbar'][t_start:t_end, :, :]
            
            # Handle masked arrays
            if np.ma.isMaskedArray(ubar_chunk):
                ubar_chunk = np.ma.filled(ubar_chunk.astype(np.float64), np.nan)
            else:
                ubar_chunk = ubar_chunk.astype(np.float64)
            if np.ma.isMaskedArray(vbar_chunk):
                vbar_chunk = np.ma.filled(vbar_chunk.astype(np.float64), np.nan)
            else:
                vbar_chunk = vbar_chunk.astype(np.float64)
            
            # Interpolate to rho grid if needed
            if need_interp_u:
                ubar_rho = np.zeros((t_end - t_start, eta_rho, xi_rho), dtype=np.float64)
                ubar_rho[:, :, 1:-1] = 0.5 * (ubar_chunk[:, :, :-1] + ubar_chunk[:, :, 1:])
                ubar_rho[:, :, 0] = ubar_chunk[:, :, 0]
                ubar_rho[:, :, -1] = ubar_chunk[:, :, -1]
                ubar_chunk = ubar_rho
            
            if need_interp_v:
                vbar_rho = np.zeros((t_end - t_start, eta_rho, xi_rho), dtype=np.float64)
                vbar_rho[:, 1:-1, :] = 0.5 * (vbar_chunk[:, :-1, :] + vbar_chunk[:, 1:, :])
                vbar_rho[:, 0, :] = vbar_chunk[:, 0, :]
                vbar_rho[:, -1, :] = vbar_chunk[:, -1, :]
                vbar_chunk = vbar_rho
            
            # Calculate variance contribution
            for t in range(ubar_chunk.shape[0]):
                valid_mask = ~np.isnan(ubar_chunk[t]) & ~np.isnan(vbar_chunk[t])
                u_prime = ubar_chunk[t] - u_mean
                v_prime = vbar_chunk[t] - v_mean
                u_var_sum[valid_mask] += (u_prime[valid_mask] ** 2)
                v_var_sum[valid_mask] += (v_prime[valid_mask] ** 2)
                n_valid2[valid_mask] += 1
    
    # Calculate EKE
    n_valid2 = np.where(n_valid2 == 0, 1, n_valid2)
    u_var = u_var_sum / n_valid2
    v_var = v_var_sum / n_valid2
    EKE = 0.5 * (u_var + v_var)
    
    log(f"\n  EKE range: {np.nanmin(EKE):.6f} to {np.nanmax(EKE):.6f} m²/s²")
    log(f"  EKE mean: {np.nanmean(EKE):.6f} m²/s²")
    
    # Calculate eddy velocity scale: V_eddy = sqrt(2 * EKE)
    V_eddy = np.sqrt(2 * EKE)
    log(f"  Eddy velocity scale range: {np.nanmin(V_eddy):.4f} to {np.nanmax(V_eddy):.4f} m/s")
    
    # Calculate Kh using mixing length formulation
    log(f"\nCalculating Kh (gamma={GAMMA}, L_mix={L_MIX_KM} km)...")
    L_mix_m = L_MIX_KM * 1000.0
    Kh = GAMMA * V_eddy * L_mix_m
    
    # Apply min/max bounds
    Kh = np.clip(Kh, KH_MIN, KH_MAX)
    
    # Apply land mask
    Kh = np.where(mask_rho == 0, 0.0, Kh)
    
    log(f"  Kh range (clipped): {np.nanmin(Kh[mask_rho==1]):.1f} to {np.nanmax(Kh):.1f} m²/s")
    log(f"  Kh mean: {np.nanmean(Kh[mask_rho==1]):.1f} m²/s")
    
    # Statistics by latitude band
    log("\nKh by latitude band:")
    for lat_target in [-40, -50, -60, -70]:
        lat_mask = (lat_rho > lat_target - 2.5) & (lat_rho < lat_target + 2.5) & (mask_rho == 1)
        if np.any(lat_mask):
            kh_band = Kh[lat_mask]
            log(f"  {lat_target}°S: mean={np.mean(kh_band):.1f}, "
                f"min={np.min(kh_band):.1f}, max={np.max(kh_band):.1f} m²/s")
    
    # Save to NetCDF
    log(f"\nSaving to: {OUTPUT_FILE}")
    
    with nc.Dataset(OUTPUT_FILE, 'w', format='NETCDF4') as out:
        # Dimensions
        out.createDimension('eta_rho', eta_rho)
        out.createDimension('xi_rho', xi_rho)
        out.createDimension('time', None)
        
        # Coordinates
        lon_var = out.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))
        lon_var.long_name = 'longitude of rho-points'
        lon_var.units = 'degrees_east'
        lon_var[:] = lon_rho
        
        lat_var = out.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))
        lat_var.long_name = 'latitude of rho-points'
        lat_var.units = 'degrees_north'
        lat_var[:] = lat_rho
        
        # Time
        time_var = out.createVariable('kh_time', 'f8', ('time',))
        time_var.long_name = 'time'
        time_var.units = 'days since 1900-01-01 00:00:00'
        time_var[0] = ocean_time
        
        # Kh field
        kh_var = out.createVariable('Kh', 'f4', ('time', 'eta_rho', 'xi_rho'),
                                    zlib=True, complevel=4)
        kh_var.long_name = 'horizontal diffusivity from EKE'
        kh_var.units = 'm2/s'
        kh_var.coordinates = 'lon_rho lat_rho'
        kh_var[0, :, :] = Kh.astype(np.float32)
        
        # EKE for reference
        eke_var = out.createVariable('EKE', 'f4', ('eta_rho', 'xi_rho'),
                                     zlib=True, complevel=4)
        eke_var.long_name = 'eddy kinetic energy'
        eke_var.units = 'm2/s2'
        eke_var[:] = EKE.astype(np.float32)
        
        # Global attributes
        out.title = 'Spatially-varying horizontal diffusivity from EKE'
        out.history = f'Created {datetime.now().isoformat()}'
        out.source = f'Calculated from {os.path.basename(CURRENTS_FILE)}'
        out.gamma = GAMMA
        out.L_mix_km = L_MIX_KM
        out.Kh_min = KH_MIN
        out.Kh_max = KH_MAX
    
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    log(f"  File size: {size_mb:.2f} MB")
    
    log("\n" + "="*70)
    log("✓ DONE!")
    log("="*70)


if __name__ == '__main__':
    main()