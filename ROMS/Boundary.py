#!/usr/bin/env python3
"""LOW MEMORY boundary generator - processes one timestep at a time"""

import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
from datetime import datetime
import os

# EDIT PATHS
CMEMS_BD1 = './cmems_data/cmems_glo_phy_P1D_BD1.nc'
CMEMS_BD2 = './cmems_data/cmems_glo_phy_P1D_BD2.nc'
GRID_FILE = 'ocean_grd_fine.nc'
OUTPUT_BRY = 'roms_bry_fine.nc'

BOUNDARY_LAT = -30.0
THETA_S = 7.0
THETA_B = 2.0
TCLINE = 200.0
N_LEVELS = 30
MAX_DEPTH = 200.0

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def clean_array(arr):
    """Convert masked array to regular array."""
    if hasattr(arr, 'mask'):
        arr = np.ma.filled(arr, np.nan)
    return np.asarray(arr, dtype=np.float64)

log("="*70)
log("LOW MEMORY BOUNDARY GENERATOR")
log("="*70)

# Read grid
log("\n1. Reading grid...")
with nc.Dataset(GRID_FILE, 'r') as grd:
    lon_rho = grd.variables['lon_rho'][:]
    h = grd.variables['h'][:]
    eta_rho, xi_rho = lon_rho.shape

bry_eta_idx = eta_rho - 1
lon_bry = clean_array(lon_rho[bry_eta_idx, :])
lon_bry_adj = (lon_bry + 180) % 360 - 180 #Normalize lon_bry to -180 to 180 to match CMEMS

h_bry = clean_array(h[bry_eta_idx, :])
h_bry = np.minimum(h_bry, MAX_DEPTH)

log(f"  Grid: {xi_rho} points")
log(f"  Bathymetry: {h_bry.min():.0f}-{h_bry.max():.0f}m")

# Calculate vertical coordinates
hc = min(TCLINE, np.nanmin(h_bry[h_bry > 0]))
N = N_LEVELS
s_rho = (np.arange(1, N+1) - N - 0.5) / N

Cs = (1 - np.cosh(THETA_S * s_rho)) / (np.cosh(THETA_S) - 1)
Cs = (np.exp(THETA_B * Cs) - 1) / (1 - np.exp(-THETA_B))

z_rho = np.zeros((N, 1, xi_rho))
for i in range(xi_rho):
    for k in range(N):
        z0 = (hc * s_rho[k] + h_bry[i] * Cs[k]) / (hc + h_bry[i])
        z_rho[k, 0, i] = h_bry[i] * z0

depths_roms = -z_rho
log(f"  Depths: {z_rho[0,0,xi_rho//2]:.1f}m (bottom) to {z_rho[-1,0,xi_rho//2]:.1f}m (surface)")

# Open CMEMS
log("\n2. Opening CMEMS files...")
ds_bd1 = nc.Dataset(CMEMS_BD1, 'r')
ds_bd2 = nc.Dataset(CMEMS_BD2, 'r')

time_cmems = ds_bd2.variables['time'][:]
time_units = ds_bd2.variables['time'].units

# Convert time
if 'hours since' in time_units.lower():
    ref = datetime.strptime(time_units.split('since')[1].strip(), '%Y-%m-%d')
    days_1900 = (ref - datetime(1900,1,1)).days
    time_roms = time_cmems / 24.0 + days_1900
else:
    time_roms = time_cmems

lon_cmems = clean_array(ds_bd2.variables['longitude'][:])
lat_cmems = clean_array(ds_bd2.variables['latitude'][:])
depth_cmems = clean_array(ds_bd2.variables['depth'][:])

bry_idx = np.argmin(np.abs(lat_cmems - BOUNDARY_LAT))
nt = len(time_roms)

log(f"  Timesteps: {nt}")
log(f"  Boundary at: {lat_cmems[bry_idx]:.2f}°S")

# Create output with ALL required time variables
log("\n3. Creating output file...")
out = nc.Dataset(OUTPUT_BRY, 'w', format='NETCDF4')

# Create dimensions - ROMS needs separate time dimensions for each variable type
out.createDimension('xi_rho', xi_rho)
out.createDimension('s_rho', N)
out.createDimension('ocean_time', None)
out.createDimension('zeta_time', None)
out.createDimension('v2d_time', None)
out.createDimension('v3d_time', None)
out.createDimension('temp_time', None)
out.createDimension('salt_time', None)

time_units_str = 'days since 1900-01-01 00:00:00'

# Create ALL time variables that ROMS expects
ocean_time = out.createVariable('ocean_time', 'f8', ('ocean_time',))
ocean_time.units = time_units_str
ocean_time.long_name = 'time for fields at RHO points'
ocean_time.calendar = 'gregorian'

zeta_time = out.createVariable('zeta_time', 'f8', ('zeta_time',))
zeta_time.units = time_units_str
zeta_time.long_name = 'time for sea surface height'
zeta_time.calendar = 'gregorian'

v2d_time = out.createVariable('v2d_time', 'f8', ('v2d_time',))
v2d_time.units = time_units_str
v2d_time.long_name = 'time for 2D velocity'
v2d_time.calendar = 'gregorian'

v3d_time = out.createVariable('v3d_time', 'f8', ('v3d_time',))
v3d_time.units = time_units_str
v3d_time.long_name = 'time for 3D velocity'
v3d_time.calendar = 'gregorian'

temp_time = out.createVariable('temp_time', 'f8', ('temp_time',))
temp_time.units = time_units_str
temp_time.long_name = 'time for potential temperature'
temp_time.calendar = 'gregorian'

salt_time = out.createVariable('salt_time', 'f8', ('salt_time',))
salt_time.units = time_units_str
salt_time.long_name = 'time for salinity'
salt_time.calendar = 'gregorian'

# Create data variables with correct time dimension
temp_var = out.createVariable('temp_north', 'f4', ('temp_time','s_rho','xi_rho'), zlib=True, complevel=4)
temp_var.long_name = 'potential temperature northern boundary'
temp_var.units = 'Celsius'

salt_var = out.createVariable('salt_north', 'f4', ('salt_time','s_rho','xi_rho'), zlib=True, complevel=4)
salt_var.long_name = 'salinity northern boundary'
salt_var.units = 'PSU'

u_var = out.createVariable('u_north', 'f4', ('v3d_time','s_rho','xi_rho'), zlib=True, complevel=4)
u_var.long_name = 'u-velocity component northern boundary'
u_var.units = 'm/s'

v_var = out.createVariable('v_north', 'f4', ('v3d_time','s_rho','xi_rho'), zlib=True, complevel=4)
v_var.long_name = 'v-velocity component northern boundary'
v_var.units = 'm/s'

ubar_var = out.createVariable('ubar_north', 'f4', ('v2d_time','xi_rho'), zlib=True, complevel=4)
ubar_var.long_name = 'barotropic u-velocity northern boundary'
ubar_var.units = 'm/s'

vbar_var = out.createVariable('vbar_north', 'f4', ('v2d_time','xi_rho'), zlib=True, complevel=4)
vbar_var.long_name = 'barotropic v-velocity northern boundary'
vbar_var.units = 'm/s'

zeta_var = out.createVariable('zeta_north', 'f4', ('zeta_time','xi_rho'), zlib=True, complevel=4)
zeta_var.long_name = 'sea surface height northern boundary'
zeta_var.units = 'm'

# Process timesteps
log(f"\n4. Processing {nt} timesteps...")

for t in range(nt):
    if t % 100 == 0:
        log(f"  {t}/{nt} ({100*t/nt:.0f}%)")
    
    # Read ONE timestep
    thetao = ds_bd2.variables['thetao'][t,:,bry_idx,:]
    so = ds_bd2.variables['so'][t,:,bry_idx,:]
    zos = ds_bd2.variables['zos'][t,bry_idx,:]
    uo = ds_bd1.variables['uo'][t,:,bry_idx,:]
    vo = ds_bd1.variables['vo'][t,:,bry_idx,:]
    
    # Clean
    if hasattr(thetao, 'mask'):
        thetao = np.ma.filled(thetao, np.nan)
        so = np.ma.filled(so, np.nan)
        zos = np.ma.filled(zos, np.nan)
        uo = np.ma.filled(uo, np.nan)
        vo = np.ma.filled(vo, np.nan)
    
    # Horizontal interpolation
    nz = len(depth_cmems)
    temp_h = np.zeros((nz, xi_rho), dtype=np.float32)
    salt_h = np.zeros((nz, xi_rho), dtype=np.float32)
    u_h = np.zeros((nz, xi_rho), dtype=np.float32)
    v_h = np.zeros((nz, xi_rho), dtype=np.float32)
    
    for k in range(nz):
        temp_h[k] = interp1d(lon_cmems, thetao[k], bounds_error=False, fill_value=np.nan)(lon_bry_adj)
        salt_h[k] = interp1d(lon_cmems, so[k], bounds_error=False, fill_value=np.nan)(lon_bry_adj)
        u_h[k] = interp1d(lon_cmems, uo[k], bounds_error=False, fill_value=np.nan)(lon_bry_adj)
        v_h[k] = interp1d(lon_cmems, vo[k], bounds_error=False, fill_value=np.nan)(lon_bry_adj)
    
    zos_h = interp1d(lon_cmems, zos, bounds_error=False, fill_value=0)(lon_bry_adj)
    
    # ============ VALIDATION: Replace unphysical zeros ============
    # Zeros indicate unfilled land/missing data - replace with NaN then fill
    temp_h = np.where(np.abs(temp_h) < 0.01, np.nan, temp_h)
    salt_h = np.where(np.abs(salt_h) < 0.1, np.nan, salt_h)
    
    # Get domain means for filling (typical Southern Ocean values as fallback)
    temp_mean = np.nanmean(temp_h)
    salt_mean = np.nanmean(salt_h)
    if np.isnan(temp_mean): temp_mean = 5.0
    if np.isnan(salt_mean): salt_mean = 34.5
    
    # Fill NaN with domain means
    temp_h = np.where(np.isnan(temp_h), temp_mean, temp_h)
    salt_h = np.where(np.isnan(salt_h), salt_mean, salt_h)
    u_h = np.where(np.isnan(u_h), 0.0, u_h)
    v_h = np.where(np.isnan(v_h), 0.0, v_h)
    # ==============================================================
    
    # Vertical interpolation
    temp_out = np.zeros((N, xi_rho), dtype=np.float32)
    salt_out = np.zeros((N, xi_rho), dtype=np.float32)
    u_out = np.zeros((N, xi_rho), dtype=np.float32)
    v_out = np.zeros((N, xi_rho), dtype=np.float32)
    
    for i in range(xi_rho):
        target = depths_roms[:,0,i]
        
        valid = ~np.isnan(temp_h[:,i])
        if np.sum(valid) >= 2:
            temp_out[:,i] = interp1d(depth_cmems[valid], temp_h[valid,i], bounds_error=False, fill_value='extrapolate')(target)
        
        valid = ~np.isnan(salt_h[:,i])
        if np.sum(valid) >= 2:
            salt_out[:,i] = interp1d(depth_cmems[valid], salt_h[valid,i], bounds_error=False, fill_value='extrapolate')(target)
        
        valid = ~np.isnan(u_h[:,i])
        if np.sum(valid) >= 2:
            u_out[:,i] = interp1d(depth_cmems[valid], u_h[valid,i], bounds_error=False, fill_value=0)(target)
        
        valid = ~np.isnan(v_h[:,i])
        if np.sum(valid) >= 2:
            v_out[:,i] = interp1d(depth_cmems[valid], v_h[valid,i], bounds_error=False, fill_value=0)(target)
    
    # Barotropic
    dz = np.abs(np.diff(depths_roms[:,0,:], axis=0))
    dz = np.vstack([dz, dz[-1:,:]])
    
    ubar = np.nansum(u_out * dz, axis=0) / np.sum(dz, axis=0)
    vbar = np.nansum(v_out * dz, axis=0) / np.sum(dz, axis=0)
    
    # Write to ALL time variables
    ocean_time[t] = time_roms[t]
    zeta_time[t] = time_roms[t]
    v2d_time[t] = time_roms[t]
    v3d_time[t] = time_roms[t]
    temp_time[t] = time_roms[t]
    salt_time[t] = time_roms[t]
    
    temp_var[t] = temp_out
    salt_var[t] = salt_out
    u_var[t] = u_out
    v_var[t] = v_out
    ubar_var[t] = ubar
    vbar_var[t] = vbar
    zeta_var[t] = zos_h

ds_bd1.close()
ds_bd2.close()
out.close()

log(f"\n✓ DONE: {OUTPUT_BRY}")
log(f"  Size: {os.path.getsize(OUTPUT_BRY)/(1024**3):.2f} GB")
log("✓ All required ROMS time variables included!")