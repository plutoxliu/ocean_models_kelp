#!/usr/bin/env python3
"""
Comprehensive Post-Analysis of Kelp Drift Results
- Regional connectivity analysis
- Seasonal patterns
- Antarctic connectivity assessment
- Static maps and animations in Antarctic Polar Stereographic projection

OPTIMIZED FOR LARGE DATASETS (2M+ particles)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from scipy.spatial import distance
import warnings
import gc
warnings.filterwarnings('ignore')

# Disable interactive mode
plt.ioff()

# =============================================================================
# PERFORMANCE SETTINGS FOR LARGE DATASETS
# =============================================================================

# Maximum particles to load at once for detailed analysis
CHUNK_SIZE = 1000

# Maximum particles to sample for statistics (per site)
MAX_SAMPLE_PER_SITE = 50000

# Samples for trajectory plots
TRAJECTORY_SAMPLE_PER_SITE = 10000

# Resolution for density plots (degrees)
DENSITY_RESOLUTION = 0.5

# =============================================================================
# CONFIGURATION
# =============================================================================

ZARR_FILE = 'kelp_3year_parcels_fine_All.zarr'
OUTPUT_DIR = './analysis_results_all'
ROMS_GRID_FILE = './ocean_grd_fine.nc'  # ROMS grid file for coastlines/bathymetry
INITIAL_FILE = './roms_ini_fine.nc'  # Initial condition file for current background

# =============================================================================
# LANDMASS DEFINITIONS - for connectivity analysis
# Bounding boxes [lon_min, lon_max, lat_min, lat_max] in -180/180 format
# =============================================================================

LANDMASSES = {
    # Major Continents
    'Antarctica': {'lon_range': (-180, 180), 'lat_range': (-90, -60), 'type': 'continent'},
    'South_America': {'lon_range': (-80, -55), 'lat_range': (-56, -30), 'type': 'continent'},
    'Australia': {'lon_range': (110, 155), 'lat_range': (-45, -10), 'type': 'continent'},
    'New_Zealand': {'lon_range': (166, 179), 'lat_range': (-48, -34), 'type': 'landmass'},
    'Tasmania': {'lon_range': (144, 149), 'lat_range': (-44, -40), 'type': 'island'},
    'South_Africa': {'lon_range': (16, 33), 'lat_range': (-35, -22), 'type': 'continent'},
    
    # Sub-Antarctic Islands - Atlantic Sector
    'Falkland_Islands': {'lon_range': (-62, -57), 'lat_range': (-53, -51), 'type': 'island'},
    'South_Georgia': {'lon_range': (-39, -35), 'lat_range': (-55, -53.5), 'type': 'island'},
    'South_Sandwich': {'lon_range': (-29, -25), 'lat_range': (-60, -56), 'type': 'island'},
    'South_Orkney': {'lon_range': (-47, -44), 'lat_range': (-61.5, -60), 'type': 'island'},
    'Bouvet_Island': {'lon_range': (2.5, 4.5), 'lat_range': (-55, -54), 'type': 'island'},
    'Gough_Island': {'lon_range': (-11, -9), 'lat_range': (-41, -40), 'type': 'island'},
    'Tristan_da_Cunha': {'lon_range': (-13, -11), 'lat_range': (-38, -37), 'type': 'island'},
    
    # Sub-Antarctic Islands - Indian Sector
    'Prince_Edward_Islands': {'lon_range': (37, 39), 'lat_range': (-47.5, -46), 'type': 'island'},
    'Crozet_Islands': {'lon_range': (50, 53), 'lat_range': (-47, -46), 'type': 'island'},
    'Kerguelen': {'lon_range': (68, 71), 'lat_range': (-50, -48.5), 'type': 'island'},
    'Heard_McDonald': {'lon_range': (72, 75), 'lat_range': (-54, -52), 'type': 'island'},
    'Amsterdam_St_Paul': {'lon_range': (77, 78), 'lat_range': (-39, -37), 'type': 'island'},
    
    # Sub-Antarctic Islands - Pacific/NZ Sector
    'Stewart_Island': {'lon_range': (167, 169), 'lat_range': (-47.5, -46.5), 'type': 'island'},
    'Snares_Islands': {'lon_range': (166, 167.5), 'lat_range': (-48.5, -47.5), 'type': 'island'},
    'Auckland_Islands': {'lon_range': (165.5, 167), 'lat_range': (-51, -50), 'type': 'island'},
    'Campbell_Island': {'lon_range': (168.5, 170), 'lat_range': (-53, -52), 'type': 'island'},
    'Antipodes_Islands': {'lon_range': (178, 179.5), 'lat_range': (-50, -49), 'type': 'island'},
    'Bounty_Islands': {'lon_range': (178.5, 180), 'lat_range': (-48, -47.5), 'type': 'island'},
    'Chatham_Islands': {'lon_range': (-177.5, -175.5), 'lat_range': (-45, -43), 'type': 'island'},
    'Macquarie_Island': {'lon_range': (158, 160), 'lat_range': (-55, -54), 'type': 'island'},
    
    # South American Islands
    'Tierra_del_Fuego': {'lon_range': (-72, -64), 'lat_range': (-56, -52), 'type': 'island'},
    'Diego_Ramirez': {'lon_range': (-69, -68), 'lat_range': (-57, -56), 'type': 'island'},
    'Cape_Horn_region': {'lon_range': (-68, -66), 'lat_range': (-56.5, -55), 'type': 'island'}
}

# Release sites - This will be auto-populated from the zarr file
# The dictionary maps site_id (origin_marker) to site info
# These are the POSSIBLE sites - actual sites used depend on scenario
ALL_RELEASE_SITES = {
    'South_Georgia': {'lon': -36.5, 'lat': -54.5, 'region': 'Atlantic'},
    'Falkland_Islands': {'lon': -59.5, 'lat': -51.8, 'region': 'Atlantic'},
    'Gough_Island': {'lon': -9.9, 'lat': -40.3, 'region': 'Atlantic'},
    'Marion_Island': {'lon': 37.8, 'lat': -46.9, 'region': 'Indian'},
    'Prince_Edward': {'lon': 37.7, 'lat': -46.6, 'region': 'Indian'},
    'Kerguelen': {'lon': 70.2, 'lat': -49.4, 'region': 'Indian'},
    'Stewart_Island': {'lon': 167.5, 'lat': -47.2, 'region': 'Pacific'},
    'Auckland_Islands': {'lon': 166.3, 'lat': -50.7, 'region': 'Pacific'},
    'Chatham_Islands': {'lon': -176.5, 'lat': -44.0, 'region': 'Pacific'},
    'Cape_Horn': {'lon': -67.3, 'lat': -55.9, 'region': 'Pacific'},
    'Chile_central': {'lon': -73.2, 'lat': -36.8, 'region': 'Pacific'},
    'Chile_south': {'lon': -74.5, 'lat': -43.2, 'region': 'Pacific'},
    'Tasmania': {'lon': 148.0, 'lat': -43.4, 'region': 'Pacific'},
    'Macquarie_Island': {'lon': 158.9, 'lat': -54.5, 'region': 'Pacific'},
}

# This will be populated by detect_sites()
RELEASE_SITES = {}

# Antarctic boundary
ANTARCTICA_LAT = -60.0

# Map extent - crop at 30°S
NORTHERN_LIMIT = -30

# Custom color palette for release sites (R colors converted to hex)
SITE_COLORS = [
    "#e31a1c",  # red
    "#33a02c",  # green
    "#0072B2",  # blue
    "#fb9a99",  # light red/pink
    "#F0E442",  # yellow
    "#6a3d9a",  # purple
    "#b2df8a",  # light green
    "#56B4E9",  # light blue
    "#ff7f00",  # orange
    "#a6cee3",  # very light blue
]

# Antarctic sectors for detailed arrival analysis
ANTARCTIC_SECTORS = {
    'Weddell Sea': {'lon_range': (-60, 20), 'lat_range': (-90, -60)},
    'East Antarctica (20-90E)': {'lon_range': (20, 90), 'lat_range': (-90, -60)},
    'East Antarctica (90-160E)': {'lon_range': (90, 160), 'lat_range': (-90, -60)},
    'Ross Sea': {'lon_range': (160, -140), 'lat_range': (-90, -60)},
    'Amundsen-Bellingshausen': {'lon_range': (-140, -60), 'lat_range': (-90, -60)},
}

# Regions for connectivity analysis
REGIONS = {
    'Weddell Sea': {'lon_range': (-60, 20), 'lat_range': (-78, -60)},
    'Indian Sector': {'lon_range': (20, 90), 'lat_range': (-70, -60)},
    'Pacific Sector': {'lon_range': (90, 160), 'lat_range': (-75, -60)},
    'Ross Sea': {'lon_range': (160, -140), 'lat_range': (-78, -60)},
    'Bellingshausen': {'lon_range': (-140, -60), 'lat_range': (-75, -60)},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_longitude(lon):
    """Normalize longitude to [-180, 180]"""
    return ((lon + 180) % 360) - 180

def is_in_region(lon, lat, region):
    """Check if point is in region"""
    lon = normalize_longitude(lon)
    lon_range = region['lon_range']
    lat_range = region['lat_range']
    
    # Handle wrap-around at dateline
    if lon_range[0] > lon_range[1]:  # Crosses dateline
        lon_ok = (lon >= lon_range[0]) or (lon <= lon_range[1])
    else:
        lon_ok = (lon_range[0] <= lon <= lon_range[1])
    
    lat_ok = (lat_range[0] <= lat <= lat_range[1])
    
    return lon_ok and lat_ok

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great circle distance in km"""
    R = 6371  # Earth radius in km
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def is_near_landmass(lon, lat, landmass, buffer_deg=0.5):
    """Check if point is near a landmass (within buffer degrees)"""
    lon = normalize_longitude(lon)
    lon_range = landmass['lon_range']
    lat_range = landmass['lat_range']
    
    # Expand range by buffer
    lon_min = lon_range[0] - buffer_deg
    lon_max = lon_range[1] + buffer_deg
    lat_min = lat_range[0] - buffer_deg
    lat_max = lat_range[1] + buffer_deg
    
    # Handle dateline crossing
    if lon_min < -180:
        lon_ok = (lon >= lon_min + 360) or (lon <= lon_max)
    elif lon_max > 180:
        lon_ok = (lon >= lon_min) or (lon <= lon_max - 360)
    else:
        lon_ok = (lon_min <= lon <= lon_max)
    
    lat_ok = (lat_min <= lat <= lat_max)
    return lon_ok and lat_ok


def identify_landmass(lon, lat, buffer_deg=0.5):
    """Identify which landmass a point is near, returns name or None"""
    lon = normalize_longitude(lon)
    for name, info in LANDMASSES.items():
        if is_near_landmass(lon, lat, info, buffer_deg):
            return name
    return None


def load_currents():
    """Load ocean currents from initial condition file for background plotting"""
    import os
    import netCDF4 as nc
    
    if not os.path.exists(INITIAL_FILE):
        print(f"  Warning: Initial condition file not found: {INITIAL_FILE}")
        return None
    
    print(f"  Loading currents from: {INITIAL_FILE}")
    with nc.Dataset(INITIAL_FILE, 'r') as f:
        # ubar is on u-grid (eta_rho, xi_u), vbar is on v-grid (eta_v, xi_rho)
        # Read first timestep
        ubar_u = f.variables['ubar'][0, :, :]  # (eta_rho, xi_u)
        vbar_v = f.variables['vbar'][0, :, :]  # (eta_v, xi_rho)
    
    # Handle masked arrays
    if np.ma.isMaskedArray(ubar_u):
        ubar_u = np.ma.filled(ubar_u.astype(np.float64), np.nan)
    else:
        ubar_u = ubar_u.astype(np.float64)
    if np.ma.isMaskedArray(vbar_v):
        vbar_v = np.ma.filled(vbar_v.astype(np.float64), np.nan)
    else:
        vbar_v = vbar_v.astype(np.float64)
    
    # Interpolate to rho grid for plotting
    # ubar: (eta_rho, xi_u) -> (eta_rho, xi_rho) by padding
    ubar = np.zeros((ubar_u.shape[0], ubar_u.shape[1] + 1))
    ubar[:, 1:-1] = 0.5 * (ubar_u[:, :-1] + ubar_u[:, 1:])
    ubar[:, 0] = ubar_u[:, 0]
    ubar[:, -1] = ubar_u[:, -1]
    
    # vbar: (eta_v, xi_rho) -> (eta_rho, xi_rho) by padding  
    vbar = np.zeros((vbar_v.shape[0] + 1, vbar_v.shape[1]))
    vbar[1:-1, :] = 0.5 * (vbar_v[:-1, :] + vbar_v[1:, :])
    vbar[0, :] = vbar_v[0, :]
    vbar[-1, :] = vbar_v[-1, :]
    
    # Calculate speed
    speed = np.sqrt(ubar**2 + vbar**2)
    
    print(f"    Current speed range: {np.nanmin(speed):.3f} to {np.nanmax(speed):.3f} m/s")
    
    return {'ubar': ubar, 'vbar': vbar, 'speed': speed}


def load_roms_grid():
    """Load ROMS grid file for coastlines and bathymetry"""
    import os
    
    if not os.path.exists(ROMS_GRID_FILE):
        print(f"  Warning: ROMS grid file not found: {ROMS_GRID_FILE}")
        return None
    
    print(f"  Loading ROMS grid: {ROMS_GRID_FILE}")
    grid = xr.open_dataset(ROMS_GRID_FILE)
    
    # Extract key variables
    lon_rho = grid['lon_rho'].values
    lat_rho = grid['lat_rho'].values
    mask_rho = grid['mask_rho'].values  # 1=water, 0=land
    h = grid['h'].values  # bathymetry
    
    grid.close()
    
    # Convert longitude from 0-360 to -180 to 180 if needed
    if lon_rho.max() > 180:
        lon_rho = np.where(lon_rho > 180, lon_rho - 360, lon_rho)
    
    return {
        'lon': lon_rho,
        'lat': lat_rho,
        'mask': mask_rho,
        'h': h
    }


def add_roms_features(ax, roms_grid, add_bathy=True, add_land=True, add_coastline=True, 
                      currents=None, add_currents=False):
    """Add ROMS grid features to a cartopy axis"""
    
    if roms_grid is None:
        # Fallback to cartopy features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
        ax.coastlines(resolution='50m', linewidth=0.5, zorder=2)
        return None
    
    lon = roms_grid['lon']
    lat = roms_grid['lat']
    mask = roms_grid['mask']
    h = roms_grid['h']
    
    pcm = None
    
    # Add ocean currents as background (if provided)
    if add_currents and currents is not None:
        speed_masked = np.ma.masked_where(mask == 0, currents['speed'])
        pcm = ax.pcolormesh(lon, lat, speed_masked, 
                           transform=ccrs.PlateCarree(),
                           cmap='Blues', 
                           vmin=0, vmax=0.5,
                           shading='auto', zorder=0)
    
    # Add bathymetry (ocean depth) - only if not showing currents
    elif add_bathy:
        h_masked = np.ma.masked_where(mask == 0, h)
        pcm = ax.pcolormesh(lon, lat, h_masked, 
                           transform=ccrs.PlateCarree(),
                           cmap='Blues', 
                           vmin=0, vmax=6000,
                           shading='auto', zorder=0)
    
    # Add land mask
    if add_land:
        land_mask = np.ma.masked_where(mask == 1, mask)
        ax.pcolormesh(lon, lat, land_mask, 
                     transform=ccrs.PlateCarree(),
                     cmap='Greys', vmin=0, vmax=1,
                     shading='auto', zorder=1)
    
    # Add coastline contour (mask boundary)
    if add_coastline:
        ax.contour(lon, lat, mask, levels=[0.5], 
                  transform=ccrs.PlateCarree(),
                  colors='black', linewidths=0.5, zorder=2)
    
    return pcm

# =============================================================================
# LOAD DATA
# =============================================================================

def load_results():
    """Load simulation results and auto-detect release sites.
    
    Handles the case where Parcels stores cumulative deltas instead of absolute positions.
    If lon/lat values are outside expected ranges, reconstructs positions from origin + deltas.
    """
    global RELEASE_SITES
    
    print("="*70)
    print("LOADING SIMULATION RESULTS")
    print("="*70)
    
    ds = xr.open_zarr(ZARR_FILE)
    
    print(f"\nDataset variables: {list(ds.data_vars)}")
    print(f"Particles (trajectories): {len(ds.trajectory):,}")
    print(f"Timesteps (obs): {len(ds.obs):,}")
    
    # Check raw lon/lat ranges
    raw_lon_min = np.nanmin(ds.lon.values)
    raw_lon_max = np.nanmax(ds.lon.values)
    raw_lat_min = np.nanmin(ds.lat.values)
    raw_lat_max = np.nanmax(ds.lat.values)
    
    print(f"\nRaw data ranges:")
    print(f"  Longitude: {raw_lon_min:.1f} to {raw_lon_max:.1f}")
    print(f"  Latitude: {raw_lat_min:.1f} to {raw_lat_max:.1f}")
    
    # Get origin positions
    origin_lons = ds.origin_lon.values
    origin_lats = ds.origin_lat.values
    if origin_lons.ndim > 1:
        origin_lons = origin_lons[:, 0]
    if origin_lats.ndim > 1:
        origin_lats = origin_lats[:, 0]
    
    # Check first observation values
    first_lons = ds.lon.values[:, 0]
    first_lats = ds.lat.values[:, 0]
    
    # Check if first observation is close to origin
    lon_diff = np.nanmean(np.abs(first_lons - origin_lons))
    lat_diff = np.nanmean(np.abs(first_lats - origin_lats))
    
    print(f"\n  Diagnostic:")
    print(f"    Mean origin: lon={np.nanmean(origin_lons):.2f}, lat={np.nanmean(origin_lats):.2f}")
    print(f"    Mean first obs: lon={np.nanmean(first_lons):.2f}, lat={np.nanmean(first_lats):.2f}")
    print(f"    Difference: lon={lon_diff:.2f}°, lat={lat_diff:.2f}°")
    
    # Parcels with particle_dlon/particle_dlat stores data in a specific way:
    # The 'lon' and 'lat' in output are the CUMULATIVE position updates
    # At t=0, lon ≈ origin_lon (small delta from first timestep)
    # At t=N, lon = origin_lon + sum of all dlons applied
    # 
    # BUT the values we see suggest it's storing absolute positions that 
    # include the origin. Check if subtracting origin from first obs gives ~0
    
    first_minus_origin_lon = np.nanmean(first_lons - origin_lons)
    first_minus_origin_lat = np.nanmean(first_lats - origin_lats)
    
    print(f"    First - Origin: lon={first_minus_origin_lon:.4f}, lat={first_minus_origin_lat:.4f}")
    
    # If first obs is very close to origin, the data format is:
    # stored_value = absolute_position (which already includes origin)
    # No transformation needed, just clean up out-of-bounds values
    
    if abs(lon_diff) < 1.0 and abs(lat_diff) < 1.0:
        print("\n✓ First observation matches origin - data is absolute positions")
        print("  Cleaning out-of-bounds values...")
        
        lon_data = ds.lon.values.copy()
        lat_data = ds.lat.values.copy()
        
        # Count bad values
        n_bad_lat = np.sum((lat_data < -90) | (lat_data > 90))
        n_total = lat_data.size
        
        print(f"  Latitude outside [-90, 90]: {n_bad_lat:,} ({100*n_bad_lat/n_total:.2f}%)")
        
        # Mark extreme values as NaN
        lat_data[(lat_data < -90) | (lat_data > 90)] = np.nan
        
        # Normalize longitude to 0-360
        lon_data = np.mod(lon_data, 360)
        
        ds['lon'] = (('traj', 'obs'), lon_data)
        ds['lat'] = (('traj', 'obs'), lat_data)
        
    else:
        # The stored values might be cumulative deltas from zero
        # Need to check if the DIFFERENCE between consecutive observations makes sense
        print("\n⚠ First observation differs from origin - checking data format...")
        
        # Look at the difference between consecutive timesteps
        lon_data = ds.lon.values
        lat_data = ds.lat.values
        
        # Calculate step-to-step changes for first particle
        lon_steps = np.diff(lon_data[0, :10])
        lat_steps = np.diff(lat_data[0, :10])
        
        print(f"    First particle lon steps[0:5]: {lon_steps[:5]}")
        print(f"    First particle lat steps[0:5]: {lat_steps[:5]}")
        
        # If steps are small (< 1 degree/day), then the stored values ARE absolute positions
        # that just happen to drift far from origin over time
        mean_lon_step = np.nanmean(np.abs(lon_steps))
        mean_lat_step = np.nanmean(np.abs(lat_steps))
        
        print(f"    Mean step size: lon={mean_lon_step:.4f}°, lat={mean_lat_step:.4f}°")
        
        if mean_lon_step < 2.0 and mean_lat_step < 2.0:
            print("\n✓ Step sizes are reasonable - data is absolute positions")
            print("  The extreme values are real particle positions (simulation issue)")
            print("  Cleaning out-of-bounds values...")
            
            lon_data = ds.lon.values.copy()
            lat_data = ds.lat.values.copy()
            
            # Mark extreme latitudes as NaN
            n_bad_lat = np.sum((lat_data < -90) | (lat_data > 90))
            n_total = lat_data.size
            print(f"  Latitude outside [-90, 90]: {n_bad_lat:,} ({100*n_bad_lat/n_total:.2f}%)")
            
            lat_data[(lat_data < -90) | (lat_data > 90)] = np.nan
            lon_data = np.mod(lon_data, 360)
            
            ds['lon'] = (('traj', 'obs'), lon_data)
            ds['lat'] = (('traj', 'obs'), lat_data)
        else:
            print("\n⚠ Large step sizes detected - may need different reconstruction")
            # Keep original data for now
    
    print(f"\n  Final ranges:")
    print(f"    Longitude: {np.nanmin(ds.lon.values):.1f} to {np.nanmax(ds.lon.values):.1f}")
    print(f"    Latitude: {np.nanmin(ds.lat.values):.1f} to {np.nanmax(ds.lat.values):.1f}")
    
    # Auto-detect release sites from origin_lon/origin_lat
    print("\nAuto-detecting release sites...")
    
    origin_lons = ds.origin_lon.values
    origin_lats = ds.origin_lat.values
    origin_markers = ds.origin_marker.values
    
    # Handle 2D arrays (trajectory x obs) - take first obs
    if origin_lons.ndim > 1:
        origin_lons = origin_lons[:, 0]
    if origin_lats.ndim > 1:
        origin_lats = origin_lats[:, 0]
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    # Find unique site IDs
    unique_markers = np.unique(origin_markers[~np.isnan(origin_markers)]).astype(int)
    print(f"  Found {len(unique_markers)} unique site markers: {unique_markers}")
    
    # Build RELEASE_SITES from data
    RELEASE_SITES = {}
    for site_id in unique_markers:
        mask = origin_markers == site_id
        site_lon = np.median(origin_lons[mask])
        site_lat = np.median(origin_lats[mask])
        
        # Try to match to known site
        site_name = f"Site_{site_id}"
        site_region = "Unknown"
        
        for name, info in ALL_RELEASE_SITES.items():
            # Check if coordinates match (within 1 degree)
            # Handle both 0-360 and -180/180 conventions
            info_lon = info['lon']
            if info_lon < 0:
                info_lon += 360
            check_lon = site_lon
            if check_lon < 0:
                check_lon += 360
            
            if abs(info_lon - check_lon) < 1.0 and abs(info['lat'] - site_lat) < 1.0:
                site_name = name
                site_region = info['region']
                break
        
        RELEASE_SITES[site_id] = {
            'name': site_name,
            'lon': float(site_lon),
            'lat': float(site_lat),
            'region': site_region,
            'n_particles': int(np.sum(mask))
        }
        print(f"    Site {site_id}: {site_name} ({site_lon:.1f}°E, {site_lat:.1f}°S) - {np.sum(mask):,} particles")
    
    # Calculate duration from age variable (more reliable than time)
    print("\nCalculating duration...")
    
    if 'age' in ds.data_vars:
        # Age is in seconds
        age_vals = ds.age.values
        max_age_seconds = np.nanmax(age_vals)
        duration_days = max_age_seconds / 86400.0
        print(f"  Duration from age: {duration_days:.1f} days ({duration_days/365:.2f} years)")
    else:
        # Fallback to time variable
        try:
            # Try to get time range
            time_vals = ds.time.values
            if time_vals.ndim > 1:
                # Flatten and get valid times
                time_flat = time_vals.flatten()
                if np.issubdtype(time_flat.dtype, np.datetime64):
                    valid_times = time_flat[~np.isnat(time_flat)]
                else:
                    valid_times = time_flat[~np.isnan(time_flat)]
                
                if len(valid_times) > 0:
                    time_start = np.min(valid_times)
                    time_end = np.max(valid_times)
                    
                    if np.issubdtype(type(time_start), np.datetime64):
                        duration_days = (time_end - time_start) / np.timedelta64(1, 'D')
                    else:
                        duration_days = float(time_end - time_start)
                        if duration_days > 100000:  # Likely in seconds
                            duration_days /= 86400.0
                else:
                    duration_days = len(ds.obs)  # Assume daily output
            else:
                duration_days = len(ds.obs)
                
            print(f"  Duration from time: {duration_days:.1f} days")
        except Exception as e:
            print(f"  Warning: Could not calculate duration: {e}")
            duration_days = len(ds.obs)  # Assume daily output
            print(f"  Using number of obs as duration: {duration_days} days")
    
    # Print final data ranges
    print("\nFinal data ranges:")
    print(f"  Longitude: {np.nanmin(ds.lon.values):.1f} to {np.nanmax(ds.lon.values):.1f}")
    print(f"  Latitude: {np.nanmin(ds.lat.values):.1f} to {np.nanmax(ds.lat.values):.1f}")
    
    # Calculate displacement and total distance traveled
    print("\nParticle movement statistics:")
    lon = ds.lon.values
    lat = ds.lat.values
    
    # Displacement (start to end)
    start_lon = lon[:, 0]
    start_lat = lat[:, 0]
    end_lon = lon[:, -1]
    end_lat = lat[:, -1]
    
    # Calculate displacement using haversine
    displacements = []
    for i in range(len(start_lon)):
        if not np.isnan(start_lon[i]) and not np.isnan(end_lon[i]):
            d = haversine_distance(start_lon[i], start_lat[i], end_lon[i], end_lat[i])
            displacements.append(d)
    
    mean_displacement = np.mean(displacements) if displacements else 0
    
    # Total distance traveled (sum of all steps)
    total_distances = []
    for i in range(lon.shape[0]):
        particle_lon = lon[i, :]
        particle_lat = lat[i, :]
        
        # Remove NaN
        valid = ~np.isnan(particle_lon) & ~np.isnan(particle_lat)
        p_lon = particle_lon[valid]
        p_lat = particle_lat[valid]
        
        if len(p_lon) > 1:
            total_dist = 0
            for j in range(1, len(p_lon)):
                total_dist += haversine_distance(p_lon[j-1], p_lat[j-1], p_lon[j], p_lat[j])
            total_distances.append(total_dist)
    
    mean_total_distance = np.mean(total_distances) if total_distances else 0
    max_total_distance = np.max(total_distances) if total_distances else 0
    
    n_days = lon.shape[1]
    
    print(f"  Mean displacement (start→end): {mean_displacement:.0f} km")
    print(f"  Mean total distance traveled:  {mean_total_distance:.0f} km")
    print(f"  Max total distance traveled:   {max_total_distance:.0f} km")
    print(f"  Tortuosity (distance/displacement): {mean_total_distance/mean_displacement:.2f}" if mean_displacement > 0 else "")
    print(f"  Average speed (from displacement): {mean_displacement / n_days / 24 * 1000 / 3600:.3f} m/s")
    print(f"  Average speed (from total dist):   {mean_total_distance / n_days / 24 * 1000 / 3600:.3f} m/s")
    print(f"  Average daily movement: {mean_total_distance / n_days:.1f} km/day")
    
    return ds


# =============================================================================
# COMPREHENSIVE SUMMARY STATISTICS BY RELEASE SITE (OPTIMIZED FOR LARGE DATA)
# =============================================================================

def compute_site_summary_statistics(ds):
    """
    Compute comprehensive summary statistics for each release site.
    OPTIMIZED for large datasets (2M+ particles) using sampling and vectorization.
    
    Returns a DataFrame with:
    - Mean/max displacement, mean/max distance, tortuosity
    - Average daily speed
    - Antarctica arrival counts, percentages, and mean arrival time
    - Sea ice encounter and trapping statistics
    """
    
    print("\n" + "="*70)
    print("COMPUTING SUMMARY STATISTICS BY RELEASE SITE")
    print("="*70)
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    n_total = len(origin_markers)
    n_timesteps = ds.dims['obs']
    n_days = n_timesteps
    
    print(f"  Total particles: {n_total:,}")
    print(f"  Timesteps: {n_timesteps}")
    
    # Check for ice data
    has_ice = 'ice_concentration' in ds.variables
    
    summary_results = []
    
    for site_id, site_info in RELEASE_SITES.items():
        print(f"  Processing {site_info['name']}...", flush=True)
        
        site_indices = np.where(origin_markers == site_id)[0]
        n_particles_total = len(site_indices)
        
        if n_particles_total == 0:
            continue
        
        # Sample if too many particles
        if n_particles_total > MAX_SAMPLE_PER_SITE:
            print(f"    Sampling {MAX_SAMPLE_PER_SITE:,} of {n_particles_total:,} particles", flush=True)
            sample_indices = np.random.choice(site_indices, MAX_SAMPLE_PER_SITE, replace=False)
        else:
            sample_indices = site_indices
        
        n_sample = len(sample_indices)
        
        # Load data for sampled particles only
        site_lons = ds.lon.values[sample_indices, :]
        site_lats = ds.lat.values[sample_indices, :]
        
        # === DISPLACEMENT (vectorized) ===
        start_lon = site_lons[:, 0]
        start_lat = site_lats[:, 0]
        
        # Find last valid position for each particle (vectorized where possible)
        # Create mask of valid positions
        valid_mask = ~np.isnan(site_lons) & ~np.isnan(site_lats)
        
        # Get last valid index for each particle
        last_valid_idx = np.zeros(n_sample, dtype=int)
        for i in range(n_sample):
            valid_indices = np.where(valid_mask[i, :])[0]
            if len(valid_indices) > 0:
                last_valid_idx[i] = valid_indices[-1]
        
        end_lon = site_lons[np.arange(n_sample), last_valid_idx]
        end_lat = site_lats[np.arange(n_sample), last_valid_idx]
        
        # Vectorized haversine
        valid_disp = ~np.isnan(start_lon) & ~np.isnan(end_lon)
        if np.any(valid_disp):
            displacements = haversine_distance(
                start_lon[valid_disp], start_lat[valid_disp],
                end_lon[valid_disp], end_lat[valid_disp]
            )
            mean_displacement = np.mean(displacements)
            max_displacement = np.max(displacements)
        else:
            mean_displacement = 0
            max_displacement = 0
        
        # === TOTAL DISTANCE (sample for speed) ===
        n_dist_sample = min(5000, n_sample)
        dist_sample_idx = np.random.choice(n_sample, n_dist_sample, replace=False)
        
        total_distances = []
        for i in dist_sample_idx:
            p_lon = site_lons[i, :]
            p_lat = site_lats[i, :]
            valid = ~np.isnan(p_lon) & ~np.isnan(p_lat)
            p_lon = p_lon[valid]
            p_lat = p_lat[valid]
            
            if len(p_lon) > 1:
                # Vectorized step distances
                step_dists = haversine_distance(p_lon[:-1], p_lat[:-1], p_lon[1:], p_lat[1:])
                total_distances.append(np.sum(step_dists))
        
        mean_distance = np.mean(total_distances) if total_distances else 0
        max_distance = np.max(total_distances) if total_distances else 0
        
        # Tortuosity and speed
        tortuosity = mean_distance / mean_displacement if mean_displacement > 0 else np.nan
        avg_daily_speed = mean_distance / n_days if n_days > 0 else 0
        
        # === ANTARCTICA ARRIVAL (vectorized) ===
        min_lats = np.nanmin(site_lats, axis=1)
        reached_antarctica = min_lats < ANTARCTICA_LAT
        n_reached_sample = np.sum(reached_antarctica)
        
        # Scale up to total particles
        pct_reached_ant = 100 * n_reached_sample / n_sample
        n_reached_ant = int(pct_reached_ant * n_particles_total / 100)
        
        # Mean arrival time (sample)
        arrival_times = []
        reached_indices = np.where(reached_antarctica)[0][:2000]  # Limit for speed
        for i in reached_indices:
            crossing = np.where(site_lats[i, :] < ANTARCTICA_LAT)[0]
            if len(crossing) > 0:
                arrival_times.append(crossing[0])
        mean_arrival_time = np.mean(arrival_times) if arrival_times else np.nan
        
        # === SEA ICE STATISTICS ===
        if has_ice:
            site_ice = ds.ice_concentration.values[sample_indices, :]
            # Flatten and filter valid
            ice_flat = site_ice.flatten()
            valid_ice = ice_flat[~np.isnan(ice_flat)]
            
            if len(valid_ice) > 0:
                pct_encounter = 100 * np.sum(valid_ice > 0) / len(valid_ice)
                pct_trapped = 100 * np.sum(valid_ice > 0.80) / len(valid_ice)
            else:
                pct_encounter = 0
                pct_trapped = 0
            del site_ice, ice_flat
        else:
            pct_encounter = np.nan
            pct_trapped = np.nan
        
        summary_results.append({
            'Site': site_info['name'],
            'N_Particles': n_particles_total,
            'Mean_Displacement_km': round(mean_displacement, 1),
            'Max_Displacement_km': round(max_displacement, 1),
            'Mean_Distance_km': round(mean_distance, 1),
            'Max_Distance_km': round(max_distance, 1),
            'Tortuosity': round(tortuosity, 2) if not np.isnan(tortuosity) else np.nan,
            'Avg_Daily_Speed_km': round(avg_daily_speed, 1),
            'N_Reached_Antarctica': n_reached_ant,
            'Pct_Reached_Antarctica': round(pct_reached_ant, 1),
            'Mean_Days_to_Antarctica': round(mean_arrival_time, 1) if not np.isnan(mean_arrival_time) else np.nan,
            'Pct_Ice_Encounter': round(pct_encounter, 1) if not np.isnan(pct_encounter) else np.nan,
            'Pct_Ice_Trapped': round(pct_trapped, 1) if not np.isnan(pct_trapped) else np.nan,
        })
        
        # Free memory
        del site_lons, site_lats
        gc.collect()
    
    # Create DataFrame
    df = pd.DataFrame(summary_results)
    
    # Save to CSV
    df.to_csv(f'{OUTPUT_DIR}/site_summary_statistics.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR}/site_summary_statistics.csv")
    
    # Print summary table
    print("\n" + "-"*120)
    print(df.to_string(index=False))
    print("-"*120)
    
    return df


# =============================================================================
# DETAILED ANTARCTICA ARRIVAL BY SECTOR
# =============================================================================

def analyze_antarctica_sectors(ds):
    """
    Analyze Antarctica arrivals by sector for each release site.
    Returns a DataFrame with counts and percentages.
    """
    
    print("\n" + "="*70)
    print("DETAILED ANTARCTICA ARRIVAL BY SECTOR")
    print("="*70)
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_lons = ds.lon.values.copy()
    all_lats = ds.lat.values
    
    # Convert longitude to -180/180 if needed
    if np.nanmax(all_lons) > 180:
        all_lons = np.where(all_lons > 180, all_lons - 360, all_lons)
    
    sector_results = []
    
    for site_id, site_info in RELEASE_SITES.items():
        site_mask = origin_markers == site_id
        if not np.any(site_mask):
            continue
        
        site_lons = all_lons[site_mask, :]
        site_lats = all_lats[site_mask, :]
        n_particles = site_lons.shape[0]
        
        site_result = {
            'Site': site_info['name'],
            'N_Particles': n_particles,
        }
        
        total_reached_any = 0
        
        for sector_name, sector_info in ANTARCTIC_SECTORS.items():
            reached = np.zeros(n_particles, dtype=bool)
            
            for i in range(n_particles):
                for j in range(site_lons.shape[1]):
                    lon_val = site_lons[i, j]
                    lat_val = site_lats[i, j]
                    
                    if np.isnan(lon_val) or np.isnan(lat_val):
                        continue
                    
                    # Check if in this sector
                    if is_in_region(lon_val, lat_val, sector_info):
                        reached[i] = True
                        break
            
            n_reached = np.sum(reached)
            pct_reached = 100 * n_reached / n_particles
            
            site_result[f'{sector_name}_N'] = n_reached
            site_result[f'{sector_name}_Pct'] = round(pct_reached, 1)
            
            if n_reached > 0:
                total_reached_any += n_reached
        
        sector_results.append(site_result)
    
    df = pd.DataFrame(sector_results)
    df.to_csv(f'{OUTPUT_DIR}/antarctica_sectors_detailed.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR}/antarctica_sectors_detailed.csv")
    
    # Print summary
    print("\nAntarctica arrivals by sector:")
    for col in df.columns:
        if col.endswith('_Pct'):
            sector = col.replace('_Pct', '')
            print(f"  {sector}:")
            for _, row in df.iterrows():
                n_col = f'{sector}_N'
                if n_col in row:
                    print(f"    {row['Site']:>20s}: {row[n_col]:>4.0f} ({row[col]:>5.1f}%)")
    
    return df


# =============================================================================
# REGIONAL CONNECTIVITY ANALYSIS
# =============================================================================

def analyze_connectivity(ds):
    """Analyze connectivity between release sites and Antarctic regions"""
    
    print("\n" + "="*70)
    print("REGIONAL CONNECTIVITY ANALYSIS")
    print("="*70)
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_lons = ds.lon.values
    all_lats = ds.lat.values
    
    connectivity_results = []
    
    for site_id, site_info in RELEASE_SITES.items():
        print(f"\n{site_info['name']}:")
        
        # Get particles from this site
        site_mask = origin_markers == site_id
        
        if not np.any(site_mask):
            print("  No particles found")
            continue
        
        site_lons = all_lons[site_mask, :]
        site_lats = all_lats[site_mask, :]
        
        # Check connectivity to each Antarctic region
        for region_name, region_info in REGIONS.items():
            # Count particles that reached this region
            reached = np.zeros(site_lons.shape[0], dtype=bool)
            
            for i in range(site_lons.shape[0]):
                for j in range(site_lons.shape[1]):
                    if not np.isnan(site_lons[i, j]) and not np.isnan(site_lats[i, j]):
                        if is_in_region(site_lons[i, j], site_lats[i, j], region_info):
                            reached[i] = True
                            break
            
            n_reached = np.sum(reached)
            pct_reached = 100 * n_reached / len(reached)
            
            print(f"  {region_name:>20s}: {n_reached:>4d} / {len(reached):>4d} ({pct_reached:>5.1f}%)")
            
            connectivity_results.append({
                'site_id': site_id,
                'site_name': site_info['name'],
                'region': region_name,
                'n_particles': len(reached),
                'n_reached': n_reached,
                'pct_reached': pct_reached,
            })
    
    # Save results
    df = pd.DataFrame(connectivity_results)
    df.to_csv(f'{OUTPUT_DIR}/connectivity_summary.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR}/connectivity_summary.csv")
    
    return df

# =============================================================================
# LAND-TO-LAND CONNECTIVITY ANALYSIS
# =============================================================================

def analyze_land_connectivity(ds, buffer_deg=0.5):
    """Analyze connectivity between release sites and all landmasses/islands"""
    
    print("\n" + "="*70)
    print("LAND-TO-LAND CONNECTIVITY ANALYSIS")
    print(f"(buffer = {buffer_deg}° from landmass boundaries)")
    print("="*70)
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_lons = ds.lon.values
    all_lats = ds.lat.values
    
    # Normalize longitudes to -180/180
    all_lons_norm = np.where(all_lons > 180, all_lons - 360, all_lons)
    
    # Create connectivity matrix: source site -> destination landmass
    connectivity_matrix = {}
    detailed_results = []
    
    for site_id, site_info in RELEASE_SITES.items():
        site_name = site_info['name']
        print(f"\nFrom {site_name}:")
        
        # Get particles from this site
        site_mask = origin_markers == site_id
        
        if not np.any(site_mask):
            print("  No particles found")
            continue
        
        site_lons = all_lons_norm[site_mask, :]
        site_lats = all_lats[site_mask, :]
        n_particles = site_lons.shape[0]
        
        connectivity_matrix[site_name] = {}
        
        # Check connectivity to each landmass
        for land_name, land_info in LANDMASSES.items():
            # Skip if this is the source landmass
            if land_name.replace('_', ' ') in site_name.replace('_', ' '):
                continue
            
            # Count particles that reached this landmass
            reached = np.zeros(n_particles, dtype=bool)
            first_arrival_time = np.full(n_particles, np.nan)
            
            for i in range(n_particles):
                for j in range(site_lons.shape[1]):
                    if not np.isnan(site_lons[i, j]) and not np.isnan(site_lats[i, j]):
                        if is_near_landmass(site_lons[i, j], site_lats[i, j], land_info, buffer_deg):
                            reached[i] = True
                            first_arrival_time[i] = j  # timestep of first arrival
                            break
            
            n_reached = np.sum(reached)
            pct_reached = 100 * n_reached / n_particles
            
            if n_reached > 0:
                mean_arrival = np.nanmean(first_arrival_time[reached])
                print(f"  → {land_name:25s}: {n_reached:>4d} ({pct_reached:>5.1f}%), avg arrival: day {mean_arrival:.0f}")
                
                connectivity_matrix[site_name][land_name] = pct_reached
                
                detailed_results.append({
                    'source': site_name,
                    'destination': land_name,
                    'dest_type': land_info['type'],
                    'n_particles': n_particles,
                    'n_reached': n_reached,
                    'pct_reached': pct_reached,
                    'mean_arrival_day': mean_arrival,
                })
    
    # Save detailed results
    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df = df.sort_values(['source', 'pct_reached'], ascending=[True, False])
        df.to_csv(f'{OUTPUT_DIR}/land_connectivity_detailed.csv', index=False)
        print(f"\n✓ Saved: {OUTPUT_DIR}/land_connectivity_detailed.csv")
        
        # Create connectivity matrix plot
        create_connectivity_matrix_plot(df)
        
        return df
    
    return None


def create_connectivity_matrix_plot(df):
    """Create a heatmap of land-to-land connectivity"""
    
    print("\nCreating connectivity matrix plot...")
    
    # Pivot to matrix form
    sources = df['source'].unique()
    destinations = df['destination'].unique()
    
    # Create matrix
    matrix = np.zeros((len(sources), len(destinations)))
    for i, src in enumerate(sources):
        for j, dst in enumerate(destinations):
            match = df[(df['source'] == src) & (df['destination'] == dst)]
            if len(match) > 0:
                matrix[i, j] = match['pct_reached'].values[0]
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=min(50, matrix.max()))
    
    # Labels
    ax.set_xticks(np.arange(len(destinations)))
    ax.set_yticks(np.arange(len(sources)))
    ax.set_xticklabels(destinations, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(sources, fontsize=9)
    
    ax.set_xlabel('Destination Landmass', fontsize=12)
    ax.set_ylabel('Source Site', fontsize=12)
    ax.set_title('Land-to-Land Connectivity (% particles reaching)', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Particles Reaching (%)', fontsize=10)
    
    # Add text annotations for significant connections
    for i in range(len(sources)):
        for j in range(len(destinations)):
            if matrix[i, j] > 1:  # Only annotate if >1%
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                              ha='center', va='center', fontsize=6,
                              color='white' if matrix[i, j] > 25 else 'black')
    
    plt.tight_layout()
    
    # Save as PDF
    pdf_path = f'{OUTPUT_DIR}/land_connectivity_matrix.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    plt.close()

# =============================================================================
# SEASONAL ANALYSIS
# =============================================================================

def analyze_seasonality(ds):
    """Analyze seasonal patterns in dispersal"""
    
    print("\n" + "="*70)
    print("SEASONAL ANALYSIS")
    print("="*70)
    
    # Get time information
    time_vals = ds.time.values
    
    # Convert to datetime if needed
    if not np.issubdtype(time_vals.dtype, np.datetime64):
        print("  Time is not in datetime format, attempting conversion...")
        try:
            # Assume time is in seconds since epoch
            time_flat = time_vals.flatten()
            valid_times = time_flat[~np.isnan(time_flat)]
            if len(valid_times) > 0:
                # Create datetime array
                base_time = datetime(2020, 1, 1)  # Arbitrary start
                time_vals = np.array([[base_time + timedelta(seconds=float(t)) if not np.isnan(t) else pd.NaT 
                                      for t in row] for row in time_vals])
        except Exception as e:
            print(f"  Could not convert time: {e}")
            return None
    
    # Flatten and get valid times
    time_flat = time_vals.flatten()
    if np.issubdtype(time_flat.dtype, np.datetime64):
        valid_times = time_flat[~pd.isna(time_flat)]
    else:
        return None
    
    if len(valid_times) == 0:
        print("  No valid times found")
        return None
    
    # Convert to pandas datetime for easier month extraction
    times_pd = pd.to_datetime(valid_times)
    months = times_pd.month
    
    # Get corresponding positions
    all_lons = ds.lon.values.flatten()
    all_lats = ds.lat.values.flatten()
    
    # Keep only valid positions
    valid_mask = ~np.isnan(all_lons) & ~np.isnan(all_lats) & ~pd.isna(time_flat)
    all_lons = all_lons[valid_mask]
    all_lats = all_lats[valid_mask]
    months_valid = months[:len(all_lons)]
    
    # Define seasons
    seasons = {
        'DJF': [12, 1, 2],   # Summer
        'MAM': [3, 4, 5],    # Autumn
        'JJA': [6, 7, 8],    # Winter
        'SON': [9, 10, 11],  # Spring
    }
    
    seasonal_results = []
    
    for season_name, season_months in seasons.items():
        season_mask = np.isin(months_valid, season_months)
        
        if not np.any(season_mask):
            continue
        
        season_lats = all_lats[season_mask]
        
        # Calculate statistics
        n_antarctic = np.sum(season_lats < ANTARCTICA_LAT)
        pct_antarctic = 100 * n_antarctic / len(season_lats)
        mean_lat = np.mean(season_lats)
        
        print(f"  {season_name}: {pct_antarctic:>5.1f}% south of {ANTARCTICA_LAT}°S, mean lat = {mean_lat:.1f}°S")
        
        seasonal_results.append({
            'season': season_name,
            'n_observations': len(season_lats),
            'n_antarctic': n_antarctic,
            'pct_antarctic': pct_antarctic,
            'mean_latitude': mean_lat,
        })
    
    # Save and plot
    if seasonal_results:
        df = pd.DataFrame(seasonal_results)
        df = df.set_index('season').reindex(['DJF', 'MAM', 'JJA', 'SON'])
        df.to_csv(f'{OUTPUT_DIR}/seasonal_patterns.csv')
        print(f"\n✓ Saved: {OUTPUT_DIR}/seasonal_patterns.csv")
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        seasons_order = ['DJF', 'MAM', 'JJA', 'SON']
        x = np.arange(len(seasons_order))
        
        # Antarctic percentages
        ax1.bar(x, df.loc[seasons_order, 'pct_antarctic'], color='steelblue', alpha=0.7)
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Percentage in Antarctic Waters (%)')
        ax1.set_title('Seasonal Antarctic Presence')
        ax1.set_xticks(x)
        ax1.set_xticklabels(seasons_order)
        ax1.grid(axis='y', alpha=0.3)
        
        # Mean latitude
        ax2.plot(x, df.loc[seasons_order, 'mean_latitude'], marker='o', linewidth=2, markersize=8, color='darkblue')
        ax2.axhline(y=ANTARCTICA_LAT, color='red', linestyle='--', alpha=0.5, label=f'{ANTARCTICA_LAT}°S')
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Mean Latitude (°S)')
        ax2.set_title('Seasonal Mean Latitude')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seasons_order)
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/seasonal_patterns.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/seasonal_patterns.png")
        plt.close()
        
        return df
    
    return None

# =============================================================================
# ANTARCTICA CONNECTIVITY
# =============================================================================

def analyze_antarctica_connectivity(ds):
    """Analyze which sites connect to Antarctica"""
    
    print("\n" + "="*70)
    print("ANTARCTICA CONNECTIVITY")
    print("="*70)
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_lats = ds.lat.values
    
    antarctica_results = []
    
    for site_id, site_info in RELEASE_SITES.items():
        site_mask = origin_markers == site_id
        
        if not np.any(site_mask):
            continue
        
        site_lats = all_lats[site_mask, :]
        
        # Check if any particle reached Antarctica
        reached_antarctica = np.any(site_lats < ANTARCTICA_LAT, axis=1)
        n_reached = np.sum(reached_antarctica)
        pct_reached = 100 * n_reached / len(reached_antarctica)
        
        # Calculate minimum latitude reached
        min_lat = np.nanmin(site_lats)
        
        # Time to reach Antarctica (first crossing)
        time_to_antarctica = []
        for i in range(site_lats.shape[0]):
            crossing_indices = np.where(site_lats[i, :] < ANTARCTICA_LAT)[0]
            if len(crossing_indices) > 0:
                time_to_antarctica.append(crossing_indices[0])
        
        mean_time = np.mean(time_to_antarctica) if time_to_antarctica else np.nan
        
        print(f"  {site_info['name']:>20s}: {n_reached:>4d} / {len(reached_antarctica):>4d} ({pct_reached:>5.1f}%) "
              f"reached Antarctica, min lat = {min_lat:.1f}°S")
        
        antarctica_results.append({
            'site_id': site_id,
            'site_name': site_info['name'],
            'site_region': site_info['region'],
            'n_particles': len(reached_antarctica),
            'n_reached_antarctica': n_reached,
            'pct_reached_antarctica': pct_reached,
            'min_latitude': min_lat,
            'mean_time_to_antarctica_obs': mean_time,
        })
    
    # Save results
    df = pd.DataFrame(antarctica_results)
    df = df.sort_values('pct_reached_antarctica', ascending=False)
    df.to_csv(f'{OUTPUT_DIR}/antarctica_arrivals.csv', index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR}/antarctica_arrivals.csv")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    bars = ax.bar(x, df['pct_reached_antarctica'], color='steelblue', alpha=0.7)
    
    # Color by region
    region_colors = {'Atlantic': 'red', 'Indian': 'green', 'Pacific': 'blue', 'Unknown': 'gray'}
    for i, (idx, row) in enumerate(df.iterrows()):
        bars[i].set_color(region_colors.get(row['site_region'], 'gray'))
    
    ax.set_xlabel('Release Site')
    ax.set_ylabel('Percentage Reaching Antarctica (%)')
    ax.set_title('Antarctic Connectivity by Release Site')
    ax.set_xticks(x)
    ax.set_xticklabels(df['site_name'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=region) 
                      for region, color in region_colors.items() 
                      if region in df['site_region'].values]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/antarctica_connectivity.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/antarctica_connectivity.png")
    plt.close()
    
    return df

# =============================================================================
# DENSITY MAP (HANDLES ALL PARTICLES EFFICIENTLY)
# =============================================================================

def create_density_map(ds):
    """
    Create a density heatmap showing ALL particle positions.
    This is memory-efficient and works with millions of particles.
    Uses 2D histogram binning instead of plotting individual points.
    """
    
    print("\n" + "="*70)
    print("CREATING PARTICLE DENSITY MAP (ALL PARTICLES)")
    print("="*70)
    
    # Load ROMS grid for land mask
    roms_grid = load_roms_grid()
    
    # Get total particles
    n_total = ds.dims['traj']
    n_timesteps = ds.dims['obs']
    print(f"  Processing {n_total:,} particles × {n_timesteps} timesteps", flush=True)
    
    # Define grid for density calculation
    lon_bins = np.arange(-180, 181, DENSITY_RESOLUTION)
    lat_bins = np.arange(-90, NORTHERN_LIMIT + 1, DENSITY_RESOLUTION)
    
    # Initialize density array
    density = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.float64)
    
    # Process in chunks to manage memory
    chunk_size = min(CHUNK_SIZE, n_total)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    
    print(f"  Processing in {n_chunks} chunks of {chunk_size:,} particles...", flush=True)
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_total)
        
        if (chunk_idx + 1) % 5 == 0 or chunk_idx == n_chunks - 1:
            print(f"    Chunk {chunk_idx + 1}/{n_chunks} (particles {start_idx:,}-{end_idx:,})", flush=True)
        
        # Load chunk
        chunk_lons = ds.lon.values[start_idx:end_idx, :].flatten()
        chunk_lats = ds.lat.values[start_idx:end_idx, :].flatten()
        
        # Remove NaN and convert longitude
        valid = ~np.isnan(chunk_lons) & ~np.isnan(chunk_lats)
        chunk_lons = chunk_lons[valid]
        chunk_lats = chunk_lats[valid]
        
        # Convert 0-360 to -180/180
        chunk_lons = np.where(chunk_lons > 180, chunk_lons - 360, chunk_lons)
        
        # Add to histogram
        hist, _, _ = np.histogram2d(chunk_lats, chunk_lons, bins=[lat_bins, lon_bins])
        density += hist
        
        # Free memory
        del chunk_lons, chunk_lats
        gc.collect()
    
    print(f"  Total observations binned: {density.sum():,.0f}", flush=True)
    
    # Create figure
    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Set extent and circular boundary
    ax.set_extent([-180, 180, -90, NORTHERN_LIMIT], ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    # Create meshgrid for plotting
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Mask zero values for cleaner plot
    density_masked = np.ma.masked_where(density == 0, density)
    
    # Plot density with log scale for better visualization
    pcm = ax.pcolormesh(lon_mesh, lat_mesh, density_masked,
                        transform=ccrs.PlateCarree(),
                        cmap='hot_r',
                        norm=LogNorm(vmin=1, vmax=density.max()),
                        shading='auto',
                        zorder=2)
    
    # Add land
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=3)
    ax.coastlines(resolution='50m', linewidth=0.5, zorder=4)
    
    # Add Antarctic circle
    antarctic_theta = np.linspace(0, 2*np.pi, 100)
    antarctic_lons = np.degrees(antarctic_theta)
    antarctic_lats = np.full_like(antarctic_lons, ANTARCTICA_LAT)
    ax.plot(antarctic_lons, antarctic_lats,
            transform=ccrs.PlateCarree(),
            color='white', linewidth=2, linestyle='--', zorder=5)
    ax.text(0, ANTARCTICA_LAT + 2, f'{abs(ANTARCTICA_LAT):.0f}°S',
            transform=ccrs.PlateCarree(),
            ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')
    
    # Add release sites
    n_sites = len(RELEASE_SITES)
    colors = [SITE_COLORS[i % len(SITE_COLORS)] for i in range(n_sites)]
    
    for idx, (site_id, site_info) in enumerate(RELEASE_SITES.items()):
        plot_lon = site_info['lon']
        if plot_lon > 180:
            plot_lon = plot_lon - 360
        ax.plot(plot_lon, site_info['lat'],
                transform=ccrs.PlateCarree(),
                marker='*', markersize=12, color=colors[idx],
                markeredgecolor='white', markeredgewidth=1,
                label=site_info['name'], zorder=6)
    
    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, shrink=0.6, pad=0.05, extend='max')
    cbar.set_label('Particle Density (observations per grid cell)', fontsize=11)
    
    ax.set_title(f'Particle Density Map - {n_total:,} Particles\n(All timesteps combined)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    plt.tight_layout()
    
    # Save as PDF
    pdf_path = f'{OUTPUT_DIR}/particle_density_map.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    plt.close()
    
    # Also save density data
    density_df = pd.DataFrame({
        'total_observations': [density.sum()],
        'max_density': [density.max()],
        'grid_resolution_deg': [DENSITY_RESOLUTION],
    })
    density_df.to_csv(f'{OUTPUT_DIR}/density_stats.csv', index=False)
    
    return density


# =============================================================================
# STATIC MAP (SAMPLED TRAJECTORIES)
# =============================================================================

def create_static_map(ds):
    """Create static map of sampled trajectories using ROMS grid with current background"""
    
    print("\n" + "="*70)
    print("CREATING STATIC MAP (SAMPLED TRAJECTORIES)")
    print("="*70)
    
    # Load ROMS grid and currents
    roms_grid = load_roms_grid()
    currents = load_currents()  # Use first timestep for background
    
    # Use Antarctic Polar Stereographic projection
    proj = ccrs.SouthPolarStereo()
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Set extent to Southern Ocean, cropped at NORTHERN_LIMIT
    ax.set_extent([-180, 180, -90, NORTHERN_LIMIT], ccrs.PlateCarree())
    
    # Set circular boundary
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    # Add ROMS grid features with currents as background
    pcm = add_roms_features(ax, roms_grid, add_bathy=False, add_land=True, add_coastline=True,
                           currents=currents, add_currents=True)
    
    # Add colorbar for currents
    if pcm is not None:
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.6, pad=0.05)
        cbar.set_label('Current Speed (m/s)', fontsize=10)
    
    # NO gridlines - only Antarctic circle
    # Add Antarctic boundary circle at -60°S
    antarctic_theta = np.linspace(0, 2*np.pi, 100)
    antarctic_lons = np.degrees(antarctic_theta)
    antarctic_lats = np.full_like(antarctic_lons, ANTARCTICA_LAT)
    ax.plot(antarctic_lons, antarctic_lats, 
           transform=ccrs.PlateCarree(),
           color='darkgray', linewidth=1.5, linestyle='--', zorder=4)
    ax.text(0, ANTARCTICA_LAT + 2, f'{abs(ANTARCTICA_LAT):.0f}°S',
           transform=ccrs.PlateCarree(),
           ha='center', va='bottom', fontsize=9, color='darkgray')
    
    # Plot trajectories by origin site
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_lons = ds.lon.values.copy()
    all_lats = ds.lat.values
    
    # Convert longitude from 0-360 to -180 to 180 if needed
    if np.nanmax(all_lons) > 180:
        all_lons = np.where(all_lons > 180, all_lons - 360, all_lons)
    
    # Use custom color palette
    n_sites = len(RELEASE_SITES)
    colors = [SITE_COLORS[i % len(SITE_COLORS)] for i in range(n_sites)]
    
    for idx, (site_id, site_info) in enumerate(RELEASE_SITES.items()):
        site_mask = origin_markers == site_id
        
        if not np.any(site_mask):
            continue
        
        site_lons = all_lons[site_mask, :]
        site_lats = all_lats[site_mask, :]
        
        # Plot sample of trajectories (plotting all can be slow)
        n_sample = min(100, site_lons.shape[0])
        sample_idx = np.random.choice(site_lons.shape[0], n_sample, replace=False)
        
        for i in sample_idx:
            valid = ~np.isnan(site_lons[i, :]) & ~np.isnan(site_lats[i, :])
            if np.any(valid):
                ax.plot(site_lons[i, valid], site_lats[i, valid], 
                       transform=ccrs.PlateCarree(),
                       color=colors[idx], alpha=0.3, linewidth=0.5, zorder=3)
        
        # Plot release site (convert from 0-360 to -180 to 180 if needed)
        plot_lon = site_info['lon']
        if plot_lon > 180:
            plot_lon = plot_lon - 360
        ax.plot(plot_lon, site_info['lat'], 
               transform=ccrs.PlateCarree(),
               marker='*', markersize=15, color=colors[idx], 
               markeredgecolor='black', markeredgewidth=1,
               label=site_info['name'], zorder=5)
    
    ax.set_title('Kelp Dispersal Trajectories - 3 Year Simulation', fontsize=16, pad=20)
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    plt.tight_layout()
    
    # Save as PDF
    pdf_path = f'{OUTPUT_DIR}/trajectories_map_static.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Saved: {pdf_path}")
    plt.close()

# =============================================================================
# ANIMATION
# =============================================================================

def create_animation(ds, duration_seconds=30, fps=10):
    """Create animation of particle dispersal using ROMS grid with currents background"""
    
    print("\n" + "="*70)
    print("CREATING ANIMATION")
    print("="*70)
    
    # Load ROMS grid and currents
    roms_grid = load_roms_grid()
    currents = load_currents()  # Use first timestep for static background
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_lons = ds.lon.values
    all_lats = ds.lat.values
    n_timesteps = all_lons.shape[1]
    
    # Convert longitude from 0-360 to -180 to 180 if needed
    if np.nanmax(all_lons) > 180:
        all_lons = np.where(all_lons > 180, all_lons - 360, all_lons)
    
    # Calculate frames needed
    total_frames = duration_seconds * fps
    frame_step = max(1, n_timesteps // total_frames)
    frames = range(0, n_timesteps, frame_step)
    
    print(f"  Total timesteps: {n_timesteps}")
    print(f"  Frame step: {frame_step}")
    print(f"  Total frames: {len(frames)}")
    
    # Setup figure
    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(14, 12))
    ax = plt.axes(projection=proj)
    ax.set_extent([-180, 180, -90, -30], ccrs.PlateCarree())
    
    # Add ROMS features with currents background
    pcm = add_roms_features(ax, roms_grid, add_bathy=False, add_land=True, add_coastline=True,
                           currents=currents, add_currents=True)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Add colorbar for currents
    if pcm is not None:
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.5, pad=0.05)
        cbar.set_label('Current Speed (m/s)', fontsize=10)
    
    # Antarctic boundary
    antarctic_lons = np.linspace(-180, 180, 100)
    antarctic_lats = np.full_like(antarctic_lons, ANTARCTICA_LAT)
    ax.plot(antarctic_lons, antarctic_lats, 
           transform=ccrs.PlateCarree(),
           color='red', linewidth=2, linestyle='--', zorder=4)
    
    # Colors for sites
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(RELEASE_SITES)))
    
    # Plot release sites
    for idx, (site_id, site_info) in enumerate(RELEASE_SITES.items()):
        ax.plot(site_info['lon'], site_info['lat'], 
               transform=ccrs.PlateCarree(),
               marker='*', markersize=15, color=colors[idx], 
               markeredgecolor='black', markeredgewidth=1, zorder=5)
    
    # Initialize scatter plot for particles
    scatter_plots = []
    for idx, site_id in enumerate(RELEASE_SITES.keys()):
        scat = ax.scatter([], [], transform=ccrs.PlateCarree(),
                         s=5, color=colors[idx], alpha=0.6, zorder=3)
        scatter_plots.append((site_id, scat))
    
    title = ax.set_title('', fontsize=14)
    
    def init():
        for site_id, scat in scatter_plots:
            scat.set_offsets(np.empty((0, 2)))
        title.set_text('Time: 0')
        return [scat for _, scat in scatter_plots] + [title]
    
    def animate(frame_idx):
        t = frames[frame_idx]
        
        for idx, (site_id, scat) in enumerate(scatter_plots):
            site_mask = origin_markers == site_id
            
            if np.any(site_mask):
                lons = all_lons[site_mask, t]
                lats = all_lats[site_mask, t]
                
                valid = ~np.isnan(lons) & ~np.isnan(lats)
                if np.any(valid):
                    points = np.column_stack([lons[valid], lats[valid]])
                    scat.set_offsets(points)
                else:
                    scat.set_offsets(np.empty((0, 2)))
        
        title.set_text(f'Timestep: {t} / {n_timesteps}')
        return [scat for _, scat in scatter_plots] + [title]
    
    print("  Creating animation (this may take a few minutes)...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(frames), interval=1000/fps,
                                  blit=True)
    
    # Save animation
    print("  Saving animation...")
    anim.save(f'{OUTPUT_DIR}/trajectories_animation.mp4', 
             writer='ffmpeg', fps=fps, dpi=150)
    print(f"✓ Saved: {OUTPUT_DIR}/trajectories_animation.mp4")
    plt.close()

# =============================================================================
# ICE INTERACTIONS
# =============================================================================

def analyze_ice_interactions(ds):
    """Analyze sea ice encounters"""
    
    print("\n" + "="*70)
    print("SEA ICE INTERACTION ANALYSIS")
    print("="*70)
    
    if 'ice_concentration' not in ds.variables:
        print("  No ice concentration data in simulation results")
        return None
    
    origin_markers = ds.origin_marker.values
    if origin_markers.ndim > 1:
        origin_markers = origin_markers[:, 0]
    
    all_ice_conc = ds.ice_concentration.values
    
    ice_results = []
    
    print("\nIce encounter statistics by site:")
    
    for site_id, site_info in RELEASE_SITES.items():
        site_mask = origin_markers == site_id
        
        if not np.any(site_mask):
            continue
        
        # Get ice concentration for this site's trajectories
        ice_conc = all_ice_conc[site_mask, :]
        
        # Flatten and remove NaNs
        ice_conc = ice_conc.flatten()
        valid = ~np.isnan(ice_conc)
        ice_conc = ice_conc[valid]
        
        if len(ice_conc) == 0:
            continue
        
        # Statistics
        n_encounter_ice = np.sum(ice_conc > 0)
        n_significant_ice = np.sum(ice_conc > 0.15)
        n_trapped = np.sum(ice_conc > 0.80)
        
        pct_encounter = 100 * n_encounter_ice / len(ice_conc)
        pct_significant = 100 * n_significant_ice / len(ice_conc)
        pct_trapped = 100 * n_trapped / len(ice_conc)
        
        mean_ice = np.mean(ice_conc[ice_conc > 0]) if n_encounter_ice > 0 else 0
        
        print(f"  {site_info['name']:>20s}: "
              f"{pct_encounter:>5.1f}% encounter, "
              f"{pct_significant:>5.1f}% significant (>15%), "
              f"{pct_trapped:>5.1f}% trapped (>80%)")
        
        ice_results.append({
            'site_id': site_id,
            'site_name': site_info['name'],
            'pct_encounter_ice': pct_encounter,
            'pct_significant_ice': pct_significant,
            'pct_trapped': pct_trapped,
            'mean_ice_when_present': mean_ice * 100,
        })
    
    # Save results
    if ice_results:
        df = pd.DataFrame(ice_results)
        df.to_csv(f'{OUTPUT_DIR}/ice_interactions.csv', index=False)
        print(f"\n✓ Saved: {OUTPUT_DIR}/ice_interactions.csv")
        
        # Plot ice encounters by site
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['pct_encounter_ice'], width, 
               label='Any Ice (>0%)', color='lightblue', alpha=0.8)
        ax.bar(x, df['pct_significant_ice'], width,
               label='Significant Ice (>15%)', color='steelblue', alpha=0.8)
        ax.bar(x + width, df['pct_trapped'], width,
               label='Trapped (>80%)', color='darkblue', alpha=0.8)
        
        ax.set_xlabel('Release Site')
        ax.set_ylabel('Percentage of Timesteps (%)')
        ax.set_title('Sea Ice Encounters by Release Site')
        ax.set_xticks(x)
        ax.set_xticklabels(df['site_name'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/ice_interactions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {OUTPUT_DIR}/ice_interactions.png")
        plt.close()
        
        return df
    
    return None


# =============================================================================
# SEASONAL ICE ANALYSIS
# =============================================================================

def analyze_seasonal_ice(ds):
    """Analyze ice encounters by season"""
    
    print("\n" + "="*70)
    print("SEASONAL ICE ANALYSIS")
    print("="*70)
    
    if 'ice_concentration' not in ds.variables:
        print("  No ice data in simulation results")
        return None
    
    # Get all times
    times = pd.to_datetime(ds.time.values.flatten())
    months = np.array([t.month for t in times])
    
    # Define seasons
    seasons = {
        'DJF': [12, 1, 2],   # Summer
        'MAM': [3, 4, 5],     # Autumn
        'JJA': [6, 7, 8],     # Winter
        'SON': [9, 10, 11],   # Spring
    }
    
    seasonal_results = []
    
    for season_name, season_months in seasons.items():
        # Get data for this season
        season_mask = np.isin(months, season_months)
        
        if not np.any(season_mask):
            continue
        
        ice_conc = ds.ice_concentration.values.flatten()[season_mask]
        valid = ~np.isnan(ice_conc)
        ice_conc = ice_conc[valid]
        
        if len(ice_conc) == 0:
            continue
        
        n_ice = np.sum(ice_conc > 0)
        n_significant = np.sum(ice_conc > 0.15)
        n_trapped = np.sum(ice_conc > 0.80)
        
        mean_ice = np.mean(ice_conc[ice_conc > 0]) if n_ice > 0 else 0
        
        seasonal_results.append({
            'season': season_name,
            'pct_ice_present': 100 * n_ice / len(ice_conc),
            'pct_significant': 100 * n_significant / len(ice_conc),
            'pct_trapped': 100 * n_trapped / len(ice_conc),
            'mean_ice_concentration': mean_ice * 100,
        })
        
        print(f"  {season_name}: "
              f"{seasonal_results[-1]['pct_ice_present']:>5.1f}% ice present, "
              f"mean {mean_ice*100:>5.1f}% when present")
    
    if seasonal_results:
        df = pd.DataFrame(seasonal_results)
        df = df.set_index('season').reindex(['DJF', 'MAM', 'JJA', 'SON'])
        df.to_csv(f'{OUTPUT_DIR}/seasonal_ice.csv')
        print(f"\n✓ Saved: {OUTPUT_DIR}/seasonal_ice.csv")
        
        return df
    
    return None

def main():
    import os
    
    print("="*70)
    print("KELP DISPERSAL POST-ANALYSIS")
    print("Southern Ocean / Antarctica Connectivity Study")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data (this also populates RELEASE_SITES)
    ds = load_results()
    
    # Check if we have sites
    if not RELEASE_SITES:
        print("\nERROR: No release sites detected in the data!")
        print("Check that origin_marker, origin_lon, origin_lat are in the zarr file.")
        return
    
    print(f"\nDetected {len(RELEASE_SITES)} release sites:")
    for site_id, info in RELEASE_SITES.items():
        print(f"  {site_id}: {info['name']} - {info.get('n_particles', '?')} particles")
    
    # NEW: Comprehensive summary statistics by release site
    summary_df = compute_site_summary_statistics(ds)
    
    # NEW: Detailed Antarctica arrival by sector
    sector_df = analyze_antarctica_sectors(ds)
    
    # Regional connectivity
    connectivity_df = analyze_connectivity(ds)
    
    # Land-to-land connectivity (islands and continents)
    land_connectivity_df = analyze_land_connectivity(ds, buffer_deg=0.5)
    
    # Seasonal patterns
    seasonal_df = analyze_seasonality(ds)
    
    # Antarctica connectivity
    antarctica_df = analyze_antarctica_connectivity(ds)
    
    # Ice interactions
    ice_df = analyze_ice_interactions(ds)
    
    # Seasonal ice patterns
    seasonal_ice_df = analyze_seasonal_ice(ds)
    
    # Create density map (ALL particles - memory efficient)
    create_density_map(ds)
    
    # Create static map with sampled trajectories (PDF format)
    create_static_map(ds)
    
    # Create animation
    try:
        create_animation(ds, duration_seconds=30, fps=10)
    except Exception as e:
        print(f"\n✗ Animation failed: {e}")
        print("  (ffmpeg may not be installed)")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nFiles created:")
    print("  - site_summary_statistics.csv (comprehensive stats per site)")
    print("  - antarctica_sectors_detailed.csv (arrivals by Antarctic sector)")
    print("  - connectivity_summary.csv")
    print("  - land_connectivity_detailed.csv")
    print("  - land_connectivity_matrix.pdf")
    print("  - seasonal_patterns.csv")
    print("  - seasonal_patterns.png")
    print("  - antarctica_arrivals.csv")
    print("  - antarctica_connectivity.png")
    print("  - ice_interactions.csv (if ice data present)")
    print("  - ice_interactions.png (if ice data present)")
    print("  - seasonal_ice.csv (if ice data present)")
    print("  - particle_density_map.pdf (ALL particles as heatmap)")
    print("  - trajectories_map_static.pdf (sampled trajectories)")
    print("  - trajectories_animation.mp4")


if __name__ == '__main__':
    main()