import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
from pyproj import Transformer

# --- 1. SETTINGS ---
res = 1  
lon_min, lon_max = 0.0, 360.0
lat_min, lat_max = -80.0, -30.0
lon_1d = np.arange(lon_min, lon_max, res)
lat_1d = np.arange(lat_min, lat_max + res, res)

# --- 1. LOAD NSIDC EASE-GRID DATA ---
nsidc = xr.open_dataset("NSIDC_EASE2-S3.125km.nc")

# --- 2. SETUP COORDINATE TRANSFORMATION ---
# From ROMS (WGS84 Lat/Lon - EPSG:4326) to EASE-Grid 2.0 South (EPSG:6932)
transformer = Transformer.from_crs("epsg:4326", "epsg:6932", always_xy=True)

# Define your ROMS grid
lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

# Convert ROMS Lon/Lat to EASE-Grid X/Y
print("Projecting ROMS coordinates to EASE-Grid 2.0 space...")
x_roms, y_roms = transformer.transform(lon_2d, lat_2d)

# --- 3. SAMPLE THE MASK ---
# Using the variable name from your KeyError
mask_var = 'sea_ice_region_NASA_surface_mask'

print(f"Sampling {mask_var} onto ROMS grid...")

# We use DataArrays for the coordinates to make interp work correctly with 2D arrays
mask_nsidc = nsidc[mask_var].interp(
    x=xr.DataArray(x_roms, dims=("eta_rho", "xi_rho")),
    y=xr.DataArray(y_roms, dims=("eta_rho", "xi_rho")),
    method="nearest"
)

# --- 4. CONVERT TO ROMS CONVENTION (1=Water, 0=Land) ---

# Diagnostic: Let's see what values actually exist in your sampled mask
# This will print to your NeSI terminal
unique_vals = np.unique(mask_nsidc)
print(f"Unique values found in NSIDC sample: {unique_vals}")

# Anything that is NOT Land(30), Land Ice(33), or Ice Shelf(34) is Water(1)

is_land = (mask_nsidc == 30) | (mask_nsidc == 33) | (mask_nsidc == 34)
mask_rho = xr.where(is_land, 0, 1).values

# Alternative if the above still looks wrong (Force the flip):
# mask_rho = 1 - mask_rho 

# Safety: Fill any NaNs (usually off-earth or northern boundary) as Water
mask_rho = np.nan_to_num(mask_rho, nan=1).astype(np.int32)

print(f"Mask created successfully.")
print(f"Total Water points: {np.sum(mask_rho == 1)}")
print(f"Total Land/Ice points: {np.sum(mask_rho == 0)}")

# --- 3. INTERPOLATE ETOPO ---
print("Interpolating ETOPO Bathymetry...")
topo_ds = xr.open_dataset("ETOPO1_Ice_g_gmt4.nc")

# Standardize coordinate names
rename_dict = {}
for lon_name in ['longitude', 'x', 'lon']:
    if lon_name in topo_ds.coords: rename_dict[lon_name] = 'lon'
for lat_name in ['latitude', 'y', 'lat']:
    if lat_name in topo_ds.coords: rename_dict[lat_name] = 'lat'
topo_ds = topo_ds.rename(rename_dict)

# Handle longitude wrapping and DUPLICATES
if topo_ds.lon.min() < 0:
    # 1. Convert to 0-360
    topo_ds = topo_ds.assign_coords(lon=(topo_ds.lon % 360))
    # 2. Sort so coordinates are increasing
    topo_ds = topo_ds.sortby('lon')
    # 3. CRITICAL: Drop duplicate longitude values (e.g., if 0 and 360 both exist)
    topo_ds = topo_ds.drop_duplicates(dim='lon')

# Determine elevation variable name
elev_var = 'z' if 'z' in topo_ds else 'elevation'

# Perform Interpolation
# Note: method='linear' is safer for bathymetry than 'cubic'
h_raw = topo_ds[elev_var].interp(lon=lon_1d, lat=lat_1d, method='linear').values

# Convert to ROMS convention (Positive Depth)
h = -h_raw 
h[h < 10.0] = 10.0      # Minimum depth floor
h[np.isnan(h)] = 10.0   # Fill any NaN holes (e.g., at the very edge of the domain)

# --- 4. CALCULATE STAGGERED MASKS & COORDS ---
lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

# Derived Masks (U, V, PSI) 
# Note: Multiplying neighboring cells ensures that if either cell is land (0), 
# the interface between them is masked as land (0).
mask_u = mask_rho[:, :-1] * mask_rho[:, 1:]
mask_v = mask_rho[:-1, :] * mask_rho[1:, :]
mask_psi = mask_u[:-1, :] * mask_u[1:, :]

# Staggered Longitude/Latitude
# U-points are between RHO-points in the X (xi) direction
lon_u = 0.5 * (lon_2d[:, :-1] + lon_2d[:, 1:])
lat_u = 0.5 * (lat_2d[:, :-1] + lat_2d[:, 1:])

# V-points are between RHO-points in the Y (eta) direction
lon_v = 0.5 * (lon_2d[:-1, :] + lon_2d[1:, :])
lat_v = 0.5 * (lat_2d[:-1, :] + lat_2d[1:, :])

# PSI-points are the corners of the RHO-cells
lon_psi = 0.5 * (lon_u[:-1, :] + lon_u[1:, :])
lat_psi = 0.5 * (lat_v[:, :-1] + lat_v[:, 1:])

# --- 5. CALCULATE PHYSICS METRICS ---
omega = 7.2921e-5
radius = 6371000.0
# Coriolis parameter (negative in the Southern Hemisphere)
f = 2 * omega * np.sin(np.deg2rad(lat_2d)) 

# Metrics: pm = 1/dx, pn = 1/dy
dlon_rad, dlat_rad = np.deg2rad(res), np.deg2rad(res)
# dx depends on latitude because of converging meridians
pm = 1.0 / (radius * np.cos(np.deg2rad(lat_2d)) * dlon_rad)
pn = np.ones_like(pm) * (1.0 / (radius * dlat_rad))

# Metric derivatives (dndx, dmde) - ROMS uses these for gradient calculations
dndx = np.zeros_like(pn)
dmde = np.zeros_like(pm)
# Center-difference calculations for non-boundary points
dndx[:, 1:-1] = (pn[:, 2:] - pn[:, :-2]) / (2.0 / pm[:, 1:-1])
dmde[1:-1, :] = (pm[2:, :] - pm[:-2, :]) / (2.0 / pn[1:-1, :])

# --- 6. ASSEMBLE DATASET ---
ds = xr.Dataset()

# Bathymetry & Masks
ds['h'] = (('eta_rho', 'xi_rho'), h)
ds['mask_rho'] = (('eta_rho', 'xi_rho'), mask_rho)
ds['mask_u'] = (('eta_rho', 'xi_u'), mask_u)
ds['mask_v'] = (('eta_v', 'xi_rho'), mask_v)
ds['mask_psi'] = (('eta_psi', 'xi_psi'), mask_psi)

# Longitude/Latitude (2D)
ds['lon_rho'] = (('eta_rho', 'xi_rho'), lon_2d)
ds['lat_rho'] = (('eta_rho', 'xi_rho'), lat_2d)
ds['lon_u'] = (('eta_rho', 'xi_u'), lon_u)
ds['lat_u'] = (('eta_rho', 'xi_u'), lat_u)
ds['lon_v'] = (('eta_v', 'xi_rho'), lon_v)
ds['lat_v'] = (('eta_v', 'xi_rho'), lat_v)
ds['lon_psi'] = (('eta_psi', 'xi_psi'), lon_psi)
ds['lat_psi'] = (('eta_psi', 'xi_psi'), lat_psi)

# Physics Metrics
ds['f'] = (('eta_rho', 'xi_rho'), f)
ds['pm'] = (('eta_rho', 'xi_rho'), pm)
ds['pn'] = (('eta_rho', 'xi_rho'), pn)
ds['dndx'] = (('eta_rho', 'xi_rho'), dndx)
ds['dmde'] = (('eta_rho', 'xi_rho'), dmde)

# Factors used for smoothing/viscosity (default to 0 and 1)
ds['angle'] = (('eta_rho', 'xi_rho'), np.zeros_like(lon_2d))
ds['visc_factor'] = (('eta_rho', 'xi_rho'), np.ones_like(lon_2d))
ds['diff_factor'] = (('eta_rho', 'xi_rho'), np.ones_like(lon_2d))

# Final attributes
ds.attrs['title'] = f'ROMS {res} degree Southern Ocean Grid'
ds.attrs['NSIDC_mask'] = 'L1,L30=water; others=Land'
ds.attrs['Vtransform'] = 2
ds.attrs['Vstretching'] = 4
ds.attrs['N'] = 30
ds.attrs['theta_s'] = 7.0
ds.attrs['theta_b'] = 2.0
ds.attrs['Tcline'] = 200.0

# Save output
output_name = f"roms_grid_SO_{res}deg.nc"
ds.to_netcdf(output_name)
print(f"Successfully saved {output_name}")