import netCDF4 as nc
import numpy as np
from scipy.ndimage import gaussian_filter

with nc.Dataset('ocean_grd_fine.nc', 'a') as ds:
    h = ds['h'][:]
    
    # Apply Gaussian smoothing
    h_smooth = gaussian_filter(h, sigma=3)
    
    # Enforce minimum
    h_smooth = np.maximum(h_smooth, 50.0)
    
    ds['h'][:] = h_smooth
    print(f'Smoothed depth range: {h_smooth.min():.1f} - {h_smooth.max():.1f} m')