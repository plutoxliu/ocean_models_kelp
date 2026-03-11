#!/usr/bin/env python3
"""
Fix the 360° column in ROMS grid file.
The Add_360.py script duplicated the 0° column to 360°, but bathymetry
and other fields need proper values interpolated from neighbors.

This script interpolates the 360° column (last column) from the 
second-to-last column and first column (periodic wrap).
"""

import numpy as np
import netCDF4 as nc
import shutil
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_FILE = './ocean_grd_test.nc'
CREATE_BACKUP = True

# Variables to fix by interpolation
VARS_TO_INTERPOLATE = [
    'h',           # bathymetry
    'f',           # Coriolis
    'pm', 'pn',    # grid metrics
    'dndx', 'dmde', # grid metrics
    'x_rho', 'y_rho',  # Cartesian coords
    'x_u', 'y_u',
    'x_v', 'y_v',
    'x_psi', 'y_psi',
    'angle',       # grid angle
]

# =============================================================================

def fix_grid_360_column(grid_file, backup=True):
    """Fix the 360° column by interpolating from column -2 and column 0."""
    
    print("="*60)
    print("FIXING 360° COLUMN IN GRID FILE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Backup
    if backup:
        backup_file = grid_file + '.bak'
        shutil.copy2(grid_file, backup_file)
        print(f"Backup: {backup_file}")
    
    with nc.Dataset(grid_file, 'r+') as ds:
        print(f"\nFile: {grid_file}")
        
        # Check longitude to confirm layout
        lon_rho = ds.variables['lon_rho'][:]
        print(f"\nlon_rho shape: {lon_rho.shape}")
        print(f"Column 0 lon: {lon_rho[50, 0]:.2f}°")
        print(f"Column -2 lon: {lon_rho[50, -2]:.2f}°")
        print(f"Column -1 lon: {lon_rho[50, -1]:.2f}°")
        
        # Fix each variable
        print(f"\nInterpolating columns 0 and -1 (360°):")
        for varname in VARS_TO_INTERPOLATE:
            if varname not in ds.variables:
                print(f"  {varname}: not found, skipping")
                continue
            
            var = ds.variables[varname]
            data = var[:]
            ndim = data.ndim
            
            if ndim == 2:
                # (eta, xi)
                # Interpolate col[0] and col[-1] from col[1] and col[-2]
                old_col0 = data[:, 0].mean()
                old_colN = data[:, -1].mean()
                
                # Average of column 1 and column -2
                avg_val = (data[:, 1] + data[:, -2]) / 2.0
                data[:, 0] = avg_val
                data[:, -1] = avg_val
                
                new_col0 = data[:, 0].mean()
                new_colN = data[:, -1].mean()
                print(f"  {varname}: col0 {old_col0:.4g}->{new_col0:.4g}, col-1 {old_colN:.4g}->{new_colN:.4g}")
                var[:] = data
                
            elif ndim == 3:
                # (time, eta, xi) or (s_rho, eta, xi)
                avg_val = (data[:, :, 1] + data[:, :, -2]) / 2.0
                data[:, :, 0] = avg_val
                data[:, :, -1] = avg_val
                print(f"  {varname}: fixed col 0 and -1")
                var[:] = data
        
        ds.sync()
    
    print(f"\n✓ Grid file fixed!")


def verify_grid(grid_file):
    """Quick verification of the fixed grid."""
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with nc.Dataset(grid_file, 'r') as ds:
        h = ds.variables['h'][:]
        lon = ds.variables['lon_rho'][:]
        
        print(f"\nBathymetry (h):")
        print(f"  Shape: {h.shape}")
        print(f"  Column 0 (0°) mean: {h[:, 0].mean():.1f} m")
        print(f"  Column 1 (0.5°) mean: {h[:, 1].mean():.1f} m")
        print(f"  Column -2 (359.5°) mean: {h[:, -2].mean():.1f} m")
        print(f"  Column -1 (360°) mean: {h[:, -1].mean():.1f} m")
        
        # Check continuity at wrap
        diff = np.abs(h[:, 0] - h[:, -1]).mean()
        print(f"  Mean diff col[0] vs col[-1]: {diff:.4g} m (should be ~0)")
        
        print(f"\nLongitude:")
        print(f"  Column 0: {lon[50, 0]:.2f}°")
        print(f"  Column 1: {lon[50, 1]:.2f}°")
        print(f"  Column -2: {lon[50, -2]:.2f}°")
        print(f"  Column -1: {lon[50, -1]:.2f}°")


if __name__ == '__main__':
    fix_grid_360_column(GRID_FILE, backup=CREATE_BACKUP)
    verify_grid(GRID_FILE)