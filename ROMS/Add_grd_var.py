#!/usr/bin/env python3
"""
Fix ROMS grid file by adding missing variables: xl, el, spherical.

Compares to reference grid and adds missing scalar variables.
"""

import numpy as np
import netCDF4 as nc
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

GRID_FILE = './ocean_grd_fine.nc'

# =============================================================================
# FUNCTIONS
# =============================================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great circle distance in meters."""
    R = 6371000  # Earth radius in meters
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def fix_grid_file(grid_file):
    """Add missing variables to ROMS grid file."""
    
    print("="*70)
    print("FIXING ROMS GRID FILE")
    print("="*70)
    print(f"File: {grid_file}")
    
    if not os.path.exists(grid_file):
        print(f"ERROR: File not found: {grid_file}")
        return False
    
    with nc.Dataset(grid_file, 'r+') as ds:
        
        # Print current variables
        print(f"\nCurrent variables: {list(ds.variables.keys())}")
        
        # Get grid coordinates
        lon_rho = ds.variables['lon_rho'][:]
        lat_rho = ds.variables['lat_rho'][:]
        
        eta_rho, xi_rho = lon_rho.shape
        print(f"\nGrid shape: eta_rho={eta_rho}, xi_rho={xi_rho}")
        
        # Calculate domain lengths using haversine formula
        # xl: total length in XI direction (along rows)
        # el: total length in ETA direction (along columns)
        
        mid_eta = eta_rho // 2
        mid_xi = xi_rho // 2
        
        # XI-direction length (along middle row)
        xi_distances = []
        for i in range(xi_rho - 1):
            d = haversine_distance(
                lon_rho[mid_eta, i], lat_rho[mid_eta, i],
                lon_rho[mid_eta, i+1], lat_rho[mid_eta, i+1]
            )
            xi_distances.append(d)
        xl = np.sum(xi_distances)
        
        # ETA-direction length (along middle column)
        eta_distances = []
        for j in range(eta_rho - 1):
            d = haversine_distance(
                lon_rho[j, mid_xi], lat_rho[j, mid_xi],
                lon_rho[j+1, mid_xi], lat_rho[j+1, mid_xi]
            )
            eta_distances.append(d)
        el = np.sum(eta_distances)
        
        print(f"\nCalculated domain lengths:")
        print(f"  xl (XI-direction):  {xl/1000:.1f} km ({xl:.1f} m)")
        print(f"  el (ETA-direction): {el/1000:.1f} km ({el:.1f} m)")
        
        # Add xl variable (scalar, no dimensions)
        if 'xl' not in ds.variables:
            print("\nAdding variable: xl")
            var_xl = ds.createVariable('xl', 'f8')  # Scalar - no dimensions
            var_xl.long_name = 'domain length in the XI-direction'
            var_xl.units = 'meter'
            var_xl[:] = xl
        else:
            print(f"\nVariable xl already exists: {ds.variables['xl'][:]}")
            # Update value
            ds.variables['xl'][:] = xl
            print(f"  Updated to: {xl}")
        
        # Add el variable (scalar, no dimensions)
        if 'el' not in ds.variables:
            print("Adding variable: el")
            var_el = ds.createVariable('el', 'f8')  # Scalar - no dimensions
            var_el.long_name = 'domain length in the ETA-direction'
            var_el.units = 'meter'
            var_el[:] = el
        else:
            print(f"Variable el already exists: {ds.variables['el'][:]}")
            ds.variables['el'][:] = el
            print(f"  Updated to: {el}")
        
        # Add spherical variable (scalar, value=1 for spherical coordinates)
        if 'spherical' not in ds.variables:
            print("Adding variable: spherical")
            var_sph = ds.createVariable('spherical', 'i4')  # Scalar - no dimensions
            var_sph.long_name = 'grid type logical switch'
            var_sph.flag_values = '0, 1'
            var_sph.flag_meanings = 'Cartesian spherical'
            var_sph[:] = 1  # 1 = Spherical coordinates
        else:
            val = ds.variables['spherical'][:]
            print(f"Variable spherical already exists: {val}")
        
        # Update history attribute
        history = getattr(ds, 'history', '')
        ds.history = f"{history}\nAdded xl, el, spherical on {datetime.now()}"
        
        print("\n" + "-"*50)
        print("VERIFICATION")
        print("-"*50)
        print(f"  xl = {float(ds.variables['xl'][:]):.1f} m ({float(ds.variables['xl'][:])/1000:.1f} km)")
        print(f"  el = {float(ds.variables['el'][:]):.1f} m ({float(ds.variables['el'][:])/1000:.1f} km)")
        print(f"  spherical = {int(ds.variables['spherical'][:])}")
        
        # List all variables now
        print(f"\nFinal variables: {list(ds.variables.keys())}")
    
    print("\n✓ Grid file updated successfully")
    return True
    
def main():
    fix_grid_file(GRID_FILE)
    print("\n" + "="*70)
    print("DONE")
    
if __name__ == '__main__':
    main()