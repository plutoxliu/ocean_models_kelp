#!/usr/bin/env python3
"""
Fix ROMS forcing file by adding 'time' attributes to variables.

Instead of renaming time variables, we add a 'time' attribute to each
data variable that tells ROMS which time variable to use.

"""

import netCDF4 as nc
import os
from datetime import datetime

BULK_FILE = './roms_forcing_test/roms_frc_bulk.nc'

# Mapping: data variable -> time variable name (as it exists in the file)
VAR_TIME_MAPPING = {
    'Pair': 'Pair_time',
    'Tair': 'Tair_time',
    'Qair': 'Qair_time',
    'swrad': 'swrad_time',
    'swrad_down': 'swrad_time',
    'lwrad': 'lwrad_time',
    'lwrad_down': 'lwrad_time',
    'Uwind': 'wind_time',
    'Vwind': 'wind_time',
    'cloud': 'cloud_time',
    'rain': 'rain_time',
}

def fix_time_attributes(filepath):
    """Add time attributes to variables in forcing file."""
    
    print("="*70)
    print("ADDING TIME ATTRIBUTES TO FORCING FILE")
    print("="*70)
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found!")
        return False
    
    with nc.Dataset(filepath, 'r+') as ds:
        print(f"\nVariables: {list(ds.variables.keys())}")
        print(f"\nAdding/updating 'time' attributes:")
        
        changes = 0
        for var_name, time_var in VAR_TIME_MAPPING.items():
            if var_name in ds.variables:
                # Check if time variable exists
                if time_var not in ds.variables:
                    print(f"  ⚠ {var_name}: time var '{time_var}' not found, skipping")
                    continue
                
                # Get current time attribute if exists
                current = getattr(ds.variables[var_name], 'time', None)
                
                # Set the time attribute
                ds.variables[var_name].time = time_var
                
                if current != time_var:
                    print(f"  ✓ {var_name}.time = '{time_var}' (was: {current})")
                    changes += 1
                else:
                    print(f"  - {var_name}.time = '{time_var}' (unchanged)")
            else:
                print(f"  - {var_name}: not in file")
        
        # Update history
        history = getattr(ds, 'history', '')
        ds.history = f"{history} | Added time attributes {datetime.now()}"
        
        ds.sync()
    
    print(f"\n✓ Updated {changes} time attributes")
    
    # Verify
    print(f"\nVerification:")
    with nc.Dataset(filepath, 'r') as ds:
        for var_name in VAR_TIME_MAPPING.keys():
            if var_name in ds.variables:
                time_attr = getattr(ds.variables[var_name], 'time', 'MISSING')
                print(f"  {var_name}.time = '{time_attr}'")
    
    return True

def check_varinfo_compatibility():
    """Check if we also need to update varinfo.yaml."""
    print("\n" + "="*70)
    print("VARINFO.YAML COMPATIBILITY CHECK")
    print("="*70)
    print("""
Update varinfo.yaml to match forcing file
   Change the 'time:' entries in varinfo.yaml:
   For swrad:     time: swrad_time    (instead of srf_time)
   For lwrad:     time: lwrad_time    (instead of lrf_time)
   For Pair:      time: Pair_time     (instead of pair_time)
   For Tair:      time: Tair_time     (instead of tair_time)
   For Qair:      time: Qair_time     (instead of qair_time)
""")

def main():
    fix_time_attributes(BULK_FILE)
    check_varinfo_compatibility()
    
    print("\n" + "="*70)
    print("UPDATE VARINFO.YAML")

if __name__ == '__main__':
    main()