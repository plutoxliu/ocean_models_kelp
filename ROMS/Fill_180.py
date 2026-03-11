#!/usr/bin/env python3
"""
Fix ONLY the 180° longitude gap in ROMS forcing files.
Detects gap width for each file and averages from valid neighbors.
"""

import numpy as np
import netCDF4 as nc
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

FORCING_DIR = './roms_forcing_fine'
GRID_FILE = './ocean_grd_fine.nc'

FORCING_FILES = [
    'roms_frc_currents.nc',
    'roms_frc_wave.nc', 
    'roms_frc_bulk.nc',
    'roms_frc_ocean.nc',
    'roms_frc_ice.nc',
]

CHUNK_SIZE = 100

# =============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def replace_fill_values(data):
    """Replace common fill values with NaN"""
    # Any value below -30000 (fill values like -32767, -31456.5)
    data = np.where(np.abs(data) > 1000, np.nan, data)
    # 9.96921e36 (NetCDF default)
    data = np.where(np.abs(data) > 1e30, np.nan, data)
    return data

def get_gap_center(grid_file):
    """Find the column index at 180° longitude"""
    with nc.Dataset(grid_file, 'r') as f:
        lon = f.variables['lon_rho'][:]
    
    mid_row = lon.shape[0] // 2
    lon_1d = lon[mid_row, :]
    idx_180 = np.argmin(np.abs(lon_1d - 180))
    
    log(f"Grid shape: {lon.shape}")
    log(f"Column {idx_180} = {lon_1d[idx_180]:.2f}° (180° center)")
    
    return idx_180

def detect_gap(fpath, center_col, search_width=10):
    """Detect the actual gap columns in a file by checking first data variable"""
    with nc.Dataset(fpath, 'r') as f:
        # Find first data variable
        skip = {'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v'}
        skip.update(v for v in f.variables if 'time' in v.lower())
        
        for varname in f.variables:
            if varname in skip:
                continue
            
            var = f.variables[varname]
            ndim = len(var.shape)
            
            # Get a sample slice near 180
            if ndim == 2:
                sample = var[:, center_col-search_width:center_col+search_width+1]
            elif ndim == 3:
                sample = var[0, :, center_col-search_width:center_col+search_width+1]
            elif ndim == 4:
                sample = var[0, 0, :, center_col-search_width:center_col+search_width+1]
            else:
                continue
            
            # Convert to float and handle masked/fill values
            if np.ma.isMaskedArray(sample):
                sample = np.ma.filled(sample.astype(np.float64), np.nan)
            sample = replace_fill_values(sample.astype(np.float64))
            
            # Check middle row for gaps (NaN columns)
            mid_row = sample.shape[0] // 2
            row_data = sample[mid_row, :]
            
            # Find which columns are NaN
            nan_mask = np.isnan(row_data)
            gap_indices = np.where(nan_mask)[0]
            
            if len(gap_indices) > 0:
                # Convert back to absolute column indices
                gap_start = center_col - search_width + gap_indices[0]
                gap_end = center_col - search_width + gap_indices[-1]
                log(f"  Detected gap: columns {gap_start} to {gap_end} ({len(gap_indices)} columns)")
                return gap_start, gap_end
            else:
                log(f"  No gap detected in {varname}")
                return None, None
    
    return None, None

def process_var(f, varname, gap_start, gap_end):
    """Process one variable - fill gap columns by averaging neighbors"""
    var = f.variables[varname]
    shape = var.shape
    ndim = len(shape)
    
    # Columns to use for averaging
    left_col = gap_start - 1
    right_col = gap_end + 1
    
    log(f"  {varname}: {shape}, filling cols {gap_start}-{gap_end} from cols {left_col} & {right_col}")
    
    if ndim == 2:
        data = var[:]
        if np.ma.isMaskedArray(data):
            data = np.ma.filled(data.astype(np.float64), np.nan)
        else:
            data = data.astype(np.float64)
        data = replace_fill_values(data)
        
        # Average neighbors for all gap columns
        avg = (data[:, left_col] + data[:, right_col]) / 2.0
        for col in range(gap_start, gap_end + 1):
            data[:, col] = avg
        var[:] = data
        log(f"    Fixed")
            
    elif ndim == 3:
        nt = shape[0]
        nchunks = (nt + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for c in range(nchunks):
            t0 = c * CHUNK_SIZE
            t1 = min(t0 + CHUNK_SIZE, nt)
            
            data = var[t0:t1, :, :]
            if np.ma.isMaskedArray(data):
                data = np.ma.filled(data.astype(np.float64), np.nan)
            else:
                data = data.astype(np.float64)
            data = replace_fill_values(data)
            
            # Average neighbors for all gap columns
            avg = (data[:, :, left_col] + data[:, :, right_col]) / 2.0
            for col in range(gap_start, gap_end + 1):
                data[:, :, col] = avg
            var[t0:t1, :, :] = data
            
            if (c + 1) % 100 == 0 or c == nchunks - 1:
                log(f"    Chunk {c+1}/{nchunks}")
        
    elif ndim == 4:
        nt = shape[0]
        nchunks = (nt + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for c in range(nchunks):
            t0 = c * CHUNK_SIZE
            t1 = min(t0 + CHUNK_SIZE, nt)
            
            data = var[t0:t1, :, :, :]
            if np.ma.isMaskedArray(data):
                data = np.ma.filled(data.astype(np.float64), np.nan)
            else:
                data = data.astype(np.float64)
            data = replace_fill_values(data)
            
            # Average neighbors for all gap columns
            avg = (data[:, :, :, left_col] + data[:, :, :, right_col]) / 2.0
            for col in range(gap_start, gap_end + 1):
                data[:, :, :, col] = avg
            var[t0:t1, :, :, :] = data
            
            if (c + 1) % 100 == 0 or c == nchunks - 1:
                log(f"    Chunk {c+1}/{nchunks}")

def process_file(fpath, center_col):
    """Process one forcing file"""
    fname = os.path.basename(fpath)
    log(f"\n{'='*50}")
    log(f"FILE: {fname}")
    log('='*50)
    
    # Detect gap width for this file
    gap_start, gap_end = detect_gap(fpath, center_col)
    
    if gap_start is None:
        log(f"No gap to fix, skipping")
        return
    
    with nc.Dataset(fpath, 'r+') as f:
        skip = {'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v'}
        skip.update(v for v in f.variables if 'time' in v.lower())
        
        vars_to_fix = [v for v in f.variables if v not in skip]
        log(f"Variables: {vars_to_fix}")
        
        for varname in vars_to_fix:
            process_var(f, varname, gap_start, gap_end)
        
        f.sync()
    
    log(f"✓ {fname} done!")

def main():
    log("="*50)
    log("FIXING 180° GAP IN FORCING FILES")
    log("(auto-detecting gap width per file)")
    log("="*50)
    
    center_col = get_gap_center(GRID_FILE)
    
    for fname in FORCING_FILES:
        fpath = os.path.join(FORCING_DIR, fname)
        if os.path.exists(fpath):
            process_file(fpath, center_col)
        else:
            log(f"Skip {fname} - not found")
    
    log("\n" + "="*50)
    log("ALL DONE!")
    log("="*50)

if __name__ == '__main__':
    main()