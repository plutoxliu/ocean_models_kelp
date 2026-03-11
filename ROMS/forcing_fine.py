"""
ROMS Forcing File Generator - Updated with Full Variable Set
=============================================================
Includes:
- Bulk forcing: Pair, Tair, Qair, Uwind, Vwind, cloud, lwrad, swrad, rain
- Wave forcing: Hwave, Uwave, Vwave, Dwave, Pwave, Lwave
- Ocean forcing: SST, SSS, zeta
- Current forcing: ubar, vbar
- Ice forcing: aice, hice, uice, vice

Time variable naming follows ROMS conventions:
- Pair_time, Tair_time, Qair_time - atmospheric variables
- wind_time - wind components
- lwrad_time, swrad_time - radiation
- rain_time - precipitation
- wave_time - wave variables
- ocean_time - ocean/current variables
- ice_time - sea ice

Fill values: 1.e37 (ROMS standard)
"""

import numpy as np
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import os
import sys
from multiprocessing import Pool, current_process
import warnings
import gc
import shutil
import time

warnings.filterwarnings('ignore')


# =============================================================================
# USER CONFIGURATION
# =============================================================================

ERA5_INPUT_DIR = './era5_data/'
CMEMS_INPUT_DIR = './cmems_data/'
OUTPUT_DIR = './roms_forcing_fine/'
GRID_FILE = './ocean_grd_fine_extended.nc'
N_CORES = 6

# =============================================================================
# FILE CONFIGURATION
# =============================================================================

ERA5_FILES = {
    'atmosphere': 'atom.nc',           # t2m, d2m, msl
    'radiation_down': 'sdrf_avg.nc',   # avg_sdswrf, avg_sdlwrf (downwelling)
    'radiation_net': 'snrf_avg.nc',    # avg_snswrf, avg_snlwrf (net)
    'clouds': 'tcc_instant.nc',        # tcc (total cloud cover)
    'waves': 'waves.nc',               # swh, ust, vst
    'waves_add': 'wave_add.nc',        # mwd, mwp, pp1d (direction, periods)
    'winds': 'winds.nc',               # u10, v10
    'precip': 'rain_new.nc',           # tp (total precipitation)
}

CMEMS_FILES = {
    'currents': 'cmems_phy_P1D.nc',
    'ice': 'cmems_phy_P1D_ice.nc',
    'ocean': 'cmems_phy_P1D_sos.nc',
}

OUTPUT_FILES = {
    'bulk': 'roms_frc_bulk.nc',
    'wave': 'roms_frc_wave.nc',
    'ocean': 'roms_frc_ocean.nc',
    'current': 'roms_frc_currents.nc',
    'ice': 'roms_frc_ice.nc',
}

# Constants
TIME_UNITS = 'days since 1900-01-01 00:00:00'
CALENDAR = 'gregorian'
KELVIN_OFFSET = 273.15
PA_TO_MBAR = 0.01
FILL_VALUE = 1.e37  # ROMS standard fill value
G = 9.81  # gravitational acceleration (m/s²)

PROGRESS_FILE = None


# =============================================================================
# Logging
# =============================================================================
def log(msg):
    """Log with timestamp to stdout and progress file."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"{timestamp} | {msg}"
    print(line, flush=True)
    
    global PROGRESS_FILE
    if PROGRESS_FILE:
        try:
            with open(PROGRESS_FILE, 'a') as f:
                f.write(line + '\n')
                f.flush()
                os.fsync(f.fileno())
        except:
            pass


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"


def format_size(bytes_size):
    if bytes_size < 1024**2:
        return f"{bytes_size/1024:.1f}KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size/1024**2:.1f}MB"
    else:
        return f"{bytes_size/1024**3:.2f}GB"


# =============================================================================
# Utility Functions
# =============================================================================

# Reference dates for time conversion
REFERENCE_DATE = datetime(1900, 1, 1, 0, 0, 0)
UNIX_EPOCH = datetime(1970, 1, 1, 0, 0, 0)
DAYS_1900_TO_1970 = (UNIX_EPOCH - REFERENCE_DATE).days  # 25567 days


def convert_time_to_roms(time_data, time_units):
    """
    Convert time values to ROMS format (days since 1900-01-01).
    
    Handles:
    - Unix timestamps (seconds since 1970-01-01)
    - Hours since some date
    - Days since some date
    - Seconds since some date
    """
    time_data = np.asarray(time_data, dtype=np.float64)
    
    # Check if it looks like Unix timestamps (values > 1e8)
    max_val = np.max(time_data)
    
    if max_val > 1e8:
        # Unix timestamps (seconds since 1970-01-01)
        days_since_1970 = time_data / 86400.0
        days_since_1900 = days_since_1970 + DAYS_1900_TO_1970
        return days_since_1900
    
    elif 'seconds since' in time_units.lower():
        try:
            ref_str = time_units.lower().replace('seconds since', '').strip()
            ref_date = datetime.strptime(ref_str.split()[0], '%Y-%m-%d')
            days_since_ref = time_data / 86400.0
            days_1900_to_ref = (ref_date - REFERENCE_DATE).days
            return days_since_ref + days_1900_to_ref
        except:
            return time_data / 86400.0 + DAYS_1900_TO_1970
    
    elif 'hours since' in time_units.lower():
        try:
            ref_str = time_units.lower().replace('hours since', '').strip()
            ref_date = datetime.strptime(ref_str.split()[0], '%Y-%m-%d')
            days_since_ref = time_data / 24.0
            days_1900_to_ref = (ref_date - REFERENCE_DATE).days
            return days_since_ref + days_1900_to_ref
        except:
            return time_data / 24.0 + DAYS_1900_TO_1970
    
    elif 'days since' in time_units.lower():
        try:
            ref_str = time_units.lower().replace('days since', '').strip()
            ref_date = datetime.strptime(ref_str.split()[0], '%Y-%m-%d')
            days_1900_to_ref = (ref_date - REFERENCE_DATE).days
            return time_data + days_1900_to_ref
        except:
            return time_data
    
    else:
        if max_val > 1e6:
            return time_data / 86400.0 + DAYS_1900_TO_1970
        return time_data


def get_grid_coords(grid_file):
    with nc.Dataset(grid_file, 'r') as grd:
        return {
            'lon_rho': grd.variables['lon_rho'][:],
            'lat_rho': grd.variables['lat_rho'][:],
            'mask_rho': grd.variables['mask_rho'][:],
            'angle': grd.variables['angle'][:],
            'eta_rho': grd.variables['lon_rho'].shape[0],
            'xi_rho': grd.variables['lon_rho'].shape[1],
        }


def get_time_info(nc_file, time_var='valid_time'):
    with nc.Dataset(nc_file, 'r') as ds:
        time_data = ds.variables[time_var][:]
        time_units = ds.variables[time_var].units
        calendar = getattr(ds.variables[time_var], 'calendar', 'gregorian')
        dates = nc.num2date(time_data, time_units, calendar=calendar)
        
        monthly_indices = {}
        for i, dt in enumerate(dates):
            key = (dt.year, dt.month) if hasattr(dt, 'year') else \
                  (dt.timetuple().tm_year, dt.timetuple().tm_mon)
            if key not in monthly_indices:
                monthly_indices[key] = []
            monthly_indices[key].append(i)
        
        return {
            'time_data': time_data,
            'time_units': time_units,
            'calendar': calendar,
            'monthly_indices': monthly_indices,
            'n_times': len(time_data)
        }


class FastInterpolator:
    """Pre-computed interpolation for speed."""
    
    def __init__(self, src_lon, src_lat, dst_lon, dst_lat):
        self.src_lat = np.asarray(src_lat).flatten()
        self.src_lon = np.asarray(src_lon).flatten()
        self.dst_shape = dst_lon.shape
        
        self.lat_flip = self.src_lat[0] > self.src_lat[-1]
        if self.lat_flip:
            self.src_lat = self.src_lat[::-1]
        
        dst_lon_adj = np.where(dst_lon > 180, dst_lon - 360, dst_lon)
        src_lon_adj = np.where(self.src_lon > 180, self.src_lon - 360, self.src_lon)
        
        self.lon_sort = np.argsort(src_lon_adj)
        self.src_lon_sorted = src_lon_adj[self.lon_sort]
        self.points = np.column_stack([dst_lat.ravel(), dst_lon_adj.ravel()])
    
    def __call__(self, src_data, fill_value=np.nan):
        if self.lat_flip:
            src_data = src_data[::-1, :]
        src_data = src_data[:, self.lon_sort]
        
        interp = RegularGridInterpolator(
            (self.src_lat, self.src_lon_sorted), src_data,
            method='linear', bounds_error=False, fill_value=fill_value
        )
        return interp(self.points).reshape(self.dst_shape).astype(np.float32)


def rotate_vectors(u, v, angle):
    """Rotate vectors from geographic to ROMS grid coordinates."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return (u * cos_a + v * sin_a).astype(np.float32), (-u * sin_a + v * cos_a).astype(np.float32)


def calc_humidity(t2m_K, d2m_K):
    """Calculate relative humidity from temperature and dewpoint."""
    T = t2m_K - KELVIN_OFFSET
    Td = d2m_K - KELVIN_OFFSET
    e_T = 6.112 * np.exp(17.67 * T / (T + 243.5))
    e_Td = 6.112 * np.exp(17.67 * Td / (Td + 243.5))
    return np.clip(e_Td / e_T, 0, 1).astype(np.float32)


def calc_wavelength(period_seconds):
    """Calculate deep-water wavelength from period: L = g*T²/(2π)"""
    return (G * period_seconds**2) / (2.0 * np.pi)


def expand_daily_to_hourly(daily_data):
    """Expand daily data to hourly by repeating values."""
    n_days = daily_data.shape[0]
    hourly = np.zeros((n_days * 24,) + daily_data.shape[1:], dtype=daily_data.dtype)
    for d in range(n_days):
        hourly[d * 24:(d + 1) * 24] = daily_data[d]
    return hourly


def mask_and_fill(data, mask, fill_value=FILL_VALUE):
    """Apply land mask and replace NaN with fill value."""
    land = mask == 0
    data[:, land] = fill_value
    return np.where(np.isnan(data), fill_value, data)


# =============================================================================
# Worker Functions - Process Monthly Data
# =============================================================================

def process_bulk_month(args):
    """Process one month of bulk forcing including all variables."""
    month_key, time_indices, grid_file, era5_dir, temp_dir, progress_file = args
    year, month = month_key
    pid = current_process().name
    t0 = time.time()
    nt = len(time_indices)
    
    def msg(s):
        line = f"{datetime.now().strftime('%H:%M:%S')} | [{pid}] BULK {year}-{month:02d}: {s}"
        print(line, flush=True)
        try:
            with open(progress_file, 'a') as f:
                f.write(line + '\n')
                f.flush()
        except:
            pass
    
    msg(f"START ({nt} steps)")
    
    grid = get_grid_coords(grid_file)
    lon_rho, lat_rho = grid['lon_rho'], grid['lat_rho']
    angle, mask = grid['angle'], grid['mask_rho']
    eta, xi = lon_rho.shape
    
    # Pre-allocate all variables
    Pair = np.zeros((nt, eta, xi), dtype=np.float32)
    Tair = np.zeros((nt, eta, xi), dtype=np.float32)
    Qair = np.zeros((nt, eta, xi), dtype=np.float32)
    swrad = np.zeros((nt, eta, xi), dtype=np.float32)      # Net shortwave
    swrad_down = np.zeros((nt, eta, xi), dtype=np.float32) # Downwelling shortwave
    lwrad = np.zeros((nt, eta, xi), dtype=np.float32)      # Net longwave
    lwrad_down = np.zeros((nt, eta, xi), dtype=np.float32) # Downwelling longwave
    Uwind = np.zeros((nt, eta, xi), dtype=np.float32)
    Vwind = np.zeros((nt, eta, xi), dtype=np.float32)
    cloud = np.zeros((nt, eta, xi), dtype=np.float32)
    rain = np.zeros((nt, eta, xi), dtype=np.float32)
    time_out = None
    
    # Atmosphere (t2m, d2m, msl)
    atm_file = os.path.join(era5_dir, ERA5_FILES['atmosphere'])
    if os.path.exists(atm_file):
        msg("Atmosphere...")
        with nc.Dataset(atm_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            time_out = ds['valid_time'][time_indices]
            for i, ti in enumerate(time_indices):
                if i % 50 == 0:
                    msg(f"  Atm {i}/{nt}")
                Pair[i] = interp(ds['msl'][ti]) * PA_TO_MBAR
                t2m = interp(ds['t2m'][ti])
                Tair[i] = t2m - KELVIN_OFFSET
                d2m = interp(ds['d2m'][ti])
                Qair[i] = calc_humidity(t2m, d2m)
    
    # Downwelling Radiation
    rad_down_file = os.path.join(era5_dir, ERA5_FILES['radiation_down'])
    if os.path.exists(rad_down_file):
        msg("Downwelling radiation...")
        with nc.Dataset(rad_down_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            for i, ti in enumerate(time_indices):
                if i % 50 == 0:
                    msg(f"  Rad down {i}/{nt}")
                swrad_down[i] = interp(ds['avg_sdswrf'][ti])
                lwrad_down[i] = interp(ds['avg_sdlwrf'][ti])
    
    # Net Radiation (if available)
    rad_net_file = os.path.join(era5_dir, ERA5_FILES.get('radiation_net', 'snrf_avg.nc'))
    if os.path.exists(rad_net_file):
        msg("Net radiation...")
        with nc.Dataset(rad_net_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            # Check which variables are available
            has_net_sw = 'avg_snswrf' in ds.variables
            has_net_lw = 'avg_snlwrf' in ds.variables
            for i, ti in enumerate(time_indices):
                if i % 50 == 0:
                    msg(f"  Rad net {i}/{nt}")
                if has_net_sw:
                    swrad[i] = interp(ds['avg_snswrf'][ti])
                if has_net_lw:
                    lwrad[i] = interp(ds['avg_snlwrf'][ti])
    else:
        # If no net radiation file, copy downwelling (user can modify)
        msg("  No net radiation file - using downwelling values")
        swrad = swrad_down.copy()
        lwrad = lwrad_down.copy()
    
    # Winds
    wind_file = os.path.join(era5_dir, ERA5_FILES['winds'])
    if os.path.exists(wind_file):
        msg("Winds...")
        with nc.Dataset(wind_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            for i, ti in enumerate(time_indices):
                if i % 50 == 0:
                    msg(f"  Wind {i}/{nt}")
                u = interp(ds['u10'][ti])
                v = interp(ds['v10'][ti])
                Uwind[i], Vwind[i] = rotate_vectors(u, v, angle)
    
    # Clouds
    cloud_file = os.path.join(era5_dir, ERA5_FILES['clouds'])
    if os.path.exists(cloud_file):
        msg("Clouds...")
        with nc.Dataset(cloud_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            for i, ti in enumerate(time_indices):
                if i % 50 == 0:
                    msg(f"  Cloud {i}/{nt}")
                cloud[i] = interp(ds['tcc'][ti])
    
    # Precipitation - rain_new.nc has same timesteps as atmosphere
    precip_file = os.path.join(era5_dir, 'rain_new.nc')
    if os.path.exists(precip_file):
        msg("Precipitation...")
        with nc.Dataset(precip_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            
            # Get variable name (tp = total precipitation)
            precip_var = 'tp' if 'tp' in ds.variables else 'mtpr' if 'mtpr' in ds.variables else None
            
            if precip_var:
                for i, ti in enumerate(time_indices):
                    if i % 50 == 0:
                        msg(f"  Precip {i}/{nt}")
                    tp = interp(np.ma.filled(ds[precip_var][ti], 0.0))
                    # Convert from m (accumulated over 1 hour) to kg/m²/s
                    # 1 m water = 1000 kg/m², 1 hour = 3600 s
                    # Rate = m/hour * 1000 kg/m³ / 3600 s = kg/m²/s
                    rain[i] = tp * 1000.0 / 3600.0
            else:
                msg("  No precip variable found (tp or mtpr)")
    else:
        msg("  No precipitation file found (rain_new.nc)")
    
    # Apply mask
    msg("Masking...")
    land = mask == 0
    for arr in [Pair, Tair, Qair, swrad, swrad_down, lwrad, lwrad_down, Uwind, Vwind, cloud, rain]:
        arr[:, land] = np.nan
    
    # Save as uncompressed npy (fast I/O)
    msg("Saving (uncompressed)...")
    prefix = os.path.join(temp_dir, f'bulk_{year}_{month:02d}')
    
    # Convert time to ROMS format
    if time_out is not None:
        time_out = np.asarray(time_out).astype(np.float64)
        with nc.Dataset(atm_file, 'r') as ds:
            src_units = ds.variables['valid_time'].units
        time_out = convert_time_to_roms(time_out, src_units)
    else:
        time_out = np.arange(nt, dtype=np.float64)
    
    np.save(f'{prefix}_time.npy', time_out)
    np.save(f'{prefix}_Pair.npy', Pair)
    np.save(f'{prefix}_Tair.npy', Tair)
    np.save(f'{prefix}_Qair.npy', Qair)
    np.save(f'{prefix}_swrad.npy', swrad)
    np.save(f'{prefix}_swrad_down.npy', swrad_down)
    np.save(f'{prefix}_lwrad.npy', lwrad)
    np.save(f'{prefix}_lwrad_down.npy', lwrad_down)
    np.save(f'{prefix}_Uwind.npy', Uwind)
    np.save(f'{prefix}_Vwind.npy', Vwind)
    np.save(f'{prefix}_cloud.npy', cloud)
    np.save(f'{prefix}_rain.npy', rain)
    
    elapsed = time.time() - t0
    msg(f"DONE in {format_time(elapsed)}")
    
    del Pair, Tair, Qair, swrad, swrad_down, lwrad, lwrad_down, Uwind, Vwind, cloud, rain
    gc.collect()
    return month_key, prefix


def process_wave_month(args):
    """Process one month of wave forcing including all variables."""
    month_key, time_indices, grid_file, era5_dir, temp_dir, progress_file = args
    year, month = month_key
    pid = current_process().name
    
    wave_file = os.path.join(era5_dir, ERA5_FILES['waves'])
    if not os.path.exists(wave_file):
        return month_key, None
    
    t0 = time.time()
    nt = len(time_indices)
    
    def msg(s):
        line = f"{datetime.now().strftime('%H:%M:%S')} | [{pid}] WAVE {year}-{month:02d}: {s}"
        print(line, flush=True)
        try:
            with open(progress_file, 'a') as f:
                f.write(line + '\n')
                f.flush()
        except:
            pass
    
    msg(f"START ({nt} steps)")
    
    grid = get_grid_coords(grid_file)
    lon_rho, lat_rho = grid['lon_rho'], grid['lat_rho']
    angle, mask = grid['angle'], grid['mask_rho']
    eta, xi = lon_rho.shape
    
    # Allocate arrays
    Hwave = np.zeros((nt, eta, xi), dtype=np.float32)  # Significant wave height
    Uwave = np.zeros((nt, eta, xi), dtype=np.float32)  # Stokes drift u
    Vwave = np.zeros((nt, eta, xi), dtype=np.float32)  # Stokes drift v
    Dwave = np.zeros((nt, eta, xi), dtype=np.float32)  # Wave direction
    Pwave = np.zeros((nt, eta, xi), dtype=np.float32)  # Peak wave period
    Lwave = np.zeros((nt, eta, xi), dtype=np.float32)  # Wavelength
    
    # Primary wave file (swh, ust, vst)
    with nc.Dataset(wave_file, 'r') as ds:
        interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
        time_out = ds['valid_time'][time_indices]
        for i, ti in enumerate(time_indices):
            if i % 50 == 0:
                msg(f"Step {i}/{nt}")
            Hwave[i] = interp(ds['swh'][ti])
            u = interp(ds['ust'][ti])
            v = interp(ds['vst'][ti])
            Uwave[i], Vwave[i] = rotate_vectors(u, v, angle)
    
    # Additional wave file (mwd, pp1d) for direction and period
    wave_add_file = os.path.join(era5_dir, ERA5_FILES.get('waves_add', 'wave_add.nc'))
    if os.path.exists(wave_add_file):
        msg("Loading wave direction and period...")
        with nc.Dataset(wave_add_file, 'r') as ds:
            interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
            for i, ti in enumerate(time_indices):
                if i % 50 == 0:
                    msg(f"  Wave add {i}/{nt}")
                Dwave[i] = interp(ds['mwd'][ti])  # Mean wave direction
                Pwave[i] = interp(ds['pp1d'][ti])  # Peak period
        
        # Calculate wavelength from period
        Lwave = calc_wavelength(Pwave)
    else:
        msg("  No wave_add file - Dwave, Pwave, Lwave will be zeros")
    
    # Apply mask
    land = mask == 0
    for arr in [Hwave, Uwave, Vwave, Dwave, Pwave, Lwave]:
        arr[:, land] = np.nan
    
    msg("Saving...")
    prefix = os.path.join(temp_dir, f'wave_{year}_{month:02d}')
    
    # Convert time to ROMS format
    time_out = np.asarray(time_out).astype(np.float64)
    with nc.Dataset(wave_file, 'r') as ds:
        src_units = ds.variables['valid_time'].units
    time_out = convert_time_to_roms(time_out, src_units)
    
    np.save(f'{prefix}_time.npy', time_out)
    np.save(f'{prefix}_Hwave.npy', Hwave)
    np.save(f'{prefix}_Uwave.npy', Uwave)
    np.save(f'{prefix}_Vwave.npy', Vwave)
    np.save(f'{prefix}_Dwave.npy', Dwave)
    np.save(f'{prefix}_Pwave.npy', Pwave)
    np.save(f'{prefix}_Lwave.npy', Lwave)
    
    msg(f"DONE in {format_time(time.time() - t0)}")
    del Hwave, Uwave, Vwave, Dwave, Pwave, Lwave
    gc.collect()
    return month_key, prefix


def process_ocean_month(args):
    """Process one month of ocean forcing (CMEMS daily -> hourly)."""
    month_key, daily_indices, hourly_time_info, grid_file, cmems_dir, temp_dir, progress_file = args
    year, month = month_key
    pid = current_process().name
    
    ocean_file = os.path.join(cmems_dir, CMEMS_FILES['ocean'])
    if not os.path.exists(ocean_file):
        return month_key, None
    
    t0 = time.time()
    nd = len(daily_indices)
    nh = nd * 24
    
    def msg(s):
        line = f"{datetime.now().strftime('%H:%M:%S')} | [{pid}] OCEAN {year}-{month:02d}: {s}"
        print(line, flush=True)
        try:
            with open(progress_file, 'a') as f:
                f.write(line + '\n')
                f.flush()
        except:
            pass
    
    msg(f"START ({nd} days -> {nh} hours)")
    
    grid = get_grid_coords(grid_file)
    lon_rho, lat_rho = grid['lon_rho'], grid['lat_rho']
    mask = grid['mask_rho']
    eta, xi = lon_rho.shape
    
    SST_d = np.zeros((nd, eta, xi), dtype=np.float32)
    SSS_d = np.zeros((nd, eta, xi), dtype=np.float32)
    zeta_d = np.zeros((nd, eta, xi), dtype=np.float32)
    
    with nc.Dataset(ocean_file, 'r') as ds:
        interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
        for i, di in enumerate(daily_indices):
            if i % 5 == 0:
                msg(f"Day {i+1}/{nd}")
            
            # Read and mask fill values
            sst_raw = np.where(ds['thetao'][di, 0] == -32767, np.nan, ds['thetao'][di, 0])
            sss_raw = np.where(ds['so'][di, 0] == -32767, np.nan, ds['so'][di, 0])
            zeta_raw = np.where(ds['zos'][di] == -32767, np.nan, ds['zos'][di])
            
            SST_d[i] = interp(sst_raw)
            SSS_d[i] = interp(sss_raw)
            zeta_d[i] = interp(zeta_raw)
    
    msg("Expanding to hourly...")
    SST = expand_daily_to_hourly(SST_d)
    SSS = expand_daily_to_hourly(SSS_d)
    zeta = expand_daily_to_hourly(zeta_d)
    del SST_d, SSS_d, zeta_d
    
    # Time - generate hourly from CMEMS daily
    with nc.Dataset(ocean_file, 'r') as ds:
        cmems_time = ds.variables['time'][daily_indices]
        cmems_units = ds.variables['time'].units
    
    daily_time_roms = convert_time_to_roms(cmems_time, cmems_units)
    time_out = np.zeros(nh, dtype=np.float64)
    for d in range(nd):
        for h in range(24):
            time_out[d * 24 + h] = daily_time_roms[d] + h / 24.0
    
    # Apply mask
    land = mask == 0
    SST[:, land] = np.nan
    SSS[:, land] = np.nan
    zeta[:, land] = np.nan
    
    msg("Saving...")
    prefix = os.path.join(temp_dir, f'ocean_{year}_{month:02d}')
    np.save(f'{prefix}_time.npy', time_out)
    np.save(f'{prefix}_SST.npy', SST)
    np.save(f'{prefix}_SSS.npy', SSS)
    np.save(f'{prefix}_zeta.npy', zeta)
    
    msg(f"DONE in {format_time(time.time() - t0)}")
    del SST, SSS, zeta
    gc.collect()
    return month_key, prefix


def process_current_month(args):
    """Process one month of current forcing."""
    month_key, daily_indices, hourly_time_info, grid_file, cmems_dir, temp_dir, progress_file = args
    year, month = month_key
    pid = current_process().name
    
    curr_file = os.path.join(cmems_dir, CMEMS_FILES['currents'])
    if not os.path.exists(curr_file):
        return month_key, None
    
    t0 = time.time()
    nd = len(daily_indices)
    nh = nd * 24
    
    def msg(s):
        line = f"{datetime.now().strftime('%H:%M:%S')} | [{pid}] CURR {year}-{month:02d}: {s}"
        print(line, flush=True)
        try:
            with open(progress_file, 'a') as f:
                f.write(line + '\n')
                f.flush()
        except:
            pass
    
    msg(f"START ({nd} days)")
    
    grid = get_grid_coords(grid_file)
    lon_rho, lat_rho = grid['lon_rho'], grid['lat_rho']
    angle, mask = grid['angle'], grid['mask_rho']
    eta, xi = lon_rho.shape
    
    ubar_d = np.zeros((nd, eta, xi), dtype=np.float32)
    vbar_d = np.zeros((nd, eta, xi), dtype=np.float32)
    
    with nc.Dataset(curr_file, 'r') as ds:
        interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
        for i, di in enumerate(daily_indices):
            if i % 5 == 0:
                msg(f"Day {i+1}/{nd}")
            
            u_raw = np.where(ds['uo'][di, 0] == -32767, np.nan, ds['uo'][di, 0])
            v_raw = np.where(ds['vo'][di, 0] == -32767, np.nan, ds['vo'][di, 0])
            u_raw = np.where(np.abs(u_raw) > 15, np.nan, u_raw)
            v_raw = np.where(np.abs(v_raw) > 15, np.nan, v_raw)
            
            u = interp(u_raw)
            v = interp(v_raw)
            ubar_d[i], vbar_d[i] = rotate_vectors(u, v, angle)
    
    msg("Expanding...")
    ubar = expand_daily_to_hourly(ubar_d)
    vbar = expand_daily_to_hourly(vbar_d)
    del ubar_d, vbar_d
    
    # Time
    with nc.Dataset(curr_file, 'r') as ds:
        cmems_time = ds.variables['time'][daily_indices]
        cmems_units = ds.variables['time'].units
    
    daily_time_roms = convert_time_to_roms(cmems_time, cmems_units)
    time_out = np.zeros(nh, dtype=np.float64)
    for d in range(nd):
        for h in range(24):
            time_out[d * 24 + h] = daily_time_roms[d] + h / 24.0
    
    land = mask == 0
    ubar[:, land] = np.nan
    vbar[:, land] = np.nan
    
    msg("Saving...")
    prefix = os.path.join(temp_dir, f'curr_{year}_{month:02d}')
    np.save(f'{prefix}_time.npy', time_out)
    np.save(f'{prefix}_ubar.npy', ubar)
    np.save(f'{prefix}_vbar.npy', vbar)
    
    msg(f"DONE in {format_time(time.time() - t0)}")
    del ubar, vbar
    gc.collect()
    return month_key, prefix


def process_ice_month(args):
    """Process one month of ice forcing."""
    month_key, daily_indices, hourly_time_info, grid_file, cmems_dir, temp_dir, progress_file = args
    year, month = month_key
    pid = current_process().name
    
    ice_file = os.path.join(cmems_dir, CMEMS_FILES['ice'])
    if not os.path.exists(ice_file):
        return month_key, None
    
    t0 = time.time()
    nd = len(daily_indices)
    nh = nd * 24
    
    def msg(s):
        line = f"{datetime.now().strftime('%H:%M:%S')} | [{pid}] ICE {year}-{month:02d}: {s}"
        print(line, flush=True)
        try:
            with open(progress_file, 'a') as f:
                f.write(line + '\n')
                f.flush()
        except:
            pass
    
    msg(f"START ({nd} days)")
    
    grid = get_grid_coords(grid_file)
    lon_rho, lat_rho = grid['lon_rho'], grid['lat_rho']
    angle, mask = grid['angle'], grid['mask_rho']
    eta, xi = lon_rho.shape
    
    aice_d = np.zeros((nd, eta, xi), dtype=np.float32)
    hice_d = np.zeros((nd, eta, xi), dtype=np.float32)
    uice_d = np.zeros((nd, eta, xi), dtype=np.float32)
    vice_d = np.zeros((nd, eta, xi), dtype=np.float32)
    
    with nc.Dataset(ice_file, 'r') as ds:
        interp = FastInterpolator(ds['longitude'][:], ds['latitude'][:], lon_rho, lat_rho)
        for i, di in enumerate(daily_indices):
            if i % 5 == 0:
                msg(f"Day {i+1}/{nd}")
            
            aice_raw = np.where(ds['siconc'][di] == -32767, np.nan, ds['siconc'][di])
            hice_raw = np.where(ds['sithick'][di] == -32767, np.nan, ds['sithick'][di])
            u_raw = np.where(ds['usi'][di] == -32767, np.nan, ds['usi'][di])
            v_raw = np.where(ds['vsi'][di] == -32767, np.nan, ds['vsi'][di])
            
            aice_d[i] = interp(aice_raw)
            hice_d[i] = interp(hice_raw)
            u = interp(u_raw)
            v = interp(v_raw)
            uice_d[i], vice_d[i] = rotate_vectors(u, v, angle)
    
    msg("Expanding...")
    aice = expand_daily_to_hourly(aice_d)
    hice = expand_daily_to_hourly(hice_d)
    uice = expand_daily_to_hourly(uice_d)
    vice = expand_daily_to_hourly(vice_d)
    del aice_d, hice_d, uice_d, vice_d
    
    # Time
    with nc.Dataset(ice_file, 'r') as ds:
        cmems_time = ds.variables['time'][daily_indices]
        cmems_units = ds.variables['time'].units
    
    daily_time_roms = convert_time_to_roms(cmems_time, cmems_units)
    time_out = np.zeros(nh, dtype=np.float64)
    for d in range(nd):
        for h in range(24):
            time_out[d * 24 + h] = daily_time_roms[d] + h / 24.0
    
    land = mask == 0
    aice[:, land] = np.nan
    hice[:, land] = np.nan
    uice[:, land] = np.nan
    vice[:, land] = np.nan
    
    msg("Saving...")
    prefix = os.path.join(temp_dir, f'ice_{year}_{month:02d}')
    np.save(f'{prefix}_time.npy', time_out)
    np.save(f'{prefix}_aice.npy', aice)
    np.save(f'{prefix}_hice.npy', hice)
    np.save(f'{prefix}_uice.npy', uice)
    np.save(f'{prefix}_vice.npy', vice)
    
    msg(f"DONE in {format_time(time.time() - t0)}")
    del aice, hice, uice, vice
    gc.collect()
    return month_key, prefix


# =============================================================================
# NetCDF Assembly Functions
# =============================================================================

def assemble_bulk(temp_files, output_file, grid_coords):
    """Assemble bulk forcing with proper time variables and fill values."""
    log(f"Assembling {os.path.basename(output_file)}...")
    t0 = time.time()
    
    sorted_months = sorted([k for k, v in temp_files.items() if v])
    if not sorted_months:
        log("  No data!")
        return
    
    # Count total timesteps
    total_nt = sum(len(np.load(f'{temp_files[mk]}_time.npy')) for mk in sorted_months)
    log(f"  Total timesteps: {total_nt}")
    eta, xi = grid_coords['eta_rho'], grid_coords['xi_rho']
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as out:
        # Dimensions
        out.createDimension('xi_rho', xi)
        out.createDimension('eta_rho', eta)
        out.createDimension('time', None)
        
        # Grid coordinates
        lon = out.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))
        lon[:] = grid_coords['lon_rho']
        lon.long_name = 'longitude of rho-points'
        lon.units = 'degree_east'
        
        lat = out.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))
        lat[:] = grid_coords['lat_rho']
        lat.long_name = 'latitude of rho-points'
        lat.units = 'degree_north'
        
        # Time variables (ROMS convention)
        time_vars = {
            'Pair_time': 'air pressure time',
            'Tair_time': 'air temperature time',
            'Qair_time': 'air humidity time',
            'wind_time': 'wind time',
            'swrad_time': 'shortwave radiation time',
            'lwrad_time': 'longwave radiation time',
            'rain_time': 'precipitation time',
            'cloud_time': 'cloud fraction time',
        }
        
        for tname, lname in time_vars.items():
            tvar = out.createVariable(tname, 'f8', ('time',))
            tvar.units = TIME_UNITS
            tvar.calendar = CALENDAR
            tvar.long_name = lname
        
        # Data variables with metadata
        vars_info = {
            'Pair': {'long_name': 'surface air pressure', 'units': 'millibar', 'time': 'Pair_time'},
            'Tair': {'long_name': 'surface air temperature', 'units': 'Celsius', 'time': 'Tair_time'},
            'Qair': {'long_name': 'surface air relative humidity', 'units': 'fraction', 'time': 'Qair_time'},
            'Uwind': {'long_name': 'surface u-wind component', 'units': 'meter second-1', 'time': 'wind_time'},
            'Vwind': {'long_name': 'surface v-wind component', 'units': 'meter second-1', 'time': 'wind_time'},
            'swrad': {'long_name': 'net shortwave radiation flux', 'units': 'watt meter-2', 'time': 'swrad_time'},
            'swrad_down': {'long_name': 'downwelling shortwave radiation flux', 'units': 'watt meter-2', 'time': 'swrad_time'},
            'lwrad': {'long_name': 'net longwave radiation flux', 'units': 'watt meter-2', 'time': 'lwrad_time'},
            'lwrad_down': {'long_name': 'downwelling longwave radiation flux', 'units': 'watt meter-2', 'time': 'lwrad_time'},
            'rain': {'long_name': 'precipitation rate', 'units': 'kilogram meter-2 second-1', 'time': 'rain_time'},
            'cloud': {'long_name': 'cloud fraction', 'units': 'nondimensional', 'time': 'cloud_time'},
        }
        
        for vname, vinfo in vars_info.items():
            v = out.createVariable(vname, 'f4', ('time', 'eta_rho', 'xi_rho'),
                                   fill_value=FILL_VALUE, zlib=True, complevel=4)
            v.long_name = vinfo['long_name']
            v.units = vinfo['units']
            v.time = vinfo['time']
            v.coordinates = 'lon_rho lat_rho'
        
        out.title = 'ROMS Bulk Flux Forcing'
        out.Conventions = 'CF-1.6'
        out.history = f'Created {datetime.now()}'
        
        # Write data month by month
        tidx = 0
        for i, mk in enumerate(sorted_months):
            year, month = mk
            log(f"  Writing {year}-{month:02d} ({i+1}/{len(sorted_months)})")
            prefix = temp_files[mk]
            
            t = np.load(f'{prefix}_time.npy')
            nt = len(t)
            
            # Write time to all time variables
            for tname in time_vars.keys():
                out.variables[tname][tidx:tidx+nt] = t
            
            # Map file names to variable names
            file_to_var = {
                'Pair': 'Pair', 'Tair': 'Tair', 'Qair': 'Qair',
                'swrad': 'swrad', 'swrad_down': 'swrad_down',
                'lwrad': 'lwrad', 'lwrad_down': 'lwrad_down',
                'Uwind': 'Uwind', 'Vwind': 'Vwind',
                'cloud': 'cloud', 'rain': 'rain'
            }
            
            for fname, vname in file_to_var.items():
                fpath = f'{prefix}_{fname}.npy'
                if os.path.exists(fpath):
                    data = np.load(fpath)
                    data = np.where(np.isnan(data), FILL_VALUE, data)
                    out.variables[vname][tidx:tidx+nt] = data
                    del data
            
            tidx += nt
            out.sync()
            gc.collect()
    
    log(f"  DONE: {format_size(os.path.getsize(output_file))} in {format_time(time.time() - t0)}")


def assemble_wave(temp_files, output_file, grid_coords):
    """Assemble wave forcing with all variables."""
    log(f"Assembling {os.path.basename(output_file)}...")
    t0 = time.time()
    
    sorted_months = sorted([k for k, v in temp_files.items() if v])
    if not sorted_months:
        log("  No data!")
        return
    
    total_nt = sum(len(np.load(f'{temp_files[mk]}_time.npy')) for mk in sorted_months)
    log(f"  Total timesteps: {total_nt}")
    eta, xi = grid_coords['eta_rho'], grid_coords['xi_rho']
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as out:
        out.createDimension('xi_rho', xi)
        out.createDimension('eta_rho', eta)
        out.createDimension('wave_time', None)  # Use wave_time as dimension
        
        lon = out.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))
        lon[:] = grid_coords['lon_rho']
        lon.long_name = 'longitude of rho-points'
        lon.units = 'degree_east'
        
        lat = out.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))
        lat[:] = grid_coords['lat_rho']
        lat.long_name = 'latitude of rho-points'
        lat.units = 'degree_north'
        
        tvar = out.createVariable('wave_time', 'f8', ('wave_time',))
        tvar.units = TIME_UNITS
        tvar.calendar = CALENDAR
        tvar.long_name = 'wave forcing time'
        
        # Wave variables
        vars_info = {
            'Hwave': {'long_name': 'significant wave height', 'units': 'meter'},
            'Dwave': {'long_name': 'mean wave direction', 'units': 'degrees from North (clockwise)'},
            'Pwave': {'long_name': 'peak wave period', 'units': 'second'},
            'Lwave': {'long_name': 'mean wavelength', 'units': 'meter'},
            'Uwave': {'long_name': 'surface Stokes drift u-component', 'units': 'meter second-1'},
            'Vwave': {'long_name': 'surface Stokes drift v-component', 'units': 'meter second-1'},
        }
        
        for vname, vinfo in vars_info.items():
            v = out.createVariable(vname, 'f4', ('wave_time', 'eta_rho', 'xi_rho'),
                                   fill_value=FILL_VALUE, zlib=True, complevel=4)
            v.long_name = vinfo['long_name']
            v.units = vinfo['units']
            v.time = 'wave_time'
        
        out.title = 'ROMS Wave Forcing'
        out.Conventions = 'CF-1.6'
        out.history = f'Created {datetime.now()}'
        
        tidx = 0
        for i, mk in enumerate(sorted_months):
            year, month = mk
            if (i + 1) % 12 == 0 or i == 0:
                log(f"  Writing {year}-{month:02d} ({i+1}/{len(sorted_months)})")
            prefix = temp_files[mk]
            
            t = np.load(f'{prefix}_time.npy')
            nt = len(t)
            tvar[tidx:tidx+nt] = t
            
            for vname in vars_info.keys():
                fpath = f'{prefix}_{vname}.npy'
                if os.path.exists(fpath):
                    data = np.load(fpath)
                    out.variables[vname][tidx:tidx+nt] = np.where(np.isnan(data), FILL_VALUE, data)
                    del data
            
            tidx += nt
            out.sync()
            gc.collect()
    
    log(f"  DONE: {format_size(os.path.getsize(output_file))} in {format_time(time.time() - t0)}")


def assemble_ocean(temp_files, output_file, grid_coords):
    """Assemble ocean forcing with sst_time dimension and separate time variables."""
    log(f"Assembling {os.path.basename(output_file)}...")
    t0 = time.time()
    
    sorted_months = sorted([k for k, v in temp_files.items() if v])
    if not sorted_months:
        return
    
    total_nt = sum(len(np.load(f'{temp_files[mk]}_time.npy')) for mk in sorted_months)
    eta, xi = grid_coords['eta_rho'], grid_coords['xi_rho']
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as out:
        out.createDimension('xi_rho', xi)
        out.createDimension('eta_rho', eta)
        out.createDimension('sst_time', None)  # Use sst_time as dimension
        
        lon = out.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))
        lon[:] = grid_coords['lon_rho']
        lat = out.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))
        lat[:] = grid_coords['lat_rho']
        
        # Create all time variables
        sst_time = out.createVariable('sst_time', 'f8', ('sst_time',))
        sst_time.units = TIME_UNITS
        sst_time.calendar = CALENDAR
        sst_time.long_name = 'sst time'
        
        sss_time = out.createVariable('sss_time', 'f8', ('sst_time',))
        sss_time.units = TIME_UNITS
        sss_time.calendar = CALENDAR
        sss_time.long_name = 'sss time'
        
        ssh_time = out.createVariable('ssh_time', 'f8', ('sst_time',))
        ssh_time.units = TIME_UNITS
        ssh_time.calendar = CALENDAR
        ssh_time.long_name = 'ssh time'
        
        # Data variables with their specific time references
        vars_info = {
            'SST': {'long_name': 'sea surface temperature', 'units': 'Celsius', 'time': 'sst_time'},
            'SSS': {'long_name': 'sea surface salinity', 'units': 'PSU', 'time': 'sss_time'},
            'zeta': {'long_name': 'sea surface height', 'units': 'meter', 'time': 'ssh_time'},
        }
        
        for vname, vinfo in vars_info.items():
            v = out.createVariable(vname, 'f4', ('sst_time', 'eta_rho', 'xi_rho'),
                                   fill_value=FILL_VALUE, zlib=True, complevel=4)
            v.long_name = vinfo['long_name']
            v.units = vinfo['units']
            v.time = vinfo['time']
        
        out.title = 'ROMS Ocean Forcing'
        out.history = f'Created {datetime.now()}'
        
        tidx = 0
        for i, mk in enumerate(sorted_months):
            if i % 10 == 0:
                log(f"  Month {i+1}/{len(sorted_months)}")
            prefix = temp_files[mk]
            t = np.load(f'{prefix}_time.npy')
            nt = len(t)
            
            # Write to all time variables
            sst_time[tidx:tidx+nt] = t
            sss_time[tidx:tidx+nt] = t
            ssh_time[tidx:tidx+nt] = t
            
            for vname in ['SST', 'SSS', 'zeta']:
                data = np.load(f'{prefix}_{vname}.npy')
                out.variables[vname][tidx:tidx+nt] = np.where(np.isnan(data), FILL_VALUE, data)
                del data
            
            tidx += nt
            out.sync()
            gc.collect()
    
    log(f"  DONE: {format_size(os.path.getsize(output_file))} in {format_time(time.time() - t0)}")


def assemble_current(temp_files, output_file, grid_coords):
    """Assemble current forcing with ocean_time dimension and separate time variables."""
    log(f"Assembling {os.path.basename(output_file)}...")
    t0 = time.time()
    
    sorted_months = sorted([k for k, v in temp_files.items() if v])
    if not sorted_months:
        return
    
    eta, xi = grid_coords['eta_rho'], grid_coords['xi_rho']
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as out:
        out.createDimension('xi_rho', xi)
        out.createDimension('eta_rho', eta)
        out.createDimension('ocean_time', None)  # Use ocean_time as dimension
        
        out.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))[:] = grid_coords['lon_rho']
        out.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))[:] = grid_coords['lat_rho']
        
        # Create all time variables
        ocean_time = out.createVariable('ocean_time', 'f8', ('ocean_time',))
        ocean_time.units = TIME_UNITS
        ocean_time.calendar = CALENDAR
        ocean_time.long_name = 'ocean time'
        
        ubar_time = out.createVariable('ubar_time', 'f8', ('ocean_time',))
        ubar_time.units = TIME_UNITS
        ubar_time.calendar = CALENDAR
        ubar_time.long_name = 'ubar time'
        
        vbar_time = out.createVariable('vbar_time', 'f8', ('ocean_time',))
        vbar_time.units = TIME_UNITS
        vbar_time.calendar = CALENDAR
        vbar_time.long_name = 'vbar time'
        
        # Data variables with their specific time references
        ubar = out.createVariable('ubar', 'f4', ('ocean_time', 'eta_rho', 'xi_rho'),
                                  fill_value=FILL_VALUE, zlib=True, complevel=4)
        ubar.long_name = 'barotropic u-velocity'
        ubar.units = 'meter second-1'
        ubar.time = 'ubar_time'
        
        vbar = out.createVariable('vbar', 'f4', ('ocean_time', 'eta_rho', 'xi_rho'),
                                  fill_value=FILL_VALUE, zlib=True, complevel=4)
        vbar.long_name = 'barotropic v-velocity'
        vbar.units = 'meter second-1'
        vbar.time = 'vbar_time'
        
        out.title = 'ROMS Current Forcing'
        out.history = f'Created {datetime.now()}'
        
        tidx = 0
        for mk in sorted_months:
            prefix = temp_files[mk]
            t = np.load(f'{prefix}_time.npy')
            nt = len(t)
            
            # Write to all time variables
            ocean_time[tidx:tidx+nt] = t
            ubar_time[tidx:tidx+nt] = t
            vbar_time[tidx:tidx+nt] = t
            
            for vname in ['ubar', 'vbar']:
                data = np.load(f'{prefix}_{vname}.npy')
                out.variables[vname][tidx:tidx+nt] = np.where(np.isnan(data), FILL_VALUE, data)
                del data
            
            tidx += nt
    
    log(f"  DONE: {format_size(os.path.getsize(output_file))} in {format_time(time.time() - t0)}")


def assemble_ice(temp_files, output_file, grid_coords):
    """Assemble ice forcing with ice_time dimension."""
    log(f"Assembling {os.path.basename(output_file)}...")
    t0 = time.time()
    
    sorted_months = sorted([k for k, v in temp_files.items() if v])
    if not sorted_months:
        return
    
    eta, xi = grid_coords['eta_rho'], grid_coords['xi_rho']
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as out:
        out.createDimension('xi_rho', xi)
        out.createDimension('eta_rho', eta)
        out.createDimension('ice_time', None)  # Use ice_time as dimension
        
        out.createVariable('lon_rho', 'f8', ('eta_rho', 'xi_rho'))[:] = grid_coords['lon_rho']
        out.createVariable('lat_rho', 'f8', ('eta_rho', 'xi_rho'))[:] = grid_coords['lat_rho']
        
        tvar = out.createVariable('ice_time', 'f8', ('ice_time',))
        tvar.units = TIME_UNITS
        tvar.calendar = CALENDAR
        tvar.long_name = 'ice forcing time'
        
        vars_info = {
            'aice': {'long_name': 'sea ice concentration', 'units': 'fraction'},
            'hice': {'long_name': 'sea ice thickness', 'units': 'meter'},
            'uice': {'long_name': 'sea ice u-velocity', 'units': 'meter second-1'},
            'vice': {'long_name': 'sea ice v-velocity', 'units': 'meter second-1'},
        }
        
        for vname, vinfo in vars_info.items():
            v = out.createVariable(vname, 'f4', ('ice_time', 'eta_rho', 'xi_rho'),
                                   fill_value=FILL_VALUE, zlib=True, complevel=4)
            v.long_name = vinfo['long_name']
            v.units = vinfo['units']
            v.time = 'ice_time'
        
        out.title = 'ROMS Ice Forcing'
        out.history = f'Created {datetime.now()}'
        
        tidx = 0
        for mk in sorted_months:
            prefix = temp_files[mk]
            t = np.load(f'{prefix}_time.npy')
            nt = len(t)
            tvar[tidx:tidx+nt] = t
            
            for vname in vars_info.keys():
                data = np.load(f'{prefix}_{vname}.npy')
                out.variables[vname][tidx:tidx+nt] = np.where(np.isnan(data), FILL_VALUE, data)
                del data
            
            tidx += nt
    
    log(f"  DONE: {format_size(os.path.getsize(output_file))} in {format_time(time.time() - t0)}")


# =============================================================================
# Diagnostic Functions
# =============================================================================

def diagnose_file(filepath, expected_vars, file_type):
    """Diagnose a forcing file for correctness."""
    log(f"\n  Diagnosing {os.path.basename(filepath)}...")
    
    if not os.path.exists(filepath):
        log(f"    ✗ File not found!")
        return False
    
    issues = []
    
    with nc.Dataset(filepath, 'r') as ds:
        # Check dimensions
        log(f"    Dimensions: {dict(ds.dimensions)}")
        
        # Check variables
        log(f"    Variables: {list(ds.variables.keys())}")
        
        for var_name in expected_vars:
            if var_name not in ds.variables:
                issues.append(f"Missing variable: {var_name}")
                continue
            
            var = ds.variables[var_name]
            data = var[:]
            
            # Check for fill values
            fill_count = np.sum(data >= 1e30)
            nan_count = np.sum(np.isnan(data))
            valid_count = np.sum((data < 1e30) & (~np.isnan(data)))
            total = data.size
            
            # Get valid data statistics
            valid_data = data[(data < 1e30) & (~np.isnan(data))]
            
            if len(valid_data) > 0:
                vmin, vmax = np.min(valid_data), np.max(valid_data)
                vmean = np.mean(valid_data)
                log(f"    ✓ {var_name:12s}: {valid_count:,}/{total:,} valid, range=[{vmin:.3f}, {vmax:.3f}], mean={vmean:.3f}")
            else:
                log(f"    ✗ {var_name:12s}: NO VALID DATA!")
                issues.append(f"No valid data in {var_name}")
            
            # Check fill value attribute
            if hasattr(var, '_FillValue'):
                if var._FillValue != FILL_VALUE:
                    issues.append(f"{var_name}: non-standard fill value ({var._FillValue})")
        
        # Check time variables
        time_vars = [v for v in ds.variables if 'time' in v.lower()]
        log(f"    Time variables: {time_vars}")
        
        for tvar in time_vars:
            t = ds.variables[tvar][:]
            if len(t) > 0:
                t_valid = t[(t > 0) & (t < 1e10)]
                if len(t_valid) > 0:
                    # Convert to dates
                    try:
                        dates = nc.num2date(t_valid[[0, -1]], TIME_UNITS)
                        log(f"      {tvar}: {len(t_valid)} steps, {dates[0]} to {dates[1]}")
                    except:
                        log(f"      {tvar}: {len(t_valid)} steps, range=[{t_valid[0]:.1f}, {t_valid[-1]:.1f}]")
    
    if issues:
        log(f"    ⚠ Issues found:")
        for issue in issues:
            log(f"      - {issue}")
        return False
    else:
        log(f"    ✓ File OK")
        return True


def run_diagnostics():
    """Run diagnostics on all output files."""
    log("\n" + "="*70)
    log("DIAGNOSTICS")
    log("="*70)
    
    all_ok = True
    
    # Bulk forcing
    bulk_vars = ['Pair', 'Tair', 'Qair', 'Uwind', 'Vwind', 'swrad', 'swrad_down', 
                 'lwrad', 'lwrad_down', 'rain', 'cloud']
    ok = diagnose_file(os.path.join(OUTPUT_DIR, OUTPUT_FILES['bulk']), bulk_vars, 'bulk')
    all_ok = all_ok and ok
    
    # Wave forcing - check dimension and time variable
    wave_vars = ['Hwave', 'Dwave', 'Pwave', 'Lwave', 'Uwave', 'Vwave']
    ok = diagnose_file(os.path.join(OUTPUT_DIR, OUTPUT_FILES['wave']), wave_vars, 'wave')
    all_ok = all_ok and ok
    # Verify wave_time dimension
    wave_file = os.path.join(OUTPUT_DIR, OUTPUT_FILES['wave'])
    if os.path.exists(wave_file):
        with nc.Dataset(wave_file, 'r') as ds:
            if 'wave_time' in ds.dimensions:
                log(f"    ✓ Dimension 'wave_time' exists")
            else:
                log(f"    ✗ Dimension 'wave_time' MISSING")
                all_ok = False
    
    # Ocean forcing - check for sst_time dimension and sss_time, ssh_time variables
    ocean_vars = ['SST', 'SSS', 'zeta']
    ok = diagnose_file(os.path.join(OUTPUT_DIR, OUTPUT_FILES['ocean']), ocean_vars, 'ocean')
    all_ok = all_ok and ok
    ocean_file = os.path.join(OUTPUT_DIR, OUTPUT_FILES['ocean'])
    if os.path.exists(ocean_file):
        with nc.Dataset(ocean_file, 'r') as ds:
            for check in ['sst_time', 'sss_time', 'ssh_time']:
                if check in ds.variables:
                    log(f"    ✓ Variable '{check}' exists")
                else:
                    log(f"    ✗ Variable '{check}' MISSING")
                    all_ok = False
            if 'sst_time' in ds.dimensions:
                log(f"    ✓ Dimension 'sst_time' exists")
            else:
                log(f"    ✗ Dimension 'sst_time' MISSING")
                all_ok = False
    
    # Current forcing - check for ocean_time dimension and ubar_time, vbar_time variables
    current_vars = ['ubar', 'vbar']
    ok = diagnose_file(os.path.join(OUTPUT_DIR, OUTPUT_FILES['current']), current_vars, 'current')
    all_ok = all_ok and ok
    current_file = os.path.join(OUTPUT_DIR, OUTPUT_FILES['current'])
    if os.path.exists(current_file):
        with nc.Dataset(current_file, 'r') as ds:
            for check in ['ocean_time', 'ubar_time', 'vbar_time']:
                if check in ds.variables:
                    log(f"    ✓ Variable '{check}' exists")
                else:
                    log(f"    ✗ Variable '{check}' MISSING")
                    all_ok = False
            if 'ocean_time' in ds.dimensions:
                log(f"    ✓ Dimension 'ocean_time' exists")
            else:
                log(f"    ✗ Dimension 'ocean_time' MISSING")
                all_ok = False
    
    # Ice forcing - check for ice_time dimension
    ice_vars = ['aice', 'hice', 'uice', 'vice']
    ok = diagnose_file(os.path.join(OUTPUT_DIR, OUTPUT_FILES['ice']), ice_vars, 'ice')
    all_ok = all_ok and ok
    ice_file = os.path.join(OUTPUT_DIR, OUTPUT_FILES['ice'])
    if os.path.exists(ice_file):
        with nc.Dataset(ice_file, 'r') as ds:
            if 'ice_time' in ds.dimensions:
                log(f"    ✓ Dimension 'ice_time' exists")
            else:
                log(f"    ✗ Dimension 'ice_time' MISSING")
                all_ok = False
    
    return all_ok


# =============================================================================
# Main
# =============================================================================

def main():
    global PROGRESS_FILE
    
    total_start = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_dir = os.path.join(OUTPUT_DIR, 'temp_monthly')
    os.makedirs(temp_dir, exist_ok=True)
    
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'progress.log')
    with open(PROGRESS_FILE, 'w') as f:
        f.write(f"Started: {datetime.now()}\n")
    
    log("="*70)
    log("ROMS Forcing Generator - Updated with Full Variable Set")
    log(f"Cores: {N_CORES}")
    log("="*70)
    
    if not os.path.exists(GRID_FILE):
        log(f"ERROR: Grid not found: {GRID_FILE}")
        return
    
    log(f"Grid: {GRID_FILE}")
    log(f"ERA5: {ERA5_INPUT_DIR}")
    log(f"CMEMS: {CMEMS_INPUT_DIR}")
    log(f"Output: {OUTPUT_DIR}")
    
    grid_coords = get_grid_coords(GRID_FILE)
    log(f"Grid: {grid_coords['eta_rho']} x {grid_coords['xi_rho']}")
    
    # Time info
    atm_file = os.path.join(ERA5_INPUT_DIR, ERA5_FILES['atmosphere'])
    hourly_time_info = get_time_info(atm_file, 'valid_time') if os.path.exists(atm_file) else None
    if hourly_time_info:
        log(f"ERA5: {hourly_time_info['n_times']} hourly, {len(hourly_time_info['monthly_indices'])} months")
    
    ocean_file = os.path.join(CMEMS_INPUT_DIR, CMEMS_FILES['ocean'])
    daily_time_info = get_time_info(ocean_file, 'time') if os.path.exists(ocean_file) else None
    if daily_time_info:
        log(f"CMEMS: {daily_time_info['n_times']} daily, {len(daily_time_info['monthly_indices'])} months")
    
    # ========== BULK ==========
    if os.path.exists(atm_file):
        log("")
        log("="*60)
        log("BULK FORCING (Pair, Tair, Qair, Wind, Radiation, Rain, Cloud)")
        log("="*60)
        
        time_info = get_time_info(atm_file, 'valid_time')
        args = [(mk, idx, GRID_FILE, ERA5_INPUT_DIR, temp_dir, PROGRESS_FILE) 
                for mk, idx in sorted(time_info['monthly_indices'].items())]
        
        with Pool(N_CORES) as pool:
            results = list(pool.imap_unordered(process_bulk_month, args))
        
        temp_files = {k: v for k, v in results}
        assemble_bulk(temp_files, os.path.join(OUTPUT_DIR, OUTPUT_FILES['bulk']), grid_coords)
    
    # ========== WAVE ==========
    wave_file = os.path.join(ERA5_INPUT_DIR, ERA5_FILES['waves'])
    if os.path.exists(wave_file):
        log("")
        log("="*60)
        log("WAVE FORCING (Hwave, Dwave, Pwave, Lwave, Uwave, Vwave)")
        log("="*60)
        
        time_info = get_time_info(wave_file, 'valid_time')
        args = [(mk, idx, GRID_FILE, ERA5_INPUT_DIR, temp_dir, PROGRESS_FILE)
                for mk, idx in sorted(time_info['monthly_indices'].items())]
        
        with Pool(N_CORES) as pool:
            results = list(pool.imap_unordered(process_wave_month, args))
        
        temp_files = {k: v for k, v in results if v}
        assemble_wave(temp_files, os.path.join(OUTPUT_DIR, OUTPUT_FILES['wave']), grid_coords)
    
    # ========== OCEAN ==========
    if os.path.exists(ocean_file) and hourly_time_info:
        log("")
        log("="*60)
        log("OCEAN FORCING (SST, SSS, zeta)")
        log("="*60)
        
        cmems_time = get_time_info(ocean_file, 'time')
        args = [(mk, idx, hourly_time_info, GRID_FILE, CMEMS_INPUT_DIR, temp_dir, PROGRESS_FILE)
                for mk, idx in sorted(cmems_time['monthly_indices'].items())]
        
        with Pool(N_CORES) as pool:
            results = list(pool.imap_unordered(process_ocean_month, args))
        
        temp_files = {k: v for k, v in results if v}
        assemble_ocean(temp_files, os.path.join(OUTPUT_DIR, OUTPUT_FILES['ocean']), grid_coords)
    
    # ========== CURRENT ==========
    curr_file = os.path.join(CMEMS_INPUT_DIR, CMEMS_FILES['currents'])
    if os.path.exists(curr_file) and hourly_time_info:
        log("")
        log("="*60)
        log("CURRENT FORCING (ubar, vbar)")
        log("="*60)
        
        cmems_time = get_time_info(curr_file, 'time')
        args = [(mk, idx, hourly_time_info, GRID_FILE, CMEMS_INPUT_DIR, temp_dir, PROGRESS_FILE)
                for mk, idx in sorted(cmems_time['monthly_indices'].items())]
        
        with Pool(N_CORES) as pool:
            results = list(pool.imap_unordered(process_current_month, args))
        
        temp_files = {k: v for k, v in results if v}
        assemble_current(temp_files, os.path.join(OUTPUT_DIR, OUTPUT_FILES['current']), grid_coords)
    
    # ========== ICE ==========
    ice_file = os.path.join(CMEMS_INPUT_DIR, CMEMS_FILES['ice'])
    if os.path.exists(ice_file) and hourly_time_info:
        log("")
        log("="*60)
        log("ICE FORCING (aice, hice, uice, vice)")
        log("="*60)
        
        cmems_time = get_time_info(ice_file, 'time')
        args = [(mk, idx, hourly_time_info, GRID_FILE, CMEMS_INPUT_DIR, temp_dir, PROGRESS_FILE)
                for mk, idx in sorted(cmems_time['monthly_indices'].items())]
        
        with Pool(N_CORES) as pool:
            results = list(pool.imap_unordered(process_ice_month, args))
        
        temp_files = {k: v for k, v in results if v}
        assemble_ice(temp_files, os.path.join(OUTPUT_DIR, OUTPUT_FILES['ice']), grid_coords)
    
    # Cleanup
    log("")
    log("Cleaning temp files...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Run diagnostics
    all_ok = run_diagnostics()
    
    log("")
    log("="*70)
    log(f"COMPLETE in {format_time(time.time() - total_start)}")
    log("="*70)
    
    total_size = 0
    for fname in OUTPUT_FILES.values():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            total_size += sz
            log(f"  {fname}: {format_size(sz)}")
    log(f"Total: {format_size(total_size)}")
    
    if all_ok:
        log("\n✓ All files validated successfully!")
    else:
        log("\n⚠ Some issues detected - check diagnostics above")


if __name__ == '__main__':
    main()