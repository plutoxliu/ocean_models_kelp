#!/usr/bin/env python3
"""
3-Year Multi-Site Kelp Drift Simulation using Parcels
Southern Ocean / Antarctica Study
"""

import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle, Variable, Field
from parcels import AdvectionRK4
try:
    from parcels.rng import ParcelsRandom
except ImportError:
    try:
        from parcels.tools import ParcelsRandom
    except ImportError:
        from parcels import ParcelsRandom
from datetime import datetime, timedelta
import xarray as xr
import math

# Handle ErrorCode import differences between Parcels versions
try:
    from parcels import ErrorCode
except ImportError:
    from parcels import StatusCode as ErrorCode

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
GRID_FILE = './ocean_grd_fine.nc'
CURRENTS_FILE = './roms_forcing_fine/roms_frc_currents.nc'
WAVE_FILE = './roms_forcing_fine/roms_frc_wave.nc'
BULK_FILE = './roms_forcing_fine/roms_frc_bulk.nc'
OCEAN_FILE = './roms_forcing_fine/roms_frc_ocean.nc'
ICE_FILE = './roms_forcing_fine/roms_frc_ice.nc'
KH_FILE = './roms_forcing_fine/Kh_field.nc'  # Spatially-varying diffusivity

OUTPUT_FILE = 'kelp_3year_parcels_fine_All.zarr'

# Release sites
RELEASE_SITES = {
    # South Atlantic
    'South_Georgia': {'lon': -36.5, 'lat': -54.5},
    'Falkland_Islands': {'lon': -59.5, 'lat': -51.8},
    'Gough_Island': {'lon': -9.9, 'lat': -40.3},
    
    # South Indian Ocean
    'Marion_Island': {'lon': 37.8, 'lat': -46.9},
    'Prince_Edward': {'lon': 37.7, 'lat': -46.6},
    'Kerguelen': {'lon': 70.2, 'lat': -49.4},
    
    # Southwest Pacific
    'Stewart_Island': {'lon': 167.5, 'lat': -47.2},
    'Auckland_Island': {'lon': 166.3, 'lat': -50.7},
    'Macquarie_Island': {'lon': 158.9, 'lat': -54.5},
    'Chatham_Island': {'lon': -176.5, 'lat': -44.0},
    'Tasmania': {'lon': 148.0, 'lat': -43.4},
    
    # Others
    'Chile_central': {'lon': -73.2, 'lat': -36.8},
    'Chile_south': {'lon': -74.5, 'lat': -43.2},
    'Cape_Horn': {'lon': -67.3, 'lat': -55.9}
}

# Simulation parameters
PARTICLES_PER_RELEASE = 3000
RELEASE_FREQUENCY_DAYS = 7
DURATION_YEARS = 3
START_DATE = datetime(2020, 1, 1)

# Physics parameters
WIND_DRIFT_FACTOR = 0.03  # 3% windage for kelp rafts
HORIZONTAL_DIFFUSIVITY = 500.0  # m²/s

# Sea ice parameters
ICE_CONCENTRATION_THRESHOLD = 0.50  # 50% ice concentration reduces drift
ICE_VELOCITY_REDUCTION = 0.2  # Reduce Stokes/wind drift by 20% in ice
ICE_TRAP_THRESHOLD = 0.90  # 90% ice concentration traps particles

# =============================================================================
# CUSTOM PARTICLE CLASS WITH KELP PHYSICS
# =============================================================================

class KelpParticle(JITParticle):
    """Kelp particle with additional properties for tracking"""
    
    # Original release location
    origin_lon = Variable('origin_lon', dtype=np.float32, initial=0.0)
    origin_lat = Variable('origin_lat', dtype=np.float32, initial=0.0)
    origin_marker = Variable('origin_marker', dtype=np.int32, initial=0)
    
    # Age tracking
    age = Variable('age', dtype=np.float32, initial=0.0, to_write=True)
    
    # Environmental properties
    temp = Variable('temp', dtype=np.float32, initial=np.nan, to_write=True)
    ice_concentration = Variable('ice_concentration', dtype=np.float32, initial=0.0, to_write=True)
    
    # Status tracking
    beached = Variable('beached', dtype=np.int32, initial=0, to_write=True)
    ice_trapped = Variable('ice_trapped', dtype=np.int32, initial=0, to_write=True)
    is_dead = Variable('is_dead', dtype=np.int32, initial=0, to_write=True)
    
    # Previous position (for reverting when hitting land)
    prev_lon = Variable('prev_lon', dtype=np.float32, initial=0.0, to_write=False)
    prev_lat = Variable('prev_lat', dtype=np.float32, initial=0.0, to_write=False)


# =============================================================================
# CUSTOM KERNELS FOR KELP PHYSICS
# =============================================================================

def StorePosition(particle, fieldset, time):
    """Store current position before advection"""
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

def CleanupKernel(particle, fieldset, time):
    if particle.is_dead == 1:
        particle.state = 7

def CheckLatBounds(particle, fieldset, time):
    """
    Check if particle is outside latitude domain and beach if so.
    """
    if particle.beached == 0:
        if particle.lat < fieldset.domain_lat_min:
            particle.beached = 1
            particle_dlon += particle.prev_lon - particle.lon
            particle_dlat += particle.prev_lat - particle.lat
        if particle.lat > fieldset.domain_lat_max:
            particle.beached = 1
            particle_dlon += particle.prev_lon - particle.lon
            particle_dlat += particle.prev_lat - particle.lat

def KelpAdvection(particle, fieldset, time):
    """
    Combined advection kernel.
    """
    # Skip if beached
    if particle.beached == 1:
        pass  # Can't use return in Parcels kernels
    else:
        # Get Environmental Data
        ice_concentration = fieldset.ice_conc[time, particle.depth, particle.lat, particle.lon]
        
        # Clamp ice
        if ice_concentration < 0.0:
            ice_concentration = 0.0
        if ice_concentration > 1.0:
            ice_concentration = 1.0
        
        # Get ocean currents
        u_current = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        v_current = fieldset.V[time, particle.depth, particle.lat, particle.lon]
        
        # Ice factor (reduces drift in ice)
        ice_factor = 1.0
        if ice_concentration > fieldset.ice_threshold:
            ice_factor = 1.0 - ((ice_concentration - fieldset.ice_threshold) / 
                               (1.0 - fieldset.ice_threshold)) * (1.0 - fieldset.ice_reduction)

        # Stokes Drift
        u_stokes = 0.0
        v_stokes = 0.0
        wave_height = fieldset.Hwave[time, particle.depth, particle.lat, particle.lon]
        
        if wave_height > 0:
            wave_period = fieldset.Pwave[time, particle.depth, particle.lat, particle.lon]
            wave_length = fieldset.Lwave[time, particle.depth, particle.lat, particle.lon]
            wave_direction = fieldset.Dwave[time, particle.depth, particle.lat, particle.lon]
            
            if wave_period > 0 and wave_length > 0:
                omega = 2.0 * 3.141592653589793 / wave_period
                k = 2.0 * 3.141592653589793 / wave_length
                stokes_speed = 0.5 * omega * k * wave_height * wave_height
                
                # Wave direction: oceanographic convention (direction to)
                wave_dir_rad = (90.0 - wave_direction) * 3.141592653589793 / 180.0
                
                u_stokes = stokes_speed * math.cos(wave_dir_rad) * ice_factor
                v_stokes = stokes_speed * math.sin(wave_dir_rad) * ice_factor

        # Wind Drift
        u_wind = fieldset.Uwind[time, particle.depth, particle.lat, particle.lon]
        v_wind = fieldset.Vwind[time, particle.depth, particle.lat, particle.lon]
        u_windage = 0.02 * u_wind * ice_factor
        v_windage = 0.02 * v_wind * ice_factor
        
        # Ice Advection
        u_ice_adv = 0.0
        v_ice_adv = 0.0
        if ice_concentration > fieldset.ice_threshold:
            u_ice = fieldset.ice_u[time, particle.depth, particle.lat, particle.lon]
            v_ice = fieldset.ice_v[time, particle.depth, particle.lat, particle.lon]
            ice_weight = (ice_concentration - fieldset.ice_threshold) / (1.0 - fieldset.ice_threshold)
            u_ice_adv = u_ice * ice_weight
            v_ice_adv = v_ice * ice_weight

        # Total Velocity
        u_total = u_current + u_stokes + u_windage + u_ice_adv
        v_total = v_current + v_stokes + v_windage + v_ice_adv
        
        # Update Position directly
        deg_to_m_lat = 111320.0
        deg_to_m_lon = 111320.0 * math.cos(particle.lat * 3.141592653589793 / 180.0)
        
        if deg_to_m_lon < 1000.0:
            deg_to_m_lon = 1000.0

        particle_dlon += (u_total / deg_to_m_lon) * particle.dt
        particle_dlat += (v_total / deg_to_m_lat) * particle.dt

def CheckLandMask(particle, fieldset, time):
    """
    Check if particle is on land.
    If yes, REVERT to previous position and mark beached.
    """
    if particle.beached == 0:
        # Sample mask (1=water, 0=land)
        mask_value = fieldset.land_mask[time, particle.depth, particle.lat, particle.lon]
        
        if mask_value < 0.5:
            particle.beached = 1
            # Revert position to stop it from entering land
            particle_dlon += particle.prev_lon - particle.lon
            particle_dlat += particle.prev_lat - particle.lat

def DiffusionKh(particle, fieldset, time):
    """
    Apply horizontal diffusion using spatially-varying Kh field.
    
    Uses a simple random walk:
    dx = sqrt(2 * Kh * dt) * random_normal
    
    This represents sub-grid scale turbulent mixing.
    """
    if particle.beached == 0:
        # Get local Kh value
        Kh_local = fieldset.Kh[time, particle.depth, particle.lat, particle.lon]
        
        # Ensure Kh is positive
        if Kh_local < 0:
            Kh_local = 0
        
        if Kh_local > 0:
            # Random displacement scale
            # std_dev = sqrt(2 * Kh * dt) in meters
            std_dev = math.sqrt(2.0 * Kh_local * particle.dt)
            
            # Generate random displacements (approximate normal distribution)
            # Using sum of uniform randoms to approximate Gaussian
            # ParcelsRandom.uniform returns [0,1), so transform to [-1,1)
            rand_u = 2.0 * (ParcelsRandom.uniform(0., 1.) - 0.5)
            rand_v = 2.0 * (ParcelsRandom.uniform(0., 1.) - 0.5)
            
            # Scale by standard deviation (factor ~1.73 to match Gaussian std)
            dx = std_dev * rand_u * 1.73
            dy = std_dev * rand_v * 1.73
            
            # Convert to degrees
            deg_to_m_lat = 111320.0
            deg_to_m_lon = 111320.0 * math.cos(particle.lat * 3.141592653589793 / 180.0)
            
            if deg_to_m_lon < 1000.0:
                deg_to_m_lon = 1000.0
            
            particle_dlon += dx / deg_to_m_lon
            particle_dlat += dy / deg_to_m_lat

def SampleTemp(particle, fieldset, time):
    """Sample sea surface temperature"""
    if particle.beached == 0:
        particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]

def SampleIce(particle, fieldset, time):
    """Sample sea ice concentration"""
    if particle.beached == 0:
        ice_concentration_value = fieldset.ice_conc[time, particle.depth, particle.lat, particle.lon]
        
        # Clamp between 0 and 1
        if ice_concentration_value < 0.0:
            ice_concentration_value = 0.0
        if ice_concentration_value > 1.0:
            ice_concentration_value = 1.0
        
        particle.ice_concentration = ice_concentration_value
        
        # Check if trapped in heavy ice
        if ice_concentration_value > fieldset.ice_trap:
            particle.ice_trapped = 1

def AgeParticle(particle, fieldset, time):
    """Increment particle age"""
    particle.age += particle.dt

def BeachingCheck(particle, fieldset, time):
    """Check for beaching/stranding from error states"""
    # Use numeric comparison for status - works in all versions
    if particle.state >= 50:  # Error states are >= 50
        particle.beached = 1

def RecoveryKernel(particle, fieldset, time):
    """
    Recovery kernel that handles out-of-bounds errors by beaching the particle.
    """
    if particle.state >= 50:  # Only act on error states
        particle.beached = 1
        # Revert to previous valid position
        particle_dlon += particle.prev_lon - particle.lon
        particle_dlat += particle.prev_lat - particle.lat
        particle.is_dead = 1
        particle.state = 1
    
# =============================================================================
# LOAD FIELDSET
# =============================================================================

def load_roms_fieldset():
    """Load ROMS forcing files into Parcels FieldSet"""
    
    print("="*70)
    print("LOADING ROMS FORCING DATA")
    print("="*70)
    
    print("\nLoading forcing files separately...")
    
    # Load each forcing component with its specific time dimension
    
    # 1. Currents (ocean_time)
    print("  Loading currents...")
    fieldset_currents = FieldSet.from_netcdf(
        filenames={'U': CURRENTS_FILE, 'V': CURRENTS_FILE},
        variables={'U': 'ubar', 'V': 'vbar'},
        dimensions={'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'ocean_time'},
        allow_time_extrapolation=False,
        mesh='spherical',  # Tells Parcels to handle lon wrapping at 0/360
    )
    
    # 2. Waves (wave_time) - including wave direction
    print("  Loading waves...")
    fieldset_waves = FieldSet.from_netcdf(
        filenames={'Hwave': WAVE_FILE, 'Lwave': WAVE_FILE, 'Pwave': WAVE_FILE, 'Dwave': WAVE_FILE},
        variables={'Hwave': 'Hwave', 'Lwave': 'Lwave', 'Pwave': 'Pwave', 'Dwave': 'Dwave'},
        dimensions={'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'wave_time'},
        allow_time_extrapolation=False,
        mesh='spherical',
    )
    
    # 3. Wind (wind_time)
    print("  Loading wind...")
    fieldset_wind = FieldSet.from_netcdf(
        filenames={'Uwind': BULK_FILE, 'Vwind': BULK_FILE},
        variables={'Uwind': 'Uwind', 'Vwind': 'Vwind'},
        dimensions={'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'wind_time'},
        allow_time_extrapolation=False,
        mesh='spherical',
    )
    
    # 4. Ocean (sst_time)
    print("  Loading ocean properties...")
    fieldset_ocean = FieldSet.from_netcdf(
        filenames={'temp': OCEAN_FILE},
        variables={'temp': 'SST'},
        dimensions={'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'sst_time'},
        allow_time_extrapolation=False,
        mesh='spherical',
    )
    
    # 5. Sea ice (ice_time)
    print("  Loading sea ice...")
    fieldset_ice = FieldSet.from_netcdf(
        filenames={'ice_conc': ICE_FILE, 'ice_u': ICE_FILE, 'ice_v': ICE_FILE},
        variables={'ice_conc': 'aice', 'ice_u': 'uice', 'ice_v': 'vice'},
        dimensions={'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'ice_time'},
        allow_time_extrapolation=False,
        mesh='spherical',
    )
    
    # 6. Load land mask from grid file
    print("  Loading land mask from grid file...")
    grid_ds = xr.open_dataset(GRID_FILE)
    
    # Get coordinates and mask
    lon_rho = grid_ds['lon_rho'].values
    lat_rho = grid_ds['lat_rho'].values
    mask_rho = grid_ds['mask_rho'].values  # 1 = water, 0 = land
    
    # Get domain bounds for boundary checking
    domain_lat_min = float(lat_rho.min())
    domain_lat_max = float(lat_rho.max())
    domain_lon_min = float(lon_rho.min())
    domain_lon_max = float(lon_rho.max())
    
    grid_ds.close()
    
    print(f"    Land mask shape: {mask_rho.shape}")
    print(f"    Lon range: {domain_lon_min:.1f} to {domain_lon_max:.1f}")
    print(f"    Lat range: {domain_lat_min:.1f} to {domain_lat_max:.1f}")
    print(f"    Water points: {np.sum(mask_rho == 1):,}")
    print(f"    Land points: {np.sum(mask_rho == 0):,}")
    
    # Create land mask field using same approach as other fields
    # Use from_netcdf which handles curvilinear grids properly
    land_mask_field = Field.from_netcdf(
        GRID_FILE,
        variable='mask_rho',
        dimensions={'lon': 'lon_rho', 'lat': 'lat_rho'},
        allow_time_extrapolation=True,
        interp_method='nearest',  # Don't interpolate mask values
        mesh='spherical',
    )
    land_mask_field.name = 'land_mask'  # Rename for use in kernels
    
    # Combine fieldsets
    print("  Combining fieldsets...")
    fieldset = fieldset_currents
    
    # Add wave fields
    fieldset.add_field(fieldset_waves.Hwave)
    fieldset.add_field(fieldset_waves.Lwave)
    fieldset.add_field(fieldset_waves.Pwave)
    fieldset.add_field(fieldset_waves.Dwave)
    
    # Add wind fields
    fieldset.add_field(fieldset_wind.Uwind)
    fieldset.add_field(fieldset_wind.Vwind)
    
    # Add ocean fields
    fieldset.add_field(fieldset_ocean.temp)
    
    # Add ice fields
    fieldset.add_field(fieldset_ice.ice_conc)
    fieldset.add_field(fieldset_ice.ice_u)
    fieldset.add_field(fieldset_ice.ice_v)
    
    # Add land mask field
    fieldset.add_field(land_mask_field)
    
    # 7. Load spatially-varying Kh field (if available)
    import os
    if os.path.exists(KH_FILE):
        print(f"  Loading Kh field: {KH_FILE}")
        fieldset_kh = FieldSet.from_netcdf(
            filenames={'Kh': KH_FILE},
            variables={'Kh': 'Kh'},
            dimensions={'lon': 'lon_rho', 'lat': 'lat_rho', 'time': 'kh_time'},
            allow_time_extrapolation=True,
            mesh='spherical',
        )
        fieldset.add_field(fieldset_kh.Kh)
        print("    ✓ Spatially-varying Kh loaded")
    else:
        print(f"  Warning: Kh file not found: {KH_FILE}")
        print(f"    Using constant Kh = {HORIZONTAL_DIFFUSIVITY} m²/s")
        # Create constant Kh field from grid
        Kh_data = np.full(mask_rho.shape, HORIZONTAL_DIFFUSIVITY, dtype=np.float32)
        Kh_data = np.where(mask_rho == 0, 0.0, Kh_data)  # Zero on land
        Kh_field = Field(
            name='Kh',
            data=Kh_data[np.newaxis, :, :],  # Add time dimension
            lon=lon_rho,
            lat=lat_rho,
            mesh='spherical',
            allow_time_extrapolation=True,
        )
        fieldset.add_field(Kh_field)
    
    # Add horizontal diffusivity constants (kept for backwards compatibility)
    fieldset.add_constant('Kh_zonal', HORIZONTAL_DIFFUSIVITY)
    fieldset.add_constant('Kh_meridional', HORIZONTAL_DIFFUSIVITY)
    
    # Add ice parameters as constants
    fieldset.add_constant('ice_threshold', ICE_CONCENTRATION_THRESHOLD)
    fieldset.add_constant('ice_reduction', ICE_VELOCITY_REDUCTION)
    fieldset.add_constant('ice_trap', ICE_TRAP_THRESHOLD)
    
    # Add domain bounds for boundary checking (0.5 degree buffer from edge)
    fieldset.add_constant('domain_lat_min', domain_lat_min - 0.5)
    fieldset.add_constant('domain_lat_max', domain_lat_max + 0.5)
    fieldset.add_constant('domain_lon_min', domain_lon_min - 0.5)
    fieldset.add_constant('domain_lon_max', domain_lon_max + 0.5)
    
    print(f"    Domain bounds (with 0.5° buffer):")
    print(f"      Lat: {domain_lat_min - 0.5:.1f} to {domain_lat_max + 0.5:.1f}")
    print(f"      Lon: {domain_lon_min - 0.5:.1f} to {domain_lon_max + 0.5:.1f}")
    
    # Add periodic halo for zonal (longitude) wrapping
    # This allows particles to cross the 0/360 boundary seamlessly
    fieldset.add_periodic_halo(zonal=True, meridional=False)
    print("  Added periodic halo for zonal wrapping")
    
    print("✓ Fieldset loaded successfully")
    
    return fieldset

# =============================================================================
# CREATE PARTICLE SET
# =============================================================================

def find_nearest_ocean(lon, lat, lon_grid, lat_grid, mask, max_search_deg=1.0):
    """
    Find nearest ocean point if particle starts on land.
    
    Parameters:
    -----------
    lon, lat : float
        Particle position
    lon_grid, lat_grid : 2D arrays
        Grid coordinates
    mask : 2D array
        Land mask (1=ocean, 0=land)
    max_search_deg : float
        Maximum search radius in degrees
    
    Returns:
    --------
    (new_lon, new_lat, is_valid) : tuple
        New position and whether a valid ocean point was found
    """
    # Find nearest grid point
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    
    # Check if already in ocean
    if mask[idx] == 1:
        return lon, lat, True
    
    # Search for nearest ocean point in expanding radius
    eta_size, xi_size = mask.shape
    
    for radius in range(1, int(max_search_deg * 10) + 1):  # ~0.1 deg steps
        search_rad = radius * 0.1  # degrees
        
        # Get indices within search radius
        nearby_mask = dist < search_rad
        
        # Find ocean points in this radius
        ocean_nearby = nearby_mask & (mask == 1)
        
        if np.any(ocean_nearby):
            # Find the closest ocean point
            ocean_dist = np.where(ocean_nearby, dist, np.inf)
            ocean_idx = np.unravel_index(np.argmin(ocean_dist), ocean_dist.shape)
            
            new_lon = lon_grid[ocean_idx]
            new_lat = lat_grid[ocean_idx]
            return new_lon, new_lat, True
    
    # No ocean point found within max_search_deg
    return lon, lat, False

def create_particle_set(fieldset, sites=None):
    """Create particle set with periodic releases, ensuring all particles start in ocean"""
    
    print("\n" + "="*70)
    print("CREATING PARTICLE SET")
    print("="*70)
    
    # Load grid for land mask checking
    print("\nLoading grid for ocean/land checking...")
    grid_ds = xr.open_dataset(GRID_FILE)
    lon_grid = grid_ds['lon_rho'].values
    lat_grid = grid_ds['lat_rho'].values
    mask_grid = grid_ds['mask_rho'].values  # 1=ocean, 0=land
    grid_ds.close()
    
    # Select sites
    if sites is None:
        sites_to_use = RELEASE_SITES
    else:
        sites_to_use = {k: RELEASE_SITES[k] for k in sites if k in RELEASE_SITES}
    
    print(f"\nUsing {len(sites_to_use)} release sites")
    
    # Calculate releases
    num_releases = int((DURATION_YEARS * 365) / RELEASE_FREQUENCY_DAYS)
    total_particles = len(sites_to_use) * PARTICLES_PER_RELEASE * num_releases
    
    print(f"Releases per site: {num_releases}")
    print(f"Particles per release: {PARTICLES_PER_RELEASE}")
    print(f"Total particles (max): {total_particles:,}")
    
    # Prepare particle data
    lons = []
    lats = []
    times = []
    origins_lon = []
    origins_lat = []
    origins_marker = []
    
    # Track statistics
    n_nudged = 0
    n_skipped = 0
    
    print("\nGenerating particle release schedule...")
    print("  (checking each particle is in ocean, nudging if needed)")
    
    for release_num in range(num_releases):
        release_time = START_DATE + timedelta(days=release_num * RELEASE_FREQUENCY_DAYS)
        
        for site_id, (site_name, site_info) in enumerate(sites_to_use.items()):
            
            # Get site longitude and convert to 0-360 if needed
            site_lon = site_info['lon']
            if site_lon < 0:
                site_lon = site_lon + 360.0
            
            # Add small random spread around release point (~10km radius)
            for _ in range(PARTICLES_PER_RELEASE):
                # Random offset in degrees
                lon_offset = np.random.uniform(-0.1, 0.1)
                lat_offset = np.random.uniform(-0.1, 0.1)
                
                particle_lon = site_lon + lon_offset
                particle_lat = site_info['lat'] + lat_offset
                
                # Ensure still in 0-360 range
                if particle_lon < 0:
                    particle_lon += 360.0
                if particle_lon >= 360:
                    particle_lon -= 360.0
                
                # Check if in ocean, nudge if on land
                new_lon, new_lat, is_valid = find_nearest_ocean(
                    particle_lon, particle_lat, lon_grid, lat_grid, mask_grid
                )
                
                if not is_valid:
                    n_skipped += 1
                    continue
                
                if new_lon != particle_lon or new_lat != particle_lat:
                    n_nudged += 1
                    particle_lon = new_lon
                    particle_lat = new_lat
                
                lons.append(particle_lon)
                lats.append(particle_lat)
                times.append(release_time)
                origins_lon.append(site_lon)  # Use original site longitude
                origins_lat.append(site_info['lat'])
                origins_marker.append(site_id)
        
        if (release_num + 1) % 12 == 0:
            print(f"  Year {(release_num + 1) // 12}: {len(lons):,} particles scheduled")
    
    print(f"\n✓ Total particles created: {len(lons):,}")
    print(f"  Nudged from land to ocean: {n_nudged:,}")
    print(f"  Skipped (no nearby ocean): {n_skipped:,}")
    
    # Create ParticleSet
    print("\nInitializing particle set...")
    
    pset = ParticleSet.from_list(
        fieldset=fieldset,
        pclass=KelpParticle,
        lon=lons,
        lat=lats,
        time=times,
        origin_lon=origins_lon,
        origin_lat=origins_lat,
        origin_marker=origins_marker,
        prev_lon=lons,  # Initialize prev position
        prev_lat=lats,
        depth=np.zeros(len(lons)),  # Surface particles (array of zeros)
    )
    
    print(f"✓ Particle set initialized")
    
    return pset

# =============================================================================
# RUN SIMULATION
# =============================================================================

def run_simulation(fieldset, pset):
    """Execute the particle tracking simulation"""
    
    print("\n" + "="*70)
    print("RUNNING SIMULATION")
    print("="*70)
    
    # Set up output file
    output_file = pset.ParticleFile(
        name=OUTPUT_FILE,
        outputdt=timedelta(hours=24)  # Daily output
    )
    
    # Define kernels (order matters!)
    # 1. Store position before advection
    # 1.2 Clean up dead particles
    # 2. Advection (currents + Stokes + wind + ice)
    # 3. Diffusion (spatially-varying Kh)
    # 4. Periodic boundary (wrap longitude 0-360)
    # 5. Check latitude bounds
    # 6. Check land mask and beach if on land
    # 7. Sample environment and update age
    # 8. Recovery kernel
    kernels = (
        pset.Kernel(StorePosition) +          # Store position before advection
        pset.Kernel(CleanupKernel) +          # Clean up outofbonds particles
        pset.Kernel(KelpAdvection) +          # Combined advection with Stokes + wind + ice
        pset.Kernel(DiffusionKh) +            # Spatially-varying horizontal diffusion
        pset.Kernel(CheckLatBounds) +         # Check latitude domain boundaries
        pset.Kernel(CheckLandMask) +          # Check land mask and beach particles
        pset.Kernel(AgeParticle) +            # Age tracking
        pset.Kernel(SampleTemp) +             # Sample temperature
        pset.Kernel(SampleIce) +              # Sample ice concentration
        pset.Kernel(BeachingCheck) +          # Check for error states
        pset.Kernel(RecoveryKernel)           # Revert to last location for beached particles  
    )
    
    # Run simulation
    print(f"\nStart: {START_DATE}")
    end_date = START_DATE + timedelta(days=int(DURATION_YEARS * 365))
    print(f"End: {end_date}")
    print(f"Duration: {DURATION_YEARS} years")
    print("\nRunning... this will take a while")
    print("Progress will be logged periodically\n")
    
    pset.execute(
        kernels,
        runtime=timedelta(days=DURATION_YEARS * 365),
        dt=timedelta(hours=1),              # 1 hour timestep
        output_file=output_file,
        verbose_progress=True
    )
    
    print("\n✓ Simulation complete!")
    print(f"Output saved to: {OUTPUT_FILE}")

# =============================================================================
# POST-PROCESSING
# =============================================================================

def analyze_results():
    """Quick analysis of results"""
    
    print("\n" + "="*70)
    print("ANALYZING RESULTS")
    print("="*70)
    
    # Load output
    ds = xr.open_zarr(OUTPUT_FILE)
    
    print(f"\nParticles: {len(ds.trajectory)}")
    print(f"Timesteps: {len(ds.obs)}")
    
    # Count beached particles
    beached = ds.beached.values
    n_beached = np.sum(np.nanmax(beached, axis=1) == 1)  # Any timestep beached
    
    print(f"\nBeached particles: {n_beached:,} ({n_beached/len(ds.trajectory)*100:.1f}%)")
    
    # Check Antarctica arrival (south of 60°S)
    final_lats = ds.lat.isel(obs=-1).values
    reached_antarctica = np.sum(final_lats < -60)
    
    print(f"Reached Antarctica (< 60°S): {reached_antarctica:,} ({reached_antarctica/len(ds.trajectory)*100:.1f}%)")
    
    # Check ice trapped
    ice_trapped = ds.ice_trapped.values
    n_ice_trapped = np.sum(np.nanmax(ice_trapped, axis=1) == 1)
    print(f"Ice trapped: {n_ice_trapped:,} ({n_ice_trapped/len(ds.trajectory)*100:.1f}%)")
    
    print(f"\n✓ Analysis complete")
    print(f"\nResults saved in: {OUTPUT_FILE}")

# =============================================================================
# SCENARIOS
# =============================================================================
SCENARIOS = {
    'all_major': ['South_Georgia', 'Marion_Island', 'Kerguelen',
                  'Macquarie_Island', 'Auckland_Island', 'Gough_Island'],
    
    'atlantic': ['South_Georgia', 'Falkland_Islands','Gough_Island'],
    
    'indian': ['Marion_Island', 'Prince_Edward', 'Kerguelen'],
    
    'pacific': ['Stewart_Island', 'Auckland_Island', 'Macquarie_Island',
               'Chatham_Island', 'Tasmania'],
    
    'south_america': ['Chile_central', 'Chile_south', 'Cape_Horn'],
}

# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    
    print("="*70)
    print("KELP DRIFT SIMULATION - PARCELS VERSION")
    print("Southern Ocean / Antarctica Connectivity Study")
    print("With Land Mask Beaching")
    print("="*70)
    
    # Get scenario from command line
    if len(sys.argv) > 1:
        scenario = sys.argv[1].lower()
    else:
        print("\nUsage: python parcel_ice.py <scenario>")
        print("Scenarios: all_major, atlantic, indian, pacific, south_america, all")
        sys.exit(1)
    
    # Select sites
    if scenario == 'all':
        selected_sites = None
        scenario_name = "All Sites"
    elif scenario in SCENARIOS:
        selected_sites = SCENARIOS[scenario]
        scenario_name = scenario.replace('_', ' ').title()
    else:
        print(f"Unknown scenario: {scenario}")
        print("Available: all_major, atlantic, indian, pacific, south_america, all")
        sys.exit(1)
    
    print(f"\nScenario: {scenario_name}")
    if selected_sites:
        print(f"Sites: {', '.join(selected_sites)}")
    
    # Load fieldset
    fieldset = load_roms_fieldset()
    
    # Create particles
    pset = create_particle_set(fieldset, sites=selected_sites)
    
    # Run simulation
    run_simulation(fieldset, pset)
    
    # Analyze results
    analyze_results()
    
    print("\n" + "="*70)
    print("✓ ALL COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()