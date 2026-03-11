"""
Generate ROMS Nudging Coefficients File for Southern Ocean Simulation

Creates a sponge layer at the northern boundary with spatially-varying
nudging coefficients that transition smoothly from 0 in the interior
to maximum value at the boundary.

Works for both fine (~9km) and coarse (~28km) grids.

ROMS requires:
  - M2_NudgeCoef(eta_rho, xi_rho)           - 2D
  - M3_NudgeCoef(s_rho, eta_rho, xi_rho)    - 3D
  - temp_NudgeCoef(s_rho, eta_rho, xi_rho)  - 3D
  - salt_NudgeCoef(s_rho, eta_rho, xi_rho)  - 3D
  - zeta_NudgeCoef(eta_rho, xi_rho)         - 2D

Usage:
    python nudge_coefficients.py
"""

import numpy as np
import netCDF4 as nc
from datetime import datetime
import os

# Configuration for both grids
GRID_CONFIGS = {
    'fine': {
        'grid_file': 'ocean_grd_fine.nc',
        'ini_file': 'roms_ini_fine.nc',        # To get s_rho dimension
        'output_file': 'roms_nudge_fine.nc',
        'sponge_width': 30,      # Wider for fine grid
        'obcfac': 10,          # 30-day timescale (1/3 day^-1)
        'n_s_rho': 30,           # Number of vertical levels (fallback if ini_file not found)
    },
    'coarse': {
        'grid_file': 'ocean_grd_test.nc',
        'ini_file': 'roms_ini_test.nc',      # To get s_rho dimension
        'output_file': 'roms_nudge_test.nc',
        'sponge_width': 10,     
        'obcfac': 10,          # 30-day timescale (1/3 day^-1)
        'n_s_rho': 30,           # Number of vertical levels (fallback if ini_file not found)
    }
}

# Common parameters
nudge_north = True          # Apply sponge to northern boundary
nudge_south = False         # Apply sponge to southern boundary (closed)
transition_type = "quadratic"  # Options: "linear", "quadratic", "exponential"


def create_transition(n_cells, transition_type="linear"):
    """
    Create transition function from 0 to 1 over n_cells.
    
    Parameters:
    -----------
    n_cells : int
        Number of cells for transition
    transition_type : str
        Type of transition function
        
    Returns:
    --------
    transition : array
        Transition values from 0 to 1
    """
    x = np.linspace(0, 1, n_cells)
    
    if transition_type == "linear":
        transition = x
    elif transition_type == "quadratic":
        transition = x**2
    elif transition_type == "exponential":
        transition = (np.exp(3*x) - 1) / (np.exp(3) - 1)
    else:
        raise ValueError(f"Unknown transition type: {transition_type}")
    
    return transition


def get_n_s_rho(config):
    """
    Get number of vertical levels from initial conditions file or config.
    """
    ini_file = config.get('ini_file', '')
    
    if os.path.exists(ini_file):
        try:
            with nc.Dataset(ini_file, 'r') as ds:
                if 's_rho' in ds.dimensions:
                    n_s_rho = len(ds.dimensions['s_rho'])
                    print(f"   Found s_rho={n_s_rho} from {ini_file}")
                    return n_s_rho
        except Exception as e:
            print(f"   Warning: Could not read {ini_file}: {e}")
    
    # Fallback to config value
    n_s_rho = config.get('n_s_rho', 30)
    print(f"   Using default s_rho={n_s_rho}")
    return n_s_rho


def create_nudging_coefficients(grid_name, config):
    """
    Create nudging coefficients file with sponge layer.
    
    Parameters:
    -----------
    grid_name : str
        Name of the grid configuration ('fine' or 'coarse')
    config : dict
        Configuration dictionary with grid_file, output_file, etc.
    """
    grid_file = config['grid_file']
    output_file = config['output_file']
    sponge_width = config['sponge_width']
    obcfac = config['obcfac']
    
    print("="*70)
    print(f"ROMS Nudging Coefficients Generator - {grid_name.upper()} GRID")
    print("="*70)
    
    # Check if grid file exists
    if not os.path.exists(grid_file):
        print(f"\n⚠️  ERROR: Grid file not found: {grid_file}")
        print(f"   Skipping {grid_name} grid...")
        return None
    
    # Get number of vertical levels
    print(f"\n1. Getting vertical levels...")
    n_s_rho = get_n_s_rho(config)
    
    # Read grid dimensions
    print(f"\n2. Reading grid file: {grid_file}")
    with nc.Dataset(grid_file, 'r') as grd:
        h = grd.variables['h'][:]
        mask_rho = grd.variables['mask_rho'][:]
        
        eta_rho = mask_rho.shape[0]
        xi_rho = mask_rho.shape[1]
        
        print(f"   Grid dimensions: eta_rho={eta_rho}, xi_rho={xi_rho}, s_rho={n_s_rho}")
        print(f"   Ocean points: {(mask_rho==1).sum():,} ({100*(mask_rho==1).sum()/mask_rho.size:.1f}%)")
    
    # Create 2D nudging coefficient array
    print(f"\n3. Creating nudging coefficient arrays")
    nudge_coef_2d = np.zeros((eta_rho, xi_rho), dtype=np.float64)
    
    # Create sponge layer at northern boundary
    if nudge_north:
        print(f"   Creating northern sponge layer:")
        print(f"   - Width: {sponge_width} cells")
        print(f"   - Maximum coefficient: {obcfac:.3f} day^-1 ({1/obcfac:.1f}-day timescale)")
        print(f"   - Transition: {transition_type}")
        
        transition = create_transition(sponge_width, transition_type)
        
        # Apply to northernmost cells
        for i in range(sponge_width):
            eta_idx = eta_rho - sponge_width + i
            if eta_idx >= 0:  # Safety check
                nudge_coef_2d[eta_idx, :] = transition[i] * obcfac
    
    # Create sponge layer at southern boundary (if needed)
    if nudge_south:
        print(f"   Creating southern sponge layer:")
        print(f"   - Width: {sponge_width} cells")
        
        transition = create_transition(sponge_width, transition_type)
        
        # Apply to southernmost cells
        for i in range(sponge_width):
            nudge_coef_2d[i, :] = transition[sponge_width-1-i] * obcfac
    
    # Apply land mask (no nudging on land)
    nudge_coef_2d = nudge_coef_2d * mask_rho
    
    # Create 3D array by broadcasting 2D to all vertical levels
    # Same nudging coefficient at all depths
    nudge_coef_3d = np.broadcast_to(nudge_coef_2d, (n_s_rho, eta_rho, xi_rho)).copy()
    
    # Statistics
    ocean_mask = mask_rho == 1
    print(f"\n4. Nudging coefficient statistics:")
    print(f"   2D shape: {nudge_coef_2d.shape}")
    print(f"   3D shape: {nudge_coef_3d.shape}")
    print(f"   Minimum: {nudge_coef_2d.min():.6f} day^-1")
    print(f"   Maximum: {nudge_coef_2d.max():.6f} day^-1")
    if ocean_mask.sum() > 0:
        print(f"   Mean (water only): {nudge_coef_2d[ocean_mask].mean():.6f} day^-1")
        print(f"   Fraction with nudging: {(nudge_coef_2d > 0).sum() / ocean_mask.sum():.2%}")
    
    # Create output NetCDF file
    print(f"\n5. Creating output file: {output_file}")
    with nc.Dataset(output_file, 'w', format='NETCDF4') as out:
        
        # Global attributes
        out.type = "ROMS NUDGING COEFFICIENTS file"
        out.title = f"Southern Ocean Kelp Dispersal - Nudging Coefficients ({grid_name} grid)"
        out.history = f"Created {datetime.now().isoformat()}"
        out.source = "nudge_coefficients.py"
        out.grid_file = grid_file
        out.sponge_width_cells = sponge_width
        out.max_nudge_coef = obcfac
        out.nudge_timescale_days = 1.0 / obcfac
        out.transition_type = transition_type
        
        # Create dimensions
        out.createDimension('xi_rho', xi_rho)
        out.createDimension('eta_rho', eta_rho)
        out.createDimension('s_rho', n_s_rho)
        out.createDimension('xi_u', xi_rho - 1)
        out.createDimension('eta_v', eta_rho - 1)
        
        # =====================================================================
        # 2D Variables (eta_rho, xi_rho)
        # =====================================================================
        
        # M2_NudgeCoef - 2D momentum (barotropic)
        var = out.createVariable('M2_NudgeCoef', 'f8', ('eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = "2D momentum inverse nudging coefficients"
        var.units = "day-1"
        var.field = "M2_NudgeCoef, scalar"
        var[:] = nudge_coef_2d
        
        # zeta_NudgeCoef - Free surface
        var = out.createVariable('zeta_NudgeCoef', 'f8', ('eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = "free-surface inverse nudging coefficients"
        var.units = "day-1"
        var.field = "zeta_NudgeCoef, scalar"
        var[:] = nudge_coef_2d
        
        # =====================================================================
        # 3D Variables (s_rho, eta_rho, xi_rho) - ROMS REQUIREMENT
        # =====================================================================
        
        # M3_NudgeCoef - 3D momentum (baroclinic)
        var = out.createVariable('M3_NudgeCoef', 'f8', ('s_rho', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = "3D momentum inverse nudging coefficients"
        var.units = "day-1"
        var.field = "M3_NudgeCoef, scalar"
        var[:] = nudge_coef_3d
        
        # temp_NudgeCoef - Temperature
        var = out.createVariable('temp_NudgeCoef', 'f8', ('s_rho', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = "temperature inverse nudging coefficients"
        var.units = "day-1"
        var.field = "temp_NudgeCoef, scalar"
        var[:] = nudge_coef_3d
        
        # salt_NudgeCoef - Salinity
        var = out.createVariable('salt_NudgeCoef', 'f8', ('s_rho', 'eta_rho', 'xi_rho'),
                                 zlib=True, complevel=4)
        var.long_name = "salinity inverse nudging coefficients"
        var.units = "day-1"
        var.field = "salt_NudgeCoef, scalar"
        var[:] = nudge_coef_3d
    
    print(f"\n6. Successfully created: {output_file}")
    
    # Verify the output
    print(f"\n7. Verification:")
    with nc.Dataset(output_file, 'r') as ds:
        print(f"   Dimensions:")
        for name, dim in ds.dimensions.items():
            print(f"      {name}: {len(dim)}")
        print(f"   Variables:")
        for name, var in ds.variables.items():
            print(f"      {name}: {var.dimensions}, shape={var.shape}")
    
    return output_file


def plot_nudging_coefficients(output_file, grid_file, grid_name):
    """
    Create a visualization of the nudging coefficients.
    """
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n8. Creating visualization for {grid_name} grid...")
        
        # Read the nudging file
        with nc.Dataset(output_file, 'r') as nud:
            # Use 2D coef for plotting (M2 or zeta)
            coef = nud.variables['M2_NudgeCoef'][:]
        
        # Read grid for proper coordinates
        with nc.Dataset(grid_file, 'r') as grd:
            if 'lon_rho' in grd.variables:
                lon = grd.variables['lon_rho'][:]
                lat = grd.variables['lat_rho'][:]
            else:
                # Create dummy coordinates
                lon = np.arange(coef.shape[1])
                lat = np.arange(coef.shape[0])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Full domain
        im1 = ax1.pcolormesh(lon, lat, coef, shading='auto', cmap='YlOrRd', vmin=0)
        ax1.set_xlabel('Longitude (°E)', fontsize=11)
        ax1.set_ylabel('Latitude (°N)', fontsize=11)
        ax1.set_title(f'Nudging Coefficients - {grid_name.capitalize()} Grid', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax1, label='Nudging Coefficient (day⁻¹)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Zoom on sponge region (northern boundary)
        zoom_rows = min(50, coef.shape[0])
        coef_zoom = coef[-zoom_rows:, :]
        
        if lon.ndim == 2:
            lon_zoom = lon[-zoom_rows:, :]
            lat_zoom = lat[-zoom_rows:, :]
        else:
            lon_zoom = lon
            lat_zoom = lat[-zoom_rows:]
        
        im2 = ax2.pcolormesh(lon_zoom, lat_zoom, coef_zoom, shading='auto', cmap='YlOrRd', vmin=0)
        ax2.set_xlabel('Longitude (°E)', fontsize=11)
        ax2.set_ylabel('Latitude (°N)', fontsize=11)
        ax2.set_title(f'Northern Boundary Zoom (last {zoom_rows} rows)', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax2, label='Nudging Coefficient (day⁻¹)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_file.replace('.nc', '_plot.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   ✔ Saved plot: {plot_file}")
        plt.close()
        
    except ImportError:
        print("   matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"   Error creating plot: {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ROMS NUDGING COEFFICIENTS GENERATOR")
    print("Generating files for both fine and coarse grids")
    print("="*70)
    print("\nROMS requires:")
    print("  - M2_NudgeCoef(eta_rho, xi_rho)           - 2D")
    print("  - M3_NudgeCoef(s_rho, eta_rho, xi_rho)    - 3D")
    print("  - temp_NudgeCoef(s_rho, eta_rho, xi_rho)  - 3D")
    print("  - salt_NudgeCoef(s_rho, eta_rho, xi_rho)  - 3D")
    print("  - zeta_NudgeCoef(eta_rho, xi_rho)         - 2D")
    
    generated_files = []
    
    # Process both grids
    for grid_name, config in GRID_CONFIGS.items():
        print()
        output_file = create_nudging_coefficients(grid_name, config)
        
        if output_file:
            generated_files.append((grid_name, output_file, config['grid_file']))
            plot_nudging_coefficients(output_file, config['grid_file'], grid_name)
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    if len(generated_files) > 0:
        print(f"\nGenerated {len(generated_files)} nudging file(s):")
        for grid_name, output_file, grid_file in generated_files:
            print(f"  ✔ {output_file} (from {grid_file})")
        
        print("\nTo use these files, add to your ROMS .in file:")
        for grid_name, output_file, _ in generated_files:
            print(f"\n  For {grid_name} grid:")
            print(f"    NUDNAME == {output_file}")
        
        print("\nAnd ensure these are enabled in roms.in:")
        print("    LnudgeM2CLM == T")
        print("    LnudgeM3CLM == T")
        print("    LnudgeTCLM == T T")
        print("="*70)
    else:
        print("\n⚠️  No files generated. Check that grid files exist.")
        print("="*70)