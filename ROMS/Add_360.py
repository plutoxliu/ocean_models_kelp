import numpy as np
import netCDF4 as nc
import shutil
from pathlib import Path

# ============================================================================
# USER CONFIGURATION
# ============================================================================
INPUT_GRID = 'ocean_grd_test.nc'
OUTPUT_GRID = 'ocean_grd_test_extended.nc'
CREATE_BACKUP = True

# ============================================================================

def extend_grid_file(input_file, output_file, backup=True):
    """
    Extend ROMS grid file by adding 360° column (duplicate of 0°).
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Grid file not found: {input_file}")
        return False
    
    # Create backup
    if backup:
        backup_path = input_path.with_suffix('.nc.bak')
        shutil.copy2(input_file, backup_path)
        print(f"Backup created: {backup_path}")
    
    print(f"Reading: {input_file}")
    print(f"Writing: {output_file}")
    
    with nc.Dataset(input_file, 'r') as src:
        with nc.Dataset(output_file, 'w') as dst:
            
            # Copy global attributes
            dst.setncatts(src.__dict__)
            
            # Identify xi dimensions to extend
            xi_dims = ['xi_rho', 'xi_u', 'xi_v', 'xi_psi']
            dims_to_extend = {d: src.dimensions[d].size for d in xi_dims if d in src.dimensions}
            
            print(f"\nExtending dimensions:")
            for dim_name, old_size in dims_to_extend.items():
                print(f"  {dim_name}: {old_size} -> {old_size + 1}")
            
            # Create dimensions
            for name, dimension in src.dimensions.items():
                if name in dims_to_extend:
                    dst.createDimension(name, dims_to_extend[name] + 1)
                else:
                    dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
            
            # Process each variable
            print(f"\nProcessing variables:")
            for name, variable in src.variables.items():
                
                # Create output variable
                outVar = dst.createVariable(
                    name,
                    variable.datatype,
                    variable.dimensions,
                    zlib=True,
                    complevel=4
                )
                
                # Copy attributes
                outVar.setncatts(variable.__dict__)
                
                # Check if variable has xi dimension
                xi_dim_in_var = [d for d in variable.dimensions if d in dims_to_extend]
                
                if xi_dim_in_var:
                    xi_dim_name = xi_dim_in_var[0]
                    xi_idx = variable.dimensions.index(xi_dim_name)
                    
                    data = variable[:]
                    new_shape = list(data.shape)
                    new_shape[xi_idx] += 1
                    extended_data = np.empty(new_shape, dtype=data.dtype)
                    
                    # Copy and extend
                    if xi_idx == 0:
                        extended_data[:-1, ...] = data
                        extended_data[-1, ...] = data[0, ...]
                    elif xi_idx == 1:
                        extended_data[:, :-1, ...] = data
                        extended_data[:, -1, ...] = data[:, 0, ...]
                    elif xi_idx == 2:
                        extended_data[:, :, :-1] = data
                        extended_data[:, :, -1] = data[:, :, 0]
                    
                    # Set longitude to 360.0 for lon_* variables
                    if name.startswith('lon_'):
                        if xi_idx == 1:
                            extended_data[:, -1] = 360.0
                            print(f"  {name}: {data.shape} -> {extended_data.shape} (set to 360.0°)")
                        else:
                            print(f"  {name}: {data.shape} -> {extended_data.shape}")
                    else:
                        print(f"  {name}: {data.shape} -> {extended_data.shape}")
                    
                    outVar[:] = extended_data
                else:
                    # No xi dimension, copy as-is
                    outVar[:] = variable[:]
    
    print(f"\n✓ Grid file extended successfully!")
    print(f" Plot new grid file to verify: {output_file}")

    return True

if __name__ == '__main__':
    extend_grid_file(INPUT_GRID, OUTPUT_GRID, backup=CREATE_BACKUP)