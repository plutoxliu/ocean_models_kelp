Steps to use the codes for generating ROMS model needed files: forcing, climatology, initial condition, sponge and nudging. 
Optimised for circumpolar grids (need to fill datagap at 180 degrees).

1. Create Grid and verify
	Need NSIDC_EASE2-S3.125km.nc for land mask & ETOPO1_Ice_g_gmt4.nc for bathymetry
	Make_grid.py
	
	2. Add grid missing variables:
	Add_grd_var.py
	
	3. Add the last column to make 360 circumpolar, and verify
	Add_360.py
	Fix_grid.py
	Plot_grd.py
	
	4. Generate forcing files
	forcing_.py
	
	5. Fix forcing file time variables
	Fixtime.py
	
	6. Check and fix missing values at 180 degrees (cmems only goes up to 179.1) 
	and any fill values of forcing file and verify
	Fill_180.py
	Plot_var.py
	
	7. Generate other needed files for ROMS and verify if needed
	- Initial condition: Initial.py; plot_ini.py (need initial data from cmems)
	- Nudging coefficients: nudge.py (check settings for different resolution)
	- Sponge layer: sponge.py (check settings for different resolution)
	- Boundary condition: Boundary.py (need boundary data from cmems)
	- Climatology: clim.py (need to do it after forcing files are done)
	
	8. Prepare ROMS
	- roms.in
	- Varinfo.yaml
	- Roms.h
	Then compile ROMS depending on settings
	Make sure model settings match each other

	9. Run parcel model with ice conditions (can be done without ROMS run using the original forcing files)
	Parcel_ice.py
Parcel_ana.py
