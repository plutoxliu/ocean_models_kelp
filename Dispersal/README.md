# Lagrangian ocean dispersal model using parcel
<br><br>
Parcel website: https://parcels-code.org/ <br>
Documentation: https://docs.parcels-code.org/en/latest/ <br>
<br><br>
Codes optimised for high resolution circumpolar simulations, using grids and forcing files generated for ROMS
<br>
- Generate spatially-varying horizontal diffusivity from Eddy Kinetic Energy (optional) <br>
  Kh_field.py
  <br>
- Simulation <br>
  parcel_run.py
  <br>
  Configuration at the start of the file, including release sites, simulation parameters, physics parameters (wind and ice)
  Constant diffusivity (backup diffusivity) can be set if Kh is not available
  <br>
- Post-simulation analysis optimised for Antarctic ciecumpolar simulation
  parcel_ana.py
  <br>
  - Analysis includes regional connectivity analysis
  - Seasonal patterns
  - Antarctic connectivity assessment
  - Static maps and animations in Antarctic Polar Stereographic projection
  <br>(NOTE: uses a lot of memory and might need further optimisation for larger simulations than 2M)

