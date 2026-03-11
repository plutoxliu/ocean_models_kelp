import netCDF4 as nc
import numpy as np

grd = nc.Dataset("ocean_grd_test.nc", "r+")

eta = grd.dimensions["eta_rho"].size
xi  = grd.dimensions["xi_rho"].size

if "visc_factor" not in grd.variables:
    grd.createVariable("visc_factor", "f8", ("eta_rho", "xi_rho"))
if "diff_factor" not in grd.variables:
    grd.createVariable("diff_factor", "f8", ("eta_rho", "xi_rho"))

nb = 5        # sponge width in grid points 10 for fine 5 for coarse
Fmax = 2.0     # max enhancement 4.0 for fine 2.0 for coarse

visc_factor = np.ones((eta, xi))
diff_factor = np.ones((eta, xi))

for j in range(eta - nb, eta):
    d = (eta - 1 - j) / nb       # 1 → 0
    taper = 0.5 * (1 + np.cos(np.pi * d))
    factor = 1 + (Fmax - 1) * taper

    visc_factor[j, :] = factor
    diff_factor[j, :] = factor

grd.variables["visc_factor"][:] = visc_factor
grd.variables["diff_factor"][:] = diff_factor

grd.close()

lat = nc.Dataset("ocean_grd_test.nc")["lat_rho"][:]
print(lat[-1, :].min(), lat[-1, :].max())
