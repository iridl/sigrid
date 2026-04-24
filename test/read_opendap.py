import xarray as xr

ds = xr.open_dataset('http://localhost:8001/NMME/COLA-RSMAS/CCSM4/prec', engine='netcdf4')
p = ds['prec']
print(p)
# May starts for JJA season over Ethiopia from 1990 to 2009
p = p.sel(S=slice('1990-01-01', '2009-12-01'))
p = p.sel(
    S=p['S'].dt.month == 5,
    L=slice(1,3),
    Y=slice(3, 15),
    X=slice(32, 48),
).mean('M').mean('L')
print(p)

# ds = xr.open_dataset('http://localhost:8001/NMME/CCSM4.0/prec.icechunk?mean(PREC,M)', engine='netcdf4')
# print(ds)
