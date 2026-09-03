import time
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf

from sigrid.harmonize.grids import ensure_bounds, regrid_conservative

# Compares sigrid's conservative regridding method with analogous methods from
# ingrid and xesmf. Not included in the main test suite because xesmf isn't
# present in the normal test environment.
#
# To run it, cd to this directory and then: pixi run python compare.py
#
# Observations:
# - sigrid agrees with xesmf to four significant figures at worst, but with
#   ingrid the descrepancy at a single cell can be more than 4x (agreement to 0
#   significant figures). For this reason, I had to give up on doing comparison
#   tests for regridded variables.
# - sigrid and xesmf preserve the global mean to six significant figures; ingrid
#   only to two.
# - xesmf takes more than 100x longer than sigrid.



def main():
    s = '1%20Mar%202011'
    m = 16
    l = 6.5
    input_file = f'https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.NMME_REALTIME_ENSEMBLE/.FLXF/.surface/.PRATE/S/({s})/VALUE/S/removeGRID/M/{m}/VALUE/M/removeGRID/L/{l}/VALUE/L/removeGRID/dods'
    #input_file = Path(__file__).parent / 'input.nc'
    ds_in = xr.open_dataset(input_file, decode_times=False).load()

    ds_in = ensure_bounds(ds_in, gaussian=True)


    ds_out = ensure_bounds(xr.Dataset({
        'Y': ('Y', np.arange(-90, 91), {'units': 'degree_north'}),
        'X': ('X', np.arange(0, 360), {'units': 'degree_east'}),
    }))

    start = time.time()
    regridded_sigrid = regrid_conservative(ds_in, ds_out)['PRATE'].load()
    print('sigrid took', time.time() - start)

    # Not timing ingrid because we can't separate data loading time from
    # computation time.
    regridded_ingrid = xr.open_dataset(f'https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.MONTHLY/.prec/L/0.5/VALUE/M/1.0/VALUE/S/last/VALUE/SOURCES/.NOAA/.NCEP/.EMC/.CFSv2/.NMME_REALTIME_ENSEMBLE/.FLXF/.surface/.PRATE/S/({s})/VALUE/M/{m}/VALUE/L/{l}/VALUE/%5BX/Y%5D/regridAverage/S/removeGRID/M/removeGRID/L/removeGRID/dods', decode_times=False)['PRATE'].load()

    start = time.time()
    regridder = xesmf.Regridder(ds_in, ds_out, method="conservative", periodic=True)
    regridded_xesmf = regridder(ds_in)['PRATE'].load()
    print('xesmf took', time.time() - start)

    print()
    print('original mean', weighted_mean(next(iter(ds_in.data_vars.values())), ds_in['Y_bnds']))
    print('sigrid mean  ', weighted_mean(regridded_sigrid, ds_out['Y_bnds']))
    print('ingrid mean  ', weighted_mean(regridded_ingrid, ds_out['Y_bnds']))
    print('xesmf mean   ', weighted_mean(regridded_xesmf, ds_out['Y_bnds']))

    diff = np.abs(regridded_sigrid - regridded_ingrid)
    idx = diff.argmax(...)
    print()
    print('Max difference between sigrid and ingrid occurs at', regridded_sigrid[idx].coords)
    print('sigrid value', regridded_sigrid[idx].item())
    print('ingrid value', regridded_ingrid[idx].item())
    print('difference', diff[idx].item())

    diff = np.abs(regridded_sigrid - regridded_xesmf)
    idx = diff.argmax(...)
    print()
    print('Max difference between sigrid and xesmf occurs at', regridded_sigrid[idx].coords)
    print('sigrid value', regridded_sigrid[idx].item())
    print('xesmf value ', regridded_xesmf[idx].item())
    print('difference', diff[idx].item())



def weighted_mean(da, lat_bounds, lat_dim='Y', lon_dim='X'):
    sin_lat_bounds =   np.sin(np.radians(lat_bounds))
    lat_weights = sin_lat_bounds.isel(nbound=1) - sin_lat_bounds.isel(nbound=0)
    norm = np.sum(lat_weights.values)
    assert np.isclose(np.abs(norm), 2, atol=1e-15, rtol=0)
    return (
        (da.mean(lon_dim) * lat_weights).sum(lat_dim) / norm
    ).item()


if __name__ == '__main__':
    main()