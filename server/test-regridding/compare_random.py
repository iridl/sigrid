import time

import numpy as np
import xarray as xr
import xesmf

from sigrid.harmonize.grids import ensure_bounds, regrid_conservative

# A randomly-generated case where agreement between sigrid and xesmf isn't as
# close as it is in compare_ingrid.py.


def main():
    nlat = 190
    nlon = 384

    nodes, _ = np.polynomial.legendre.leggauss(nlat)
    lats = np.degrees(np.arcsin(nodes))
    lons = np.linspace(0, 360, nlon, endpoint=False)

    rng = np.random.default_rng(0)
    ds_in = ensure_bounds(xr.DataArray(
        data=rng.random((len(lats), len(lons))),
        coords={
            'Y': ('Y', lats, {'units': 'degree_north'}),
            'X': ('X', lons, {'units': 'degree_east'}),
        },
        name='PRATE'
    ), gaussian=True)

    # original coordinates: S = 2011-03-12T00, L=0
    # resampled coordinates: S = 2011-04-01, M=1, L=0

    ds_out = ensure_bounds(xr.Dataset({
        'Y': np.arange(-90, 91),
        'X': np.arange(0, 360),
    }))

    start = time.time()
    regridded1 = regrid_conservative(ds_in, ds_out)['PRATE'].load()
    print(time.time() - start)

    start = time.time()
    # Setting units allows xesmf to identify the right coords
    ds_out['Y'].attrs['units'] = 'degree_north'
    ds_out['X'].attrs['units'] = 'degree_east'
    regridder = xesmf.Regridder(ds_in, ds_out, method="conservative", periodic=True)
    regridded2 = regridder(ds_in)['PRATE'].load()
    print(time.time() - start)

    diff = np.abs(regridded1 - regridded2)
    idx = diff.argmax(...)
    print('Max difference occurs at', regridded1[idx].coords)
    print('regridded1 value', regridded1[idx].item())
    print('regridded2 value', regridded2[idx].item())
    print('difference', diff[idx].item())

    print()
    print('original mean  ', weighted_mean(next(iter(ds_in.data_vars.values())), ds_in['Y_bnds']))
    print('regridded1 mean', weighted_mean(regridded1, ds_out['Y_bnds']))
    print('regridded2 mean', weighted_mean(regridded2, ds_out['Y_bnds']))


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

