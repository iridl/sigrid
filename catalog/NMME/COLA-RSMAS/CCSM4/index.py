import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'NMME/COLA-RSMAS/CCSM4/{varname}')
    ds = ds.rename({
        'IRIDL_time':  'S',
        'TIME': 'L',
        'LAT': 'Y',
        'LON': 'X',
        'TIME_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    orig_name = next(iter(ds.data_vars))
    ds = ds.rename({orig_name: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    del ds[varname].attrs['lon']
    del ds[varname].attrs['lat']
    ds[varname] = ds[varname].assign_coords(L=('L', range(len(ds['L']))))
    return ds
