import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'NMME/COLA-RSMAS/CESM1/{varname}')
    ds = ds.rename({
        'IRIDL_time':  'S',
        'TIME': 'L',
        'LAT': 'Y',
        'LON': 'X',
        'TIME_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    original_names = {
        'prec': 'PRECIP_1X1',
        'tref': 'T_REF_1X1',
        'sst': 'SST_1X1',
    }
    ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    del ds[varname].attrs['lon']
    del ds[varname].attrs['lat']
    ds = ds.assign_coords(L=('L', range(len(ds['L']))))
    return ds
