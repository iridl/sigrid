import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'NMME/NASA-GMAO/GEOSS2S/forecast/{varname}')
    #ds = ds.squeeze(dim="level", drop=True)
    ds = ds.rename({
        'IRIDL_time':  'S',
        'time': 'L',
        'latitude': 'Y',
        'longitude': 'X',
        'time_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    # has singleton dim level
    # is not incorrect but could be removed for homogenization with other NMME
    original_names = {
        'prec': 'precip',
        'tref': 'tref',
        'sst': 'sst',
    }
    ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    ds = ds.assign_coords(L=('L', range(len(ds['L']))))
    return ds
