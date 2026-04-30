import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'NMME/ECCC/CanSIPS-IC4/forecast/{varname}')
    ds = ds.rename({
        'IRIDL_time': 'S',
        'step': 'L',
        'latitude': 'Y',
        'longitude': 'X',
        'number': "M",
    })
    # Need to create target but need to agree on L first
    orig_name = next(iter(ds.data_vars))
    ds = ds.rename({orig_name: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    ds[varname] = ds[varname].assign_coords(L=('L', range(len(ds['L']))))
    return ds
