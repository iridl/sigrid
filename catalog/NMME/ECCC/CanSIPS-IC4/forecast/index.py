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
    original_names = {
        'prec': 'prate',
        'tref': 'avg_2t',
        'sst': 'avg_sst',
    }
    ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    del ds.attrs['history'] # temporary until a pydap fix
    ds[varname] = ds[varname].assign_coords(L=('L', range(len(ds['L']))))
    return ds
