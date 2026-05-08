import xarray as xr
import numpy as np
import pandas as pd

import pydap_icechunk



def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/ECCC/CanSIPS-IC4/forecast/{varname}'#, decode_times=False
    )
    # Some varnames have scalar coordinates
    ds = ds.drop_vars([name for name, coord in ds.coords.items() if coord.dims == ()])
    ds = ds.rename({
        'IRIDL_time': 'S',
        'step': 'L',
        'latitude': 'Y',
        'longitude': 'X',
        'number': "M",
    })
    original_names = {
        'prec': 'prate',
        'tref': 'avg_2t',
        'sst': 'avg_sst',
    }
    if varname in original_names:
        ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    del ds.attrs['history'] # temporary until a pydap fix
    ds = ds.assign_coords(L=('L', range(ds.sizes['L'])))
    ds = ds.assign_coords(target=pydap_icechunk.S_L_to_target(ds['S'], ds['L']))
    ds = pydap_icechunk.encode_time(ds)
    # Force into coords
    ds[varname].attrs['coordinates'] = "valid_time_expanded time target"
    return ds

def list_vars():
    return ['prec', 'tref', 'sst']
