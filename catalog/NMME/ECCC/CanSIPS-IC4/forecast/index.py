import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/ECCC/CanSIPS-IC4/forecast/{varname}', decode_times=False
    )
    # Some varnames have scalar coordinates
    ds = ds.drop_vars(["surface", "heightAboveGround"], errors="ignore")
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
    ds = ds.assign_coords(L=('L', range(len(ds['L']))))
    # Need to assign target
    return ds
