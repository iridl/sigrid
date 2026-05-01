import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/ECCC/CanSIPS-IC4/forecast/{varname}', decode_times=False
    )
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
    ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    del ds.attrs['history'] # temporary until a pydap fix
    ds = ds.assign_coords(L=('L', range(len(ds['L']))))
    target = xr.DataArray(
        data=[
            [
                datetime.datetime(
                    s.dt.year.item(), s.dt.month.item(), s.dt.day.item()
                ) + relativedelta(months=l)
                for l in ds["L"]
            ] for s in ds["S"]
        ],
        coords=dict(S=ds["S"], L=ds["L"]),
    )
    ds = ds.assign(target=target)
    print(ds)
    return ds
