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
    orig_name = next(iter(ds.data_vars))
    ds = ds.rename({orig_name: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    del ds[varname].attrs['lon']
    del ds[varname].attrs['lat']
    # so the first lead of a forecast issued in April 2026 is targetting April 2026
    # L=0 could be acceptable except that L=0 would tend to mean that these are the 
    # initial conditions, which they are not, this target is a monthly aggregate of 
    # the first steps (whatever they were originally) of the model run.
    # So I would rather go with either 0.5 or 1.
    # 0.5 may be confusing users to think in half-months while
    # 1 could be ambiguous meaning this month or next month
    # even though target information is there so in any case, ambiguity or confusion
    # can be lifted.
    ds[varname] = ds[varname].assign_coords(L=('L', range(len(ds['L']))))
    return ds
