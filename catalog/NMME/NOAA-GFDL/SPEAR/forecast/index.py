import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/NOAA-GFDL/SPEAR/forecast/{varname}', decode_times=False
    )
    ds = ds.rename({
        'IRIDL_time':  'S',
        'TIME': 'L',
        # sst's spatial grids have same value but names LAT and LON
        # so these work only for prec / tref
        'LAT1': 'Y',
        'LON1': 'X',
        'TIME_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    original_names = {
        'prec': 'PRECIP_1X1',
        'tref': 'T_REF_1X1',
        'sst': 'SST_1X1',
    }
    ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    # units attribute is missing, however, 
    # how do we want to write things that depend on varname?
    ds = ds.assign_coords(L=('L', range(len(ds['L']))))
    return ds
