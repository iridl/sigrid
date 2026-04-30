import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'NMME/NOAA-GFDL/SPEAR/forecast/{varname}')
    ds = ds.rename({
        'IRIDL_time':  'S',
        'TIME': 'L',
        # sst's spatial grids have same value but names LAT and LON
        # so these work only for prec / tref
        'LAT1': 'Y',
        'LON1': 'X',
        'TIME_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    # This returns random results very most likely because varname is not the only 
    # var in the ds: there is also a TIME_bnds one.
    # I am unsure how next/iter work but it sometimes a hit, sometimes an error, 
    # sometimes TIME_bnds that is renamed (not leading to an error)
    orig_name = next(iter(ds.data_vars))
    print(orig_name)
    ds = ds.rename({orig_name: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    # units attribute is missing, however, 
    # how do we want to write things that depend on varname?
    ds[varname] = ds[varname].assign_coords(L=('L', range(len(ds['L']))))
    return ds
