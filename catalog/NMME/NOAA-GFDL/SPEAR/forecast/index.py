import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/NOAA-GFDL/SPEAR/forecast/{varname}', decode_times=False
    )
    original_coords_names = {
        'prec': {'Y': 'LAT1', 'X': 'LON1'}, 
        'tref': {'Y': 'LAT1', 'X': 'LON1'},
        'sst': {'Y': 'LAT', 'X': 'LON'},
    }
    ds = ds.rename({
        'IRIDL_time':  'S',
        'TIME': 'L',
        original_coords_names[varname]['Y']: 'Y',
        original_coords_names[varname]['X']: 'X',
        'TIME_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    original_names = {
        'prec': 'PRECIP_1X1',
        'tref': 'T_REF_1X1',
        'sst': 'SST_1X1',
    }
    if varname in original_names:
        ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    ds = ds.assign_coords(L=('L', range(len(ds['L']))))
    units = {
        'prec': 'mm/s',
        'tref': 'K',
        'sst': 'degree_Celsius',
    }
    ds[varname].attrs['units'] = units[varname]
    return ds
