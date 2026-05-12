import xarray as xr
from collections import namedtuple

import pydap_icechunk


# These as vocation to move to cataloging.py
UNITS_CONVERTER = namedtuple('UNITS_CONVERTER', 'offset scale name')
UNITS_CONVERSIONS = {
    'm/s': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24, 'mm/day'),
    'Kelvin': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'degreeC': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
}


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/COLA-RSMAS/CCSM4/{varname}', decode_times=False
    )
    ds = ds.rename({
        'IRIDL_time':  'S',
        'TIME': 'L',
        'LAT': 'Y',
        'LON': 'X',
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
    del ds[varname].attrs['lon']
    del ds[varname].attrs['lat']
    ds = ds.assign_coords(L=('L', range(ds.sizes['L'])))
    original_units = [
        name
        for name, conv in UNITS_CONVERSIONS.items()
        if name == ds[varname].attrs['units']
    ][0]
    if not (
        UNITS_CONVERSIONS[original_units].scale == 1
        and UNITS_CONVERSIONS[original_units].offset == 0
    ):
        ds[varname] = (
            ds[varname]
            * UNITS_CONVERSIONS[original_units].scale
            + UNITS_CONVERSIONS[original_units].offset
        )
    ds[varname].attrs['units'] = UNITS_CONVERSIONS[original_units].name
    return ds

def list_vars():
    return ['prec', 'tref', 'sst']
