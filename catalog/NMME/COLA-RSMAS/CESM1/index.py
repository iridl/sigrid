import xarray as xr
from pint import UnitRegistry

import pydap_icechunk


VARS_UNITS = {
    'prec': 'mm/day',
    'tref': 'degree_Celsius',
    'sst': 'degree_Celsius',
}

ORIGINAL_UNITS = {
    'Kelvin': 'kelvin',
}


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/COLA-RSMAS/CESM1/{varname}', decode_times=False
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
    ureg = UnitRegistry()
    # This is an option but I can't have the program read the file
    #ureg.load_definitions('src/original_untis_def.txt')
    for orig_unit, valid_unit in ORIGINAL_UNITS.items():
        ureg.define(f'{orig_unit} = {valid_unit}')
    Q_ = ureg.Quantity
    ds[varname].data = Q_(
        ds[varname].data, ds[varname].attrs['units']
    ).to(VARS_UNITS[varname])
    ds[varname].attrs['units'] = VARS_UNITS[varname]
    return ds

def list_vars():
    return ['prec', 'tref', 'sst']
