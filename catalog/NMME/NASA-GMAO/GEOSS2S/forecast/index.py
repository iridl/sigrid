import xarray as xr
import pint_xarray
import pint

import pydap_icechunk


ORIGINAL_UNITS = {
    'degrees_east': 'degree',
    'degrees_north': 'degree',
}


def parse_units(units):
    return units.replace('-', ' ** -')


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/NASA-GMAO/GEOSS2S/forecast/{varname}', decode_times=False
    )
    ds = ds.squeeze(dim="level", drop=True)
    ds = ds.rename({
        'IRIDL_time':  'S',
        'time': 'L',
        'latitude': 'Y',
        'longitude': 'X',
        'time_expanded': 'target',  # TODO convert target from noleap, or just drop and recreate it
    })
    original_names = {
        'prec': 'precip',
    }
    if varname in original_names:
        ds = ds.rename({original_names[varname]: varname})
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    ds = ds.assign_coords(L=('L', range(ds.sizes['L'])))
    ureg = pint.UnitRegistry(force_ndarray_like=True)
    for orig_unit, valid_unit in ORIGINAL_UNITS.items():
        ureg.define(f'{orig_unit} = {valid_unit}')
    ds[varname].attrs['units'] = parse_units(ds[varname].attrs['units'])
    ureg.define('water_density = 1 * kilogram / m ** 3')
    ds = ds.pint.quantify(unit_registry=ureg)
    if ds[varname].data.check('[mass] / [length] ** 2 / [time]'):
        ds[varname] = ds[varname] / ureg.water_density
    ds[varname] = ds[varname].pint.to('mm/day')
    ds = ds.pint.dequantify()
    return ds

def list_vars():
    return ['prec', 'tref', 'sst']
