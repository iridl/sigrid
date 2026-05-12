import xarray as xr
from pint import UnitRegistry


import pydap_icechunk


VARS_UNITS = {
    'prec': 'mm/day',
    'tref': 'degree_Celsius',
    'sst': 'degree_Celsius',
}

ORIGINAL_UNITS = {}


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(
        f'NMME/ECCC/CanSIPS-IC4/forecast/{varname}', decode_times=False
    )
    # Some varnames have scalar coordinates
    ds = ds.drop_vars([name for name, coord in ds.coords.items() if coord.dims == ()])
    ds = ds.drop_vars(["valid_time_expanded", "time"])
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
    #ds = ds.assign_coords(target=pydap_icechunk.S_L_to_target(ds['S'], ds['L']))
    ureg = UnitRegistry()
    # This is an option but I can't have the program read the file
    #ureg.load_definitions('src/original_untis_def.txt')
    for orig_unit, valid_unit in ORIGINAL_UNITS.items():
        ureg.define(f'{orig_unit} = {valid_unit}')
    ureg.define('water_density = 1000 kg/m^3')
    Q_ = ureg.Quantity
    ds[varname].data = Q_(
        ds[varname].data, ds[varname].attrs['units']
    )
    if ds[varname].data.check('[mass] / [length] ** 2 / [time]'):
        ds[varname].data = ds[varname].data / ureg.water_density
    ds[varname].data = ds[varname].data.to(VARS_UNITS[varname])
    ds[varname].attrs['units'] = VARS_UNITS[varname]
    #ds = pydap_icechunk.encode_time(ds)
    # Force into coords
    ds[varname].attrs['coordinates'] = "target"
    return ds

def list_vars():
    return ['prec', 'tref', 'sst']
