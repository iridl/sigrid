import xarray as xr
import numpy as np
from collections import namedtuple

import pydap_icechunk


UNITS_CONVERTER = namedtuple('UNITS_CONVERTER', 'offset scale name')
UNITS_CONVERSIONS = {
    'degree_Celsius': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'degreeC': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'K': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'Kelvin': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'kg m**-2 s**-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'mm/day': UNITS_CONVERTER(0, 1, 'mm/day'),
    'm/s': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24, 'mm/day'),
}

# Keys are the conventional names used by pydap-icechunk.
# They can not be changed and serve as well as identifiers for the different objects.
# Values are what are going to be shown on, and served, by the pydap server
# They can be changed to the taste of the Catalog administrator
COORDS_NAMES = {
    'S': 'S',
    'L': 'L',
    'Y': 'Y',
    'X': 'X',
    'M': 'M',
    'target': 'target',
}
# Same as above, additionally, keys are the icechunk variables names.
VARS_NAMES = {
    'prec': 'prec',
    'tref': 'tref',
    'sst': 'sst',
}
# Change the dictionary values 
# should you different time encoding throughout your system
ENCODING = {
    'cf_units': 'hours since 1960-01-01',
    'calendar': 'standard',
}


def convert_units(dsvar, missing_units=None):
    if missing_units is not None:
        dsvar.attrs['units'] = missing_units[dsvar.name]
    #original_units = [
    #    name
    #    for name, conv in UNITS_CONVERSIONS.items()
    #    if name == dsvar.attrs['units']
    #][0]
    original_units = dsvar.attrs['units']
    if not (
        UNITS_CONVERSIONS[original_units].scale == 1
        and UNITS_CONVERSIONS[original_units].offset == 0
    ):
        dsvar = (
            dsvar
            * UNITS_CONVERSIONS[original_units].scale
            + UNITS_CONVERSIONS[original_units].offset
        )
    dsvar.attrs['units'] = UNITS_CONVERSIONS[original_units].name
    return dsvar


def encode_time(
    ds,
    cf_catalog=ENCODING['calendar'],
    cf_units=ENCODING['cf_units'],
):
    time_coords = [
        coord for coord in ds.coords
        if ds[coord].dtype in ['datetime64[ns]', 'timedelta64[ns]']
    ]
    for tc in time_coords:
        data, units, calendar = xr.coding.times.encode_cf_datetime(
            ds[tc], cf_units, cf_catalog, dtype=np.dtype("int64")
        )
        ds = ds.assign_coords({tc: (ds[tc].dims, data)})
        ds[tc].attrs['units'] = units
        ds[tc].attrs['calendar'] = calendar
    return ds


def S_L_to_target(S, L):
    return xr.DataArray(
        data=[
            xr.date_range(
                start=s.item(),
                # TODO may not be wise to simply rely on len(L)
                periods=len(L),
                freq='MS',
            )
            for s in S
        ],
        coords=dict(S=S, L=L),
        attrs={'long_name': 'target date'},
    )


def catalog(
    varname,
    varpath,
    original_names,
    missing_units=None,
    lead_is_month=False,
    ):
    icechunk_var = [key for key, value in VARS_NAMES.items() if value == varname][0]
    ds = pydap_icechunk.open_icechunk(
        f'{varpath}/{icechunk_var}',
    )
    # Some varnames have scalar coordinates that break pydap
    ds = ds.drop_vars(
        [name for name, coord in ds.coords.items() if coord.dims == ()]
    )
    # Squeeze coords of size 1
    for coord in ds.dims:
        if ds.sizes[coord] == 1 :
            ds.squeeze(coord, drop=True)
    # Renaming
    for var in original_names:
        if not isinstance(original_names[var], list):
            orig_names = [original_names[var]]
        else :
            orig_names = original_names[var]
        for orig_name in orig_names:
            if var in COORDS_NAMES and orig_name != COORDS_NAMES[var] :
                ds = ds.rename({orig_name: COORDS_NAMES[var]})
            if (
                var in VARS_NAMES
                and VARS_NAMES[var] == varname
                and orig_name != VARS_NAMES[var]
            ):
                ds = ds.rename({original_names[var]: varname})
    # Drop coords not standard
    ds = ds.drop_vars(
        [
            name
            for name, coord in ds.coords.items()
            if name not in COORDS_NAMES.values()
        ]
    )
    # Deleting buggy attributes
    for attr in list(ds.attrs):
        if str(ds.attrs[attr]).find('"') != -1 :
            del ds.attrs[attr]
    # Deleting dummy attributes
    dummy_attrs = ['lon', 'lat']
    for attr in dummy_attrs:
        if attr in ds.attrs:
            del ds.attrs[attr]
    if lead_is_month:
        # Set lead times
        ds = ds.assign_coords({
            COORDS_NAMES['L']: range(ds.sizes[COORDS_NAMES['L']])
        })
        # Set target
        ds = ds.assign_coords({
            COORDS_NAMES["target"]: S_L_to_target(
                ds[COORDS_NAMES['S']], ds[COORDS_NAMES['L']]
            )
        })
    # Convert varname units
    ds[varname] = convert_units(ds[varname], missing_units=missing_units)
    # Encode time
    ds = encode_time(ds)
    # Force into coords
    ds[varname].attrs['coordinates'] = COORDS_NAMES["target"]
    return ds
