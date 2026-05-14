import xarray as xr
import numpy as np
from collections import namedtuple

import pydap_icechunk


UNITS_CONVERTER = namedtuple('UNITS_CONVERTER', 'offset scale name')
UNITS_CONVERSIONS = {
    # This may sound silly but is needed as things are written now
    # it does nothing other than redefining units attr to the same value
    'degree_Celsius': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'degreeC': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'K': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'Kelvin': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'kg m**-2 s**-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'kg m-2 s-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'mm/day': UNITS_CONVERTER(0, 1, 'mm/day'),
    'mm/s': UNITS_CONVERTER(0, 60 * 60 * 24, 'mm/day'),
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
TIME_ENCODING = {
    'units': 'hours since 1960-01-01',
    'calendar': 'standard',
}

# Note the time vars' units and calendar are dealt with separately
# as well as variables units conversion and definition
STANDARD_ATTRS = {
    'S': {
        'long_name': 'Forecast start time',
        'standard_name': 'forecast_reference_time',
    },
    'L': {
        'long_name': 'Lead',
        'standard_name': 'forecast_period',
    },
    'Y': {
        'long_name': 'Latitude',
        'standard_name': 'latitude',
        'units': 'degree_north',
    },
    'X': {
        'long_name': 'Longitude',
        'standard_name': 'longitude',
        'units': 'degree_east',
    },
    'M': {
        'long_name': 'Ensemble member',
        'standard_name': 'realization',
        'units': 'unitless',
    },
    'target': {
        'long_name': 'Forecast target period',
        # CF Conventions standard names table says:
        # forecast_reference_time: The forecast reference time in NWP
        # is the "data time", the time of the analysis from which the
        # forecast was made. It is not the time for which the forecast
        # is valid; the standard name of time should be used for that time.
        # https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
        'standard_name': 'time',
    },
    'prec': {
        'long_name': 'Total precipitation',
        'standard_name': 'lwe_precipitation_rate',
    },
    'tref': {
        'long_name': 'Reference temperature',
        'standard_name': 'air_temperature',
    },
    'sst': {
        'long_name': 'Sea surface temperature',
        'standard_name': 'sea_surface_temperature',
    }
}

DS_STANDARD_ATTRS = {
    'Conventions': 'CF-1.13',
}


def standardize(
    ds,
    varname,
    units=None,
    time_enconding=TIME_ENCODING,
):
    # Convert varname units and apply standard attrs
    if units is not None:
        # GFDL data has simply no units attribute
        ds[varname].attrs['units'] = units[varname]
    original_units = ds[varname].attrs['units']
    if not (
        UNITS_CONVERSIONS[original_units].scale == 1
        and UNITS_CONVERSIONS[original_units].offset == 0
    ):
        ds[varname] = (
            ds[varname]
            * UNITS_CONVERSIONS[original_units].scale
            + UNITS_CONVERSIONS[original_units].offset
        )
    ds[varname].attrs = dict(
        STANDARD_ATTRS[varname],
        coordinates=COORDS_NAMES['target'],
        units=UNITS_CONVERSIONS[original_units].name,
    )
    # Encode time coords and apply standard attrs
    time_coords = [
        coord for coord in ds.coords
        if ds[coord].dtype in ['datetime64[ns]', 'timedelta64[ns]']
    ]
    for tc in time_coords:
        data, time_units, calendar = xr.coding.times.encode_cf_datetime(
            ds[tc],
            TIME_ENCODING['units'],
            TIME_ENCODING['calendar'],
            dtype=np.dtype("int64"),
        )
        ds = ds.assign_coords({tc: (ds[tc].dims, data)})
        ds[tc].attrs = dict(
            STANDARD_ATTRS[tc],    
            units=time_units,
            calendar=calendar,
        )
    # Apply standard attrs to other coords
    other_coords = [coord for coord in ds.coords if coord not in time_coords]
    for coord in other_coords:
        ds[coord].attrs = dict(STANDARD_ATTRS[coord])
    # Top ds attrs standardization
    # cfgrib generates a large number of mostly useless attributes. Until
    # we get around to identifying the interesting ones, drop them all.
    ds.attrs = {
        k: v for k, v in ds.attrs.items()
        if not k.startswith('GRIB')
    }
    # Keep the provider's remaining dataset-level attributes, and add our own.
    ds.attrs.update(DS_STANDARD_ATTRS)
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
    # to define if not define in-file
    units=None,
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
    # Squeeze dims of size 1
    for dim in ds.dims:
        if ds.sizes[dim] == 1 :
            ds = ds.squeeze(dim, drop=True)
    # Renaming
    # TODO This lot has become wild. Revisit
    for var in original_names:
        # Need to cover list of original names 
        # when different vars have different coords names, see e.g. SPEAR
        if not isinstance(original_names[var], list):
            orig_names = [original_names[var]]
        else :
            orig_names = original_names[var]
        for orig_name in orig_names:
            if (
                # accomodates when different vars have diffrent coord names
                orig_name in ds.dims
                # only conventional coords
                and var in COORDS_NAMES
                # rename can't rename with same name
                and orig_name != COORDS_NAMES[var]
            ):
                ds = ds.rename({orig_name: COORDS_NAMES[var]})
            if (
                # only conventional vars
                var in VARS_NAMES
                # accommodates same catalog catalogs all variables
                and VARS_NAMES[var] == varname
                # rename can't rename with same name
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
    # Drop vars not standard
    # This is for SPEAR TIME_bnds
    # TODO all bounds
    ds = ds.drop_vars(
        [
            name
            for name in ds.keys()
            if name not in VARS_NAMES.values()
        ]
    )
    # Deleting buggy attributes
    for attr in list(ds.attrs):
        if str(ds.attrs[attr]).find('"') != -1 :
            del ds.attrs[attr]
    # Deleting dummy attributes
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
    # Convert units, encode time and standardize attrs
    ds = standardize(ds, varname, units=units)

    return ds
