# This is the index.py file for a multi-model ensemble called my-MME.
# Its transform function ensures that comparable variables from
# different models are represented in a consistent way. It assumes
# that variable names have already been normalized by transform
# functions lower in the hierarchy.

from cataloging import DatasetConfig, standardize

da_attrs = {
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
    'prcp': {
        'long_name': 'Total precipitation',
        'standard_name': 'lwe_precipitation_rate',
        'units': 'mm/day',
    },
}

encodings = {
    'S': {
        'units': 'hours since 1960-01-01',
        'calendar': 'standard',
        'dtype': 'int32',
    },
}

config = DatasetConfig(
    da_attrs=da_attrs,
    ds_attrs = {
        'Conventions': 'CF-1.13',
    },
    encodings=encodings,
    bare_dims=[],
    lead_is_month=True,
)

def transform(ds):
    ds = standardize(ds, config)
    return ds
