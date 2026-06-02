# pyright: strict, reportUnknownMemberType=false

from typing import Callable, Mapping, cast

import xarray as xr
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime


type UnitConverter = Callable[[xr.DataArray], xr.DataArray]

def linear_converter(offset: float, scale: float) -> UnitConverter:
    def converter(da: xr.DataArray):
        return da * scale + offset
    return converter

def null_converter(da: xr.DataArray):
    return da

def convert_units(da: xr.DataArray, new_units: str | None) -> xr.DataArray:
    original_units = da.attrs.get('units')
    if new_units == original_units:
        pass
    elif original_units is None:
        # We know that new_units is not None, or they would have been ==
        raise Exception("Can't convert to {new_units} because I don't know the original units")
    else:  # original_units is not None
        if new_units is None:
            pass  # leave quantities as is, drop the units attr
        else:  # new_units is not None
            try:
                converter = UNIT_CONVERSIONS[(original_units, new_units)]
            except KeyError as e:
                e.add_note(f"Don't know how to convert from {original_units} to {new_units}")
                raise
            da = converter(da)
    return da

UNIT_CONVERSIONS: Mapping[tuple[str, str], UnitConverter] = {
    ('degrees_north', 'degree_north'): null_converter,
    ('degrees_east', 'degree_east'): null_converter,
    ('degreeC', 'degree_Celsius'): null_converter,
    ('K', 'degree_Celsius'): linear_converter(-273.15, 1),
    ('Kelvin', 'degree_Celsius'): linear_converter(-273.15, 1),
    ('gpm', 'm'): null_converter,
    ('kg/m2', 'mm'): null_converter,  # density of water is 1000kg/m3
    ('kg m**-2 s**-1', 'mm/day'): linear_converter(0, 1000 * 60 * 60 * 24 / 1000),
    ('kg m-2 s-1', 'mm/day'): linear_converter(0, 1000 * 60 * 60 * 24 / 1000),
    ('kg m^-2 s^-1', 'mm/day'): linear_converter(0, 1000 * 60 * 60 * 24 / 1000),
    ('mm/s', 'mm/day'): linear_converter(0, 60 * 60 * 24),
    ('m/s', 'mm/day'): linear_converter(0, 1000 * 60 * 60 * 24),
    ('m s**-1', 'mm/day'): linear_converter(0, 1000 * 60 * 60 * 24),
    ('hPa', 'Pa'): linear_converter(0, 100),
    # Volumetric latent heat of vaporization: 2453 MJ m-3
    ('watt/m^2', 'mm/day'): linear_converter(0, 1000 * 60 * 60 * 24 / 2453e6),
}

# Keys are the conventional names used by pydap-icechunk.
# They can not be changed and serve as well as identifiers for the different objects.
# Values are what are going to be shown on, and served, by the pydap server
# They can be changed to the taste of the Catalog administrator
NAMES = {
    'S': 'S',
    'L': 'L',
    'Y': 'Y',
    'X': 'X',
    'M': 'M',
    'P': 'P',
    'target': 'target',
    # NB target_bnds should be named after target
    'target_bnds': 'target_bnds',
    # Additionally, these keys are also the icechunk variables names.
    'pr': 'pr',
    'prcp': 'prcp',
    'tas': 'tas',
    'tasmax': 'tasmax',
    'tasmin': 'tasmin',
    'tmin': 'tmin',
    'tmax': 'tmax',
    't2m': 't2m',
    'sst': 'sst',
    'psl': 'psl',
    'uas': 'uas',
    'vas': 'vas',
    'z': 'z',
    'zg': 'zg',
    'evap': 'evap',
    'runoff': 'runoff',
    'sm': 'sm',
}
# Change the dictionary values 
# should you different time encoding throughout your system
DATETIME_ENCODING = {
    'units': 'hours since 1960-01-01',
    'calendar': 'standard',
    'dtype': 'int32',
}
TIMEDELTA_ENCODING = {
    'units': 'hours',
    'dtype': 'int32',
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
        # units: implicitly months, but that's not allowed by CF
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
        # No units. From
        # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html#dimensionless-units
        # "A variable with no units attribute is assumed to be dimensionless."
    },
    'P': {
        'long_name': 'Pressure level',
        'standard_name': 'air_pressure',
        'units': 'Pa',
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
        'bounds': NAMES['target_bnds'],
    },
    'target_bnds': {},
    'pr': {
        'long_name': 'Total precipitation',
        'standard_name': 'lwe_precipitation_rate',
        'units': 'mm/day',
    },
    'prcp': {
        'long_name': 'Total precipitation',
        'standard_name': 'lwe_precipitation_rate',
        'units': 'mm/day',
    },
    'tas': {
        'long_name': 'Air temperature',
        'standard_name': 'air_temperature',
        'units': 'degree_Celsius',
    },
    'tasmax': {
        'long_name': 'Maximum air temperature',
        'standard_name': 'air_temperature',
        'units': 'degree_Celsius',
    },
    'tasmin': {
        'long_name': 'Minimum air temperature',
        'standard_name': 'air_temperature',
        'units': 'degree_Celsius',
    },
    't2m': {
        'long_name': 'Air temperature',
        'standard_name': 'air_temperature',
        'units': 'degree_Celsius',
    },
    'tmax': {
        'long_name': 'Maximum air temperature',
        'standard_name': 'air_temperature',
        'units': 'degree_Celsius',
    },
    'tmin': {
        'long_name': 'Minimum air temperature',
        'standard_name': 'air_temperature',
        'units': 'degree_Celsius',
    },
    'sst': {
        'long_name': 'Sea surface temperature',
        'standard_name': 'sea_surface_temperature',
        'units': 'degree_Celsius',
    },
    'psl': {
        'long_name': 'Pressure at sea level',
        'standard_name': 'air_pressure_at_sea_level',
        'units': 'Pa',
    },
    'uas': {
        'long_name': '10m eastward wind',
        'standard_name': 'eastward_wind',
        'units': 'm s-1',
    },
    'vas': {
        'long_name': '10m northward wind',
        'standard_name': 'northward_wind',
        'units': 'm s-1',
    },
    'z': {
        'long_name': 'Geopotential height',
        'standard_name': 'geopotential_height',
        'units': 'm',
    },
    'zg': {
        'long_name': 'Geopotential height',
        'standard_name': 'geopotential_height',
        'units': 'm',
    },
    'evap': {
        'long_name': 'Canopy evaporation',
        'standard_name': 'water_evaporation_flux_from_canopy',
        'units': 'mm/day',
    },
    'runoff': {
        'long_name': 'Runoff',
        'standard_name': 'runoff_flux',
    },
    'sm': {
        'long_name': 'Soil moisture',
        'standard_name': 'soil_moisture_content',
    },
}

DS_STANDARD_ATTRS = {
    'Conventions': 'CF-1.13',
}


def standardize_ds(
    ds: xr.Dataset,
    lead_is_month: bool,
):
    coords = {
        name: standardize_da(name, da)
        for name, da in coords_of(ds).items()
    }
    data_vars = {
        name: standardize_da(name, da)
        for name, da in data_vars_of(ds).items()
    }
    for name, da in data_vars.items():
        # xarray knows which variables are aux coords,
        # but cf_encoder fails to encode that information,
        # so we do it ourselves.
        aux_coords: list[str] = []
        if NAMES['S'] in da.dims:
            aux_coords.extend([NAMES['target'], NAMES['target_bnds']])
        if aux_coords:
            da.attrs['coordinates'] = ' '.join(aux_coords)

        if lead_is_month and name == 'prcp':
            target_length = (
                ds['target_bnds'].isel(nbound=1, drop=True)
                - ds['target_bnds'].isel(nbound=0, drop=True)
            ).dt.days
            data_vars[name] = da * target_length.variable
            data_vars[name].attrs['units'] = 'mm'


    # Top ds attrs standardization
    # cfgrib generates a large number of mostly useless attributes. Until
    # we get around to identifying the interesting ones, drop them all.
    attrs = {
        k: v for k, v in ds.attrs.items()
        if not k.startswith('GRIB')
    }
    # Keep the provider's remaining dataset-level attributes, and add our own.
    attrs.update(DS_STANDARD_ATTRS)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return ds


def standardize_da(name: str, da: xr.DataArray):
    da = da.copy()
    new_units = STANDARD_ATTRS[name].get('units')
    da = convert_units(da, new_units)

    da.attrs = dict(STANDARD_ATTRS[name])

    # Override provider's encoding for datetimes
    if np.issubdtype(da, np.datetime64):
        da.encoding = dict(DATETIME_ENCODING)
    elif np.issubdtype(da, np.timedelta64):
        da.encoding = dict(TIMEDELTA_ENCODING)

    # convert units

    return da


def S_L_to_target(S: xr.DataArray, L: xr.DataArray):

    target_bnds = xr.DataArray(
        data=[
            np.transpose([
                # This cast is only valid when using standard calendar
                cast(pd.DatetimeIndex, xr.date_range(
                    start=s.item(),
                    # TODO may not be wise to simply rely on len(L)
                    periods=len(L),
                    freq='MS',
                )),
                cast(pd.DatetimeIndex, xr.date_range(
                    start=datetime.datetime(
                        s.dt.year.item(), s.dt.month.item(), s.dt.day.item()
                    ) + relativedelta(months=1),
                    # TODO may not be wise to simply rely on len(L)
                    periods=len(L),
                    freq='MS',
                )),
            ])
            for s in S
        ],
        dims=['S', 'L', 'nbound'],
        coords={'S': S, 'L': L},
    )
    return target_bnds.isel(nbound=0, drop=True), target_bnds


def catalog(
    ds: xr.Dataset,
    original_names: Mapping[str, str],
    # to define if not define in-file (or to overwrite what's defined in-file)
    # Definition must be a key of UNITS_CONVERSIONS
    units: Mapping[str, str] | None = None,
    lead_is_month: bool = False,
):
    # Renaming std and dropping non-std
    for name in (set(vars_of(ds)) | set(sizes_of(ds))):
        if name in original_names:
            if name != original_names[name]: 
                ds = ds.rename({name: NAMES[original_names[name]]})
        elif name not in NAMES:
            if name in ds.variables:
                ds = ds.drop_vars(name)
    # Checking everything is standard:
    non_std_names = [
        name for name in (set(ds.variables) | set(ds.sizes)) if name not in NAMES
    ]
    if len(non_std_names) > 0:
        raise Exception(f'non standard {*non_std_names,} in dataset')

    # Add missing units
    if units is not None:
        for name, units_str in units.items():
            if name in ds:
                ds[name].attrs['units'] = units_str

    if lead_is_month:
        # Set lead times
        leads = np.arange(ds.sizes[NAMES['L']])
        # Set target
        targets, targets_bnds = S_L_to_target(
            ds[NAMES['S']], ds[NAMES['L']]
        )
        targets_bnds = targets_bnds.data
    else:
        targets = ds[NAMES["target"]]
        leads = (
            targets.isel({NAMES['S']: 0}, drop=True)
            - ds[NAMES['S']].isel({NAMES['S']: 0}, drop=True)
        )
        targets_bnds = np.stack([targets, targets + np.timedelta64(1, 'D')], axis=2)
    ds = ds.assign_coords({
        NAMES['L']: leads,
        NAMES['target']: targets,
        NAMES['target_bnds']: (
            (NAMES['S'], NAMES['L'], 'nbound'), targets_bnds
        ),
    })

    # Convert units, standardize attrs
    ds = standardize_ds(ds, lead_is_month)

    return ds


def coords_of(ds: xr.Dataset | xr.DataArray):
    """A version of the coords attribute with a better type hint"""
    assert all(isinstance(k, str) for k in ds.coords)
    return cast(Mapping[str, xr.DataArray], ds.coords)


def vars_of(ds: xr.Dataset):
    assert all(isinstance(k, str) for k in ds.variables)
    return cast(Mapping[str, xr.Variable], ds.variables)


def data_vars_of(ds: xr.Dataset):
    assert all(isinstance(k, str) for k in ds.data_vars)
    return cast(Mapping[str, xr.DataArray], ds.data_vars)


def sizes_of(ds: xr.Dataset | xr.DataArray):
    assert all(isinstance(k, str) for k in ds.sizes)
    return cast(Mapping[str, int], ds.sizes)
