# pyright: strict, reportUnknownMemberType=false

from dataclasses import dataclass
from typing import Mapping, cast

import xarray as xr
import xarray.conventions
import xarray.coding.times
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime

import pydap_icechunk


@dataclass
class UNITS_CONVERTER:
    offset: float
    scale: float
    name: str


UNITS_CONVERSIONS = {
    # This may sound silly but is needed as things are written now
    # it does nothing other than redefining units attr to the same value
    'degree_north': UNITS_CONVERTER(0, 1, 'degree_north'),
    'degrees_north': UNITS_CONVERTER(0, 1, 'degree_north'),
    'degree_east': UNITS_CONVERTER(0, 1, 'degree_east'),
    'degrees_east': UNITS_CONVERTER(0, 1, 'degree_east'),
    'unitless': UNITS_CONVERTER(0, 1, 'unitless'),
    '1': UNITS_CONVERTER(0, 1, 'unitless'),
    'degree_Celsius': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'degreeC': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'K': UNITS_CONVERTER(-273.15, 1, 'degree_Celsius'),
    'Kelvin': UNITS_CONVERTER(-273.15, 1, 'degree_Celsius'),
    'gpm': UNITS_CONVERTER(0, 1, 'm'),
    # Let's pray there won't be prcp in m s-1
    'm s-1': UNITS_CONVERTER(0, 1, 'm s-1'),
    'kg m**-2 s**-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'kg m-2 s-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'kg m^-2 s^-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'mm/day': UNITS_CONVERTER(0, 1, 'mm/day'),
    'mm/s': UNITS_CONVERTER(0, 60 * 60 * 24, 'mm/day'),
    'm/s': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24, 'mm/day'),
    'm s**-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24, 'mm/day'),
    'Pa': UNITS_CONVERTER(0, 1, 'Pa'),
    'hPa': UNITS_CONVERTER(0, 100, 'Pa'),
}

# Keys are the conventional names used by pydap-icechunk.
# They can not be changed and serve as well as identifiers for the different objects.
# Values are what are going to be shown on, and served, by the pydap server
# They can be changed to the taste of the Catalog administrator
NAMES = {
    'S': 'S',
    'L': 'L',
    'L_bnds': 'L_bnds',
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
    't2m': 't2m',
    'sst': 'sst',
    'psl': 'psl',
    'uas': 'uas',
    'vas': 'vas',
    'zg': 'zg',
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
    'calendar': 'standard',
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
        'bounds': NAMES['L_bnds'],
    },
    'L_bnds': {},
    'Y': {
        'long_name': 'Latitude',
        'standard_name': 'latitude',
    },
    'X': {
        'long_name': 'Longitude',
        'standard_name': 'longitude',
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
    },
    'prcp': {
        'long_name': 'Total precipitation',
        'standard_name': 'lwe_precipitation_rate',
    },
    'tas': {
        'long_name': 'Air temperature',
        'standard_name': 'air_temperature',
    },
    'tasmax': {
        'long_name': 'Maximum air temperature',
        'standard_name': 'air_temperature',
    },
    'tasmin': {
        'long_name': 'Minimum air temperature',
        'standard_name': 'air_temperature',
    },
    't2m': {
        'long_name': 'Air temperature',
        'standard_name': 'air_temperature',
    },
    'sst': {
        'long_name': 'Sea surface temperature',
        'standard_name': 'sea_surface_temperature',
    },
    'psl': {
        'long_name': 'Pressure at sea level',
        'standard_name': 'air_pressure_at_sea_level',
    },
    'uas': {
        'long_name': '10m eastward wind',
        'standard_name': 'eastward_wind',
    },
    'vas': {
        'long_name': '10m northward wind',
        'standard_name': 'northward_wind',
    },
    'zg': {
        'long_name': 'Geopotential height',
        'standard_name': 'geopotential_height',
    },
}

DS_STANDARD_ATTRS = {
    'Conventions': 'CF-1.13',
}


def standardize(
    ds: xr.Dataset,
    lead_is_month: bool,
    units: Mapping[str, str] | None = None,
):
    if units is None:
        units = {}

    vars: Mapping[str, xr.Variable] = {}

    for name, var in vars_of(ds).items():
        # save the original attributes, then replace them
        # with standard ones
        original_attrs = var.attrs
        var.attrs = dict(STANDARD_ATTRS[name])

        # Override provider's encoding for datetimes
        if np.issubdtype(var, np.datetime64):
            var.encoding = dict(DATETIME_ENCODING)

        if name in ds.data_vars:
            # xarray knows which variables are aux coords,
            # but cf_encoder fails to encode that information,
            # so we do it ourselves.
            aux_coords: list[str] = []
            if NAMES['L'] in var.dims:
                aux_coords.append(NAMES['L_bnds'])
                if NAMES['S'] in var.dims:
                    aux_coords.extend([NAMES['target'], NAMES['target_bnds']])
            if aux_coords:
                var.attrs['coordinates'] = ' '.join(aux_coords)

            # convert units
            # TODO will need to generalize to coords, e.g. Z in Pa vs hPa
            if name in units:
                # provide units explicitly if provider didn't (e.g. GFDL)
                original_units = units[name]
            else:
                original_units = original_attrs.get('units')
            if original_units is not None:
                conversion = UNITS_CONVERSIONS[original_units]
                if not (conversion.scale == 1 and conversion.offset == 0):
                    var = var * conversion.scale + conversion.offset
                var.attrs['units'] = conversion.name

            if lead_is_month and name == 'prcp':
                target_length = (
                    ds['target_bnds'].isel(nbound=1, drop=True)
                    - ds['target_bnds'].isel(nbound=0, drop=True)
                ).dt.days
                var = var * target_length.variable
                var.attrs['units'] = 'mm'

        vars[name] = var


    # Top ds attrs standardization
    # cfgrib generates a large number of mostly useless attributes. Until
    # we get around to identifying the interesting ones, drop them all.
    attrs = {
        k: v for k, v in ds.attrs.items()
        if not k.startswith('GRIB')
    }
    # Keep the provider's remaining dataset-level attributes, and add our own.
    attrs.update(DS_STANDARD_ATTRS)

    timedelta_vars = {
        name: var
        for name, var in vars.items()
        if np.issubdtype(var, np.timedelta64)
    }
    for name, var in timedelta_vars.items():
        encoded_var, encoded_units = xarray.coding.times.encode_cf_timedelta(var, TIMEDELTA_ENCODING['units'], TIMEDELTA_ENCODING['dtype'])
        encoded_var = xr.DataArray(
            data=encoded_var,
            dims=var.dims,
            attrs=var.attrs,
        )
        encoded_var.attrs['units'] = encoded_units
        timedelta_vars[name] = encoded_var
    other_vars = {
        name: var
        for name, var in vars.items()
        if ~np.issubdtype(var, np.timedelta64)
    }
    other_vars, attrs = cast(
        tuple[Mapping[str, xr.Variable], Mapping[str, str]],
        xarray.conventions.cf_encoder(other_vars, attrs)
    )
    vars = other_vars | timedelta_vars
    ds = xr.Dataset(
        data_vars={k: v for k, v in vars.items() if k in ds.data_vars},
        coords={k: v for k, v in vars.items() if k in ds.coords},
        attrs=attrs,
    )

    return ds


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
    varname: str,
    varpath: str,
    original_names: Mapping[str, str],
    # to define if not define in-file (or to overwrite what's defined in-file)
    # Definition must be a key of UNITS_CONVERSIONS
    units: Mapping[str, str] | None = None,
    lead_is_month: bool = False,
    ):
    icechunk_var = [key for key, value in NAMES.items() if value == varname][0]
    ds = pydap_icechunk.open_icechunk(
        f'{varpath}/{icechunk_var}',
    )
    # Some varnames have scalar coordinates that break pydap
    ds = ds.drop_vars(
        [name for name, coord in coords_of(ds).items() if coord.dims == ()]
    )
    # Squeeze dims of size 1
    for dim in ds.dims:
        if ds.sizes[dim] == 1 :
            ds = ds.squeeze(dim, drop=True)
    # Renaming std and dropping non-std
    for name in (set(ds.variables) | set(ds.sizes)):
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
    # Deleting buggy attributes
    for attr in list(ds.attrs):
        if str(ds.attrs[attr]).find('"') != -1 :
            del ds.attrs[attr]

    if lead_is_month:
        # Set lead times
        leads = np.arange(ds.sizes[COORDS_NAMES['L']])
        leads_bnds = np.stack([leads, leads + 1], axis=1)
        # Set target
        targets, targets_bnds = S_L_to_target(
            ds[COORDS_NAMES['S']], ds[COORDS_NAMES['L']]
        )
        targets_bnds = targets_bnds.data
    else:
        targets = ds[COORDS_NAMES["target"]]
        leads = (
            targets.isel({COORDS_NAMES['S']: 0}, drop=True)
            - ds[COORDS_NAMES['S']].isel({COORDS_NAMES['S']: 0}, drop=True)
        )
        leads_bnds = np.stack([leads, leads + np.timedelta64(1, 'D')], axis=1)
        targets_bnds = np.stack([targets, targets + np.timedelta64(1, 'D')], axis=2)
    ds = ds.assign_coords({
        COORDS_NAMES['L']: leads,
        COORDS_NAMES['L_bnds']: ((COORDS_NAMES['L'], 'nbound'), leads_bnds),
        COORDS_NAMES['target']: targets,
        COORDS_NAMES['target_bnds']: (
            (COORDS_NAMES['S'], COORDS_NAMES['L'], 'nbound'), targets_bnds
        ),
    })

    # Convert units, do cf-encoding and standardize attrs
    ds = standardize(ds, lead_is_month, units)

    return ds


def coords_of(ds: xr.Dataset | xr.DataArray):
    """A version of the coords attribute with a better type hint"""
    assert all(isinstance(k, str) for k in ds.coords)
    return cast(Mapping[str, xr.DataArray], ds.coords)


def vars_of(ds: xr.Dataset):
    assert all(isinstance(k, str) for k in ds.variables)
    return cast(Mapping[str, xr.Variable], ds.variables)