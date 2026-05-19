# pyright: strict, reportUnknownMemberType=false

from dataclasses import dataclass
from typing import Mapping, cast

import xarray as xr
import xarray.conventions
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
    'degree_Celsius': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'degreeC': UNITS_CONVERTER(0, 1, 'degree_Celsius'),
    'K': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'Kelvin': UNITS_CONVERTER(-272.15, 1, 'degree_Celsius'),
    'kg m**-2 s**-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'kg m-2 s-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24 / 1000, 'mm/day'),
    'mm/day': UNITS_CONVERTER(0, 1, 'mm/day'),
    'mm/s': UNITS_CONVERTER(0, 60 * 60 * 24, 'mm/day'),
    'm/s': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24, 'mm/day'),
    'm s**-1': UNITS_CONVERTER(0, 1000 * 60 * 60 * 24, 'mm/day'),
}

# Keys are the conventional names used by pydap-icechunk.
# They can not be changed and serve as well as identifiers for the different objects.
# Values are what are going to be shown on, and served, by the pydap server
# They can be changed to the taste of the Catalog administrator
COORDS_NAMES = {
    'S': 'S',
    'L': 'L',
    'L_bnds': 'L_bnds',
    'Y': 'Y',
    'X': 'X',
    'M': 'M',
    'target': 'target',
    # NB target_bnds should be named after target
    'target_bnds': 'target_bnds'
}
# Same as above, additionally, keys are the icechunk variables names.
VARS_NAMES = {
    'prcp': 'prcp',
    't2m': 't2m',
    'sst': 'sst',
}
# Change the dictionary values 
# should you different time encoding throughout your system
DATETIME_ENCODING = {
    'units': 'hours since 1960-01-01',
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
        'bounds': COORDS_NAMES['L_bnds'],
    },
    'L_bnds': {},
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
    'target': {
        'long_name': 'Forecast target period',
        # CF Conventions standard names table says:
        # forecast_reference_time: The forecast reference time in NWP
        # is the "data time", the time of the analysis from which the
        # forecast was made. It is not the time for which the forecast
        # is valid; the standard name of time should be used for that time.
        # https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
        'standard_name': 'time',
        'bounds': COORDS_NAMES['target_bnds'],
    },
    'target_bnds': {},
    'prcp': {
        'long_name': 'Total precipitation',
        'standard_name': 'lwe_precipitation_rate',
    },
    't2m': {
        'long_name': 'Air temperature',
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
        dl_name = [
            key
            for key, value in dict(VARS_NAMES, **COORDS_NAMES).items()
            if value == name
        ][0]
        var.attrs = dict(STANDARD_ATTRS[dl_name])

        # Override provider's encoding for datetimes
        if np.issubdtype(var, np.datetime64):
            var.encoding = dict(DATETIME_ENCODING)

        if name in ds.data_vars:
            # xarray knows which variables are aux coords,
            # but cf_encoder fails to encode that information,
            # so we do it ourselves.
            aux_coords: list[str] = []
            if COORDS_NAMES['L'] in var.dims:
                aux_coords.append(COORDS_NAMES['L_bnds'])
                if COORDS_NAMES['S'] in var.dims:
                    aux_coords.extend([COORDS_NAMES['target'], COORDS_NAMES['target_bnds']])
            if aux_coords:
                var.attrs['coordinates'] = ' '.join(aux_coords)

            # convert units
            # TODO will need to generalize to coords, e.g. Z in Pa vs hPa
            if dl_name in units:
                # provide units explicitly if provider didn't (e.g. GFDL)
                original_units = units[dl_name]
            else:
                original_units = original_attrs.get('units')
            if original_units is not None:
                conversion = UNITS_CONVERSIONS[original_units]
                if not (conversion.scale == 1 and conversion.offset == 0):
                    var = var * conversion.scale + conversion.offset
                var.attrs['units'] = conversion.name

            if lead_is_month and name == VARS_NAMES['prcp']: #(or dl_name == 'prcp' :) )
                target_length = (
                    ds[COORDS_NAMES['target_bnds']].isel(nbound=1, drop=True)
                    - ds[COORDS_NAMES['target_bnds']].isel(nbound=0, drop=True)
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

    vars, attrs = cast(
        tuple[Mapping[str, xr.Variable], Mapping[str, str]],
        xarray.conventions.cf_encoder(vars, attrs)
    )

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
    icechunk_var = [key for key, value in VARS_NAMES.items() if value == varname][0]
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
            for name in ds.coords
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

    if lead_is_month:
        # Set lead times
        l = np.arange(ds.sizes[COORDS_NAMES['L']])
        l_bnds = np.stack([l, l+1], axis=1)
        ds = ds.assign_coords({
            COORDS_NAMES['L']: l,
            COORDS_NAMES['L_bnds']: ((COORDS_NAMES['L'], 'nbound'), l_bnds)
        })
        # Set target
        target, target_bnds = S_L_to_target(
            ds[COORDS_NAMES['S']], ds[COORDS_NAMES['L']]
        )
        ds = ds.assign_coords({
            COORDS_NAMES["target"]: target,
            COORDS_NAMES["target_bnds"]: target_bnds,
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