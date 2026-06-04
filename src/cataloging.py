# pyright: strict, reportUnknownMemberType=false

from dataclasses import dataclass
from enum import StrEnum
import functools
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Self, TypedDict, cast

import icechunk
import xarray as xr
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime


# TODO reconcile pydap config vs catalog config
ICECHUNK_ROOT = Path(os.environ['PYDAP_ICECHUNK_PROCESSED_ROOT'])
CATALOG_ROOT = Path(os.environ['PYDAP_CATALOG_ROOT'])


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


class CaseSensitiveStrEnum(StrEnum):
    """Like StrEnum but it doesn't downcase the names"""
    @staticmethod
    def _generate_next_value_(name: str, start: Any, count: Any, last_values: Any) -> str:
        return name


Coords = CaseSensitiveStrEnum(
    'Coords',
    ['S', 'L', 'M', 'P', 'Y', 'X', 'target', 'target_bnds', 'nbound']
    # nbound isn't actually a coord, only a dim. But target isn't a dim, only
    # a coord, so we can't call this Dims either.
)


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

# Note the time var's units and calendar are dealt with separately
# as well as variables units conversion and definition
STANDARD_ATTRS: dict[str, dict[str, str]] = {
    Coords.S: {
        'long_name': 'Forecast start time',
        'standard_name': 'forecast_reference_time',
    },
    Coords.L: {
        'long_name': 'Lead',
        'standard_name': 'forecast_period',
        # units: implicitly months, but that's not allowed by CF
    },
    Coords.Y: {
        'long_name': 'Latitude',
        'standard_name': 'latitude',
        'units': 'degree_north',
    },
    Coords.X: {
        'long_name': 'Longitude',
        'standard_name': 'longitude',
        'units': 'degree_east',
    },
    Coords.M: {
        'long_name': 'Ensemble member',
        'standard_name': 'realization',
        # No units. From
        # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html#dimensionless-units
        # "A variable with no units attribute is assumed to be dimensionless."
    },
    Coords.P: {
        'long_name': 'Pressure level',
        'standard_name': 'air_pressure',
        'units': 'Pa',
    },
    Coords.target: {
        'long_name': 'Forecast target period',
        # CF Conventions standard names table says:
        # forecast_reference_time: The forecast reference time in NWP
        # is the "data time", the time of the analysis from which the
        # forecast was made. It is not the time for which the forecast
        # is valid; the standard name of time should be used for that time.
        # https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
        'standard_name': 'time',
        'bounds': Coords.target_bnds,
    },
    Coords.target_bnds: {},
    Coords.nbound: {},
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


class DatetimeEncoding(TypedDict):
    units: str
    calendar: str
    dtype: str


class TimedeltaEncoding(TypedDict):
    units: str
    dtype: str


@dataclass
class DatasetConfig:
    datetime_encoding: DatetimeEncoding
    timedelta_encoding: TimedeltaEncoding
    standard_attrs: Mapping[str, Mapping[str, str]]
    toplevel_standard_attrs: Mapping[str, str]


config = DatasetConfig(
    datetime_encoding=DatetimeEncoding(**DATETIME_ENCODING),
    timedelta_encoding=TimedeltaEncoding(**TIMEDELTA_ENCODING),
    standard_attrs=STANDARD_ATTRS,
    toplevel_standard_attrs=DS_STANDARD_ATTRS,
)


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
        if Coords.S in da.dims:
            aux_coords.extend([Coords.target, Coords.target_bnds])
        if aux_coords:
            da.attrs['coordinates'] = ' '.join(aux_coords)

        if lead_is_month and name == 'prcp':
            target_length = (
                ds[Coords.target_bnds].isel({Coords.nbound: 1}, drop=True)
                - ds[Coords.target_bnds].isel({Coords.nbound: 0}, drop=True)
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
    attrs.update(config.toplevel_standard_attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return ds


def standardize_da(name: str, da: xr.DataArray):
    da = da.copy()
    new_units = config.standard_attrs[name].get('units')
    da = convert_units(da, new_units)

    da.attrs = dict(config.standard_attrs[name])

    # Override provider's encoding for datetimes
    if np.issubdtype(da, np.datetime64):
        da.encoding = dict(config.datetime_encoding)
    elif np.issubdtype(da, np.timedelta64):
        da.encoding = dict(config.timedelta_encoding)

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
        dims=[Coords.S, Coords.L, Coords.nbound],
        coords={Coords.S: S, Coords.L: L},
    )
    return target_bnds.isel({Coords.nbound: 0}, drop=True), target_bnds


def catalog(
    ds: xr.Dataset,
    original_names: Mapping[str, str],
    lead_is_month: bool = False,
):
    # Renaming std and dropping non-std
    for name in (set(vars_of(ds)) | set(sizes_of(ds))):
        if name in original_names:
            if name != original_names[name]: 
                ds = ds.rename({name: original_names[name]})
        elif name not in config.standard_attrs:
            if name in ds.variables:
                ds = ds.drop_vars(name)
    # Checking everything is standard:
    non_std_names = [
        name for name in (set(ds.variables) | set(ds.sizes)) if name not in config.standard_attrs
    ]
    if len(non_std_names) > 0:
        raise Exception(f'non standard {*non_std_names,} in dataset')

    if lead_is_month:
        # Set lead times
        leads = np.arange(ds.sizes[Coords.L])
        # Set target
        targets, targets_bnds = S_L_to_target(
            ds[Coords.S], ds[Coords.L]
        )
        targets_bnds = targets_bnds.data
    else:
        targets = ds[Coords.target]
        leads = (
            targets.isel({Coords.S: 0}, drop=True)
            - ds[Coords.S].isel({Coords.S: 0}, drop=True)
        )
        targets_bnds = np.stack([targets, targets + np.timedelta64(1, 'D')], axis=2)
    ds = ds.assign_coords({
        Coords.L: leads,
        Coords.target: targets,
        Coords.target_bnds: (
            (Coords.S, Coords.L, Coords.nbound), targets_bnds
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


def open_icechunk(rel_path: str, decode_times: bool = True, drop_variables: Iterable[str] = ()):
    abspath = ICECHUNK_ROOT / rel_path
    storage = icechunk.local_filesystem_storage(str(abspath))
    # Workaround for https://github.com/earth-mover/icechunk/issues/2105
    if not icechunk.Repository.exists(storage):
        raise Exception(f'No repository exists at {abspath}')
    try:
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        ds = xr.open_zarr(
            session.store,
            zarr_format=3,
            decode_times=decode_times,
            drop_variables=drop_variables,
        )
        return ds
    except Exception as e:
        e.add_note(f'When trying to open {abspath}')
        raise


type Opener = Callable[[], xr.Dataset]


@dataclass
class DisplayDataset:
    subdatasets: Iterable[str]
    variables: Iterable[str]


class Catalog:
    def __init__(self):
        self.root_node = CatalogNode('', None)

    def open_variable(self, catalog_path: str) -> xr.Dataset | None:
        if not catalog_path.startswith('/'):
            raise Exception(f'Absolute catalog path was expected; got relative path {catalog_path}')
        if catalog_path.endswith('/'):
            return None  # Variable paths can't end in slash.
        node = self.root_node
        components = catalog_path[1:].split('/')
        for i, component in enumerate(components):
            sub = node.subdatasets.get(component)
            if sub is None:
                if i == len(components) - 1:
                    opener = node.variables.get(component)
                    if opener is None:
                        return None
                    return opener()
                else:
                    return None
            node = sub
        # If we reach here, we bottomed out at a Dataset, not a Variable.
        return None
    
    def open_dataset(self, catalog_path: str) -> DisplayDataset | None:
        if not catalog_path.startswith('/'):
            raise Exception(f'Absolute catalog path was expected; got relative path {catalog_path}')
        if not catalog_path.endswith('/'):
            return None  # Dataset paths must end in slash.
        node = self.root_node
        components = catalog_path[1:-1].split('/')
        if components == ['']:
            components = []
        for component in components:
            sub = node.subdatasets.get(component)
            if sub is None:
                return None
            node = sub
        return DisplayDataset(
            [name for name, sub in node.subdatasets.items() if not sub.hidden],
            list(node.variables)
        )


class CatalogNode:
    def __init__(self, catalog_path: str, parent: Self | None) -> None:
        self.catalog_path = catalog_path
        self.parent = parent
        self.hidden = (CATALOG_ROOT / catalog_path / 'hidden').exists()
        self._module = None

    @property
    def subdatasets(self) -> dict[str, Self]:
        return {
            d.name: self.__class__(str(d.relative_to(CATALOG_ROOT)), self)
            for d in (CATALOG_ROOT / self.catalog_path).iterdir()
            if d.is_dir() and d.name != '__pycache__'
        }

    @property
    def variables(self) -> dict[str, Opener]:
        if self.module is None or not hasattr(self.module, 'list_vars'):
            return {}
        
        var_names: Iterable[str]

        if hasattr(self.module, 'list_vars'):
            var_names = self.module.list_vars()
        else:
            var_names = []

        return {
            var: functools.partial(self.open_var, var)
            for var in var_names
        }

    @functools.cached_property
    def module(self):
        index_path = CATALOG_ROOT / self.catalog_path / 'index.py'
        if not index_path.exists():
            return None
        return load_index(index_path)

    def open_var(self, varname: str) -> xr.Dataset:
        if self.module is None or not hasattr(self.module, 'open'):
            # This should never happen. We only call _open with names that
            # come from variables()
            assert False
        ds = self.module.open(varname)
        ds = self.transform(ds)
        return ds
    
    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        if self.module is not None and hasattr(self.module, 'transform'):
            ds = self.module.transform(ds)
        if self.parent is not None:
            ds = self.parent.transform(ds)
        return ds


def load_index(file_path: Path):
    spec = importlib.util.spec_from_file_location('catalog', file_path)
    assert spec is not None  # we already checked that it exists
    module = importlib.util.module_from_spec(spec)
    # Pyright says the loader could be None, but I don't see how that
    # could happen.
    assert spec.loader
    spec.loader.exec_module(module)
    return module
