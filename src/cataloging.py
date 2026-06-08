# pyright: strict, reportUnknownMemberType=false

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import functools
import importlib.util
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Self, TypedDict, cast

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

STANDARD_UNIT_CONVERSIONS: Mapping[tuple[str, str], UnitConverter] = MappingProxyType({
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
})

def convert_units_da(
        da: xr.DataArray,
        new_units: str | None,
        conversions: Mapping[tuple[str, str], UnitConverter] = STANDARD_UNIT_CONVERSIONS,
) -> xr.DataArray:
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
                converter = conversions[(original_units, new_units)]
            except KeyError as e:
                e.add_note(f"Don't know how to convert from {original_units} to {new_units}")
                raise
            da = converter(da)
    return da


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


class DatetimeEncoding(TypedDict):
    units: str
    calendar: str
    dtype: str


class TimedeltaEncoding(TypedDict):
    units: str
    dtype: str


@dataclass
class DatasetConfig:
    standard_attrs: Mapping[str, Mapping[str, str]]
    toplevel_standard_attrs: Mapping[str, str]
    bare_dims: Iterable[str]
    lead_is_month: bool
    datetime_encoding: DatetimeEncoding
    timedelta_encoding: TimedeltaEncoding


def standardize(ds: xr.Dataset, config: DatasetConfig):
    ds = drop_non_std(ds, config.standard_attrs, config.bare_dims)            # list is at ensemble level. Logic anywhere.
    ds = convert_units(ds, config.standard_attrs)        # mapping at ensemble level, logic anywhere
    ds = add_target(ds, config.lead_is_month)  # ensemble level. Shouldn't be included in standard function. But has to precede standardize_ds.
    ds = standardize_attrs(
        ds,
        standard_attrs=config.standard_attrs,
        toplevel_standard_attrs=config.toplevel_standard_attrs,
        datetime_encoding=config.datetime_encoding,
        timedelta_encoding=config.timedelta_encoding,
    )  # data at ensemble or site level, logic anywhere but has to be after add_target, and probably after convert_units.
    ds = seasonal_total(ds)           # ensemble level

    return ds


def seasonal_total(ds: xr.Dataset):
    if 'prcp' in ds:
        da = ds['prcp']
        target_length = (
                ds[Coords.target_bnds].isel({Coords.nbound: 1}, drop=True)
                - ds[Coords.target_bnds].isel({Coords.nbound: 0}, drop=True)
            ).dt.days
        da = da * target_length.variable
        da.attrs['units'] = 'mm'
        ds['prcp'] = da
    return ds


def standardize_attrs(
    ds: xr.Dataset,
    standard_attrs: Mapping[str, Mapping[str, str]],
    toplevel_standard_attrs: Mapping[str, str],
    datetime_encoding: DatetimeEncoding,
    timedelta_encoding: TimedeltaEncoding,
):
    coords = {
        name: standardize_da(name, da, standard_attrs, datetime_encoding, timedelta_encoding)
        for name, da in coords_of(ds).items()
    }
    data_vars = {
        name: standardize_da(name, da, standard_attrs, datetime_encoding, timedelta_encoding)
        for name, da in data_vars_of(ds).items()
    }

    # Top ds attrs standardization
    # cfgrib generates a large number of mostly useless attributes. Until
    # we get around to identifying the interesting ones, drop them all.
    attrs = {
        k: v for k, v in ds.attrs.items()
        if not k.startswith('GRIB')
    }
    # Keep the provider's remaining dataset-level attributes, and add our own.
    attrs.update(toplevel_standard_attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return ds


def standardize_da(
        name: str,
        da: xr.DataArray,
        standard_attrs: Mapping[str, Mapping[str, str]],
        datetime_encoding: DatetimeEncoding,
        timedelta_encoding: TimedeltaEncoding,
):
    # Wipe out provider's attrs and replace with standard ones
    da.attrs = dict(standard_attrs[name])

    # Override provider's encoding for datetimes
    if np.issubdtype(da, np.datetime64):
        da.encoding = dict(datetime_encoding)
    elif np.issubdtype(da, np.timedelta64):
        da.encoding = dict(timedelta_encoding)

    return da

def convert_units(ds: xr.Dataset, standard_attrs: Mapping[str, Mapping[str, str]]):
    data_vars = {
        name: convert_units_da(da, standard_attrs[name].get('units'))
        for name, da in data_vars_of(ds).items()
    }
    return xr.Dataset(data_vars, coords=ds.coords, attrs=ds.attrs)


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


def add_target(ds: xr.Dataset, lead_is_month: bool):
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

    return ds

def drop_non_std(ds: xr.Dataset, standard_attrs: Mapping[str, Mapping[str, str]], bare_dims: Iterable[str]):
    ds = ds.drop_vars([
        name for name in ds.variables
        if name not in standard_attrs
    ])

    non_std_dims = [
        name for name in ds.dims if name not in set(bare_dims) | set(standard_attrs)
    ]
    if len(non_std_dims) > 0:
        raise Exception(f'non standard dims {*non_std_dims,} in dataset')
    return ds

def rename(ds: xr.Dataset, mapping: Mapping[str, str]):
    ds = ds.rename({
        provider_name: standard_name
        for provider_name, standard_name in mapping.items()
        if provider_name in set(ds.variables) | set(ds.dims)
        and provider_name != standard_name
    })

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
