# pyright: strict, reportUnknownMemberType=false

import functools
import importlib.util
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Self, cast

import icechunk
import numpy as np
import xarray as xr

# TODO reconcile pydap config vs catalog config
ICECHUNK_ROOT = Path(os.environ['ICECHUNK_ROOT'])
COOKED_CATALOG_ROOT = Path(os.environ['COOKED_CATALOG_ROOT'])


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


@dataclass
class DatasetConfig:
    da_attrs: Mapping[str, Mapping[str, str]]
    ds_attrs: Mapping[str, str]
    encodings: Mapping[str, Mapping[str, str]]
    bare_dims: Iterable[str]
    lead_is_month: bool


def rename(ds: xr.Dataset, mapping: Mapping[str, str]):
    ds = ds.rename({
        provider_name: standard_name
        for provider_name, standard_name in mapping.items()
        if provider_name in set(ds.variables) | set(ds.dims)
        and provider_name != standard_name
    })

    return ds


def standardize(ds: xr.Dataset, config: DatasetConfig):
    ds = drop_non_std(ds, config.da_attrs, config.bare_dims)
    ds = convert_units(ds, config.da_attrs)
    ds = add_target(ds, config.lead_is_month)
    ds = standardize_attrs(
        ds,
        da_attrs=config.da_attrs,
        ds_attrs=config.ds_attrs,
        encodings=config.encodings,
    )
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


def convert_units(ds: xr.Dataset, standard_attrs: Mapping[str, Mapping[str, str]]):
    data_vars = {
        name: convert_units_da(da, standard_attrs[name].get('units'))
        for name, da in data_vars_of(ds).items()
    }
    return xr.Dataset(data_vars, coords=ds.coords, attrs=ds.attrs)

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


def standardize_attrs(
    ds: xr.Dataset,
    da_attrs: Mapping[str, Mapping[str, str]],
    ds_attrs: Mapping[str, str],
    encodings: Mapping[str, Mapping[str, str]],
):
    coords = {
        name: standardize_attrs_da(da, da_attrs, encodings)
        for name, da in coords_of(ds).items()
    }
    data_vars = {
        name: standardize_attrs_da(da, da_attrs, encodings)
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
    attrs.update(ds_attrs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return ds


def standardize_attrs_da(
        da: xr.DataArray,
        attrs: Mapping[str, Mapping[str, str]],
        encodings: Mapping[str, Mapping[str, str]],
):
    assert isinstance(da.name, str)
    # Wipe out provider's attrs and replace with standard ones
    da.attrs = dict(attrs[da.name])

    if da.name in encodings:
        da.encoding = dict(encodings[da.name])

    return da


def add_target(ds: xr.Dataset, lead_is_month: bool):
    if lead_is_month:
        # Set lead times
        leads = range(ds.sizes[Coords.L])
        ds = ds.assign_coords({Coords.L: leads})
        # Set target
        targets, targets_bnds = S_L_to_target_monthly(
            ds[Coords.S], leads
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

def S_L_to_target_monthly(S: xr.DataArray, l_values: Sequence[int]):
    target_values = (
        S.values.astype('datetime64[M]')[:, np.newaxis] +
        np.array(l_values).astype('timedelta64[M]')
    )
    target_bnds_values = (
        target_values[:, :, np.newaxis] +
        np.arange(2).astype('timedelta64[M]')
    )
    target_bnds = xr.DataArray(
        data=target_bnds_values,
        dims=[Coords.S, Coords.L, Coords.nbound],
        coords={Coords.S: S, Coords.L: l_values},
    )

    return target_bnds.isel({Coords.nbound: 0}, drop=True), target_bnds


def monthly_total(ds: xr.Dataset, vars: Iterable[str]):
    for var in vars:
        if var in ds:
            da = ds[var]
            units = da.attrs.get('units')
            # Not sure if this .endswith test is robust enough, e.g. might there be
            # a space between the slash and the d? If this turns out not to work,
            # maybe it's time to bring in udunits.
            if not (units and units.endswith('/day')):
                raise Exception(f'Expected a rate per day but found {units}')
            target_length = (
                    ds[Coords.target_bnds].isel({Coords.nbound: 1}, drop=True)
                    - ds[Coords.target_bnds].isel({Coords.nbound: 0}, drop=True)
                ).dt.days
            da = da * target_length
            da.attrs['units'] = units[:-len('/day')]
            ds[var] = da
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
        self.hidden = (COOKED_CATALOG_ROOT / catalog_path / 'hidden').exists()
        self._module = None

    @property
    def subdatasets(self) -> dict[str, Self]:
        return {
            d.name: self.__class__(str(d.relative_to(COOKED_CATALOG_ROOT)), self)
            for d in (COOKED_CATALOG_ROOT / self.catalog_path).iterdir()
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
        index_path = COOKED_CATALOG_ROOT / self.catalog_path / 'index.py'
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
