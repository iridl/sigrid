from collections.abc import Mapping
import itertools
from typing import Any, cast

import icechunk
import numpy as np
import pytest
import xarray as xr
import xarray.testing

from sigrid import preprocess

config = preprocess.config_from_env()
raw_cat = preprocess.FileSetCatalog(
    config.raw_catalog_root,
    config.orig_root
)

to_fix = (
    'SEAS51',
    'CanSIPS',
    'SPEAR',
    'CCSM4',
    'CESM1',
    'SubC/EMC/GEFSv12/forecast/psl',
    'SubC/EMC/GEFSv12/forecast/tas',
    'SubC/EMC/GEFSv12/forecast/tasmax',
    'SubC/EMC/GEFSv12/forecast/tasmin',
    'SubC/EMC/GEFSv12/forecast/ua_10m',
    'SubC/EMC/GEFSv12/forecast/va_10m',
    'SubC/EMC/GEFSv12/forecast/zg_500',
    'ERSSTv5',
)

def marks(var_path: str):
    if any(partial_name in var_path for partial_name in to_fix):
        return pytest.mark.xfail(strict=True)
    else:
        return ()

@pytest.mark.parametrize(
        'var_path',
        [
            pytest.param(var_path, marks=marks(var_path))
            for var_path in raw_cat.list_all()
        ],
)
def test_it(var_path: str):
    descriptor, icechunk_info = raw_cat.get_entry(var_path)
    storage = icechunk.in_memory_storage()
    repo_config = icechunk.RepositoryConfig.default()
    repo = icechunk.Repository.create(storage, repo_config)
    session = repo.writable_session('main')
    new_count = preprocess.update(
        # Looks like in_memory_storage doesn't handle parallel writes, so we
        # have to either use parallel=0 or write to disk.
        # TODO is parallel with local storage faster?
        # TODO fix in_memory_storage?
        session, descriptor, limit=2, first=None, parallel=0
    )
    assert new_count
    session.commit(f'update from {descriptor.dir}')
    ds_new = remove_irrelevant(xr.open_zarr(session.store))
    ds_old = remove_irrelevant(
        preprocess.open_icechunk(icechunk_info.relpath)
        .isel(IRIDL_time=slice(0,2))
    )
    xarray.testing.assert_identical(ds_new, ds_old)
    encodings_old = {name: replace_nans(v.encoding) for name, v in variables_of(ds_old).items()}
    encodings_new = {name: replace_nans(v.encoding) for name, v in variables_of(ds_new).items()}
    assert encodings_new == encodings_old

class NaNSentinel:
    def __repr__(self):
        return "NaN"
NAN = NaNSentinel()

def replace_nans(d: dict[str, Any]) -> dict[str, Any]:
    return {k: replace_nan(v) for k, v in d.items()}

def replace_nan(v):
    if isinstance(v, float) and np.isnan(v):
        return NAN
    return v

def variables_of(ds: xr.Dataset):
    assert all(isinstance(k, str) for k in ds.variables)
    return cast(Mapping[str, xr.Variable], ds.variables)

def remove_irrelevant(ds: xr.Dataset):
    # cfgrib creates 'history' and 'source' attributes that change on every run.
    # The former contains a timestamp, and the latter depends on $CWD.
    attrs = ds.attrs.copy()
    for attr in ('history', 'source'):
        attrs.pop(attr, None)

    return xr.Dataset(ds.data_vars, ds.coords, attrs)
