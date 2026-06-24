from collections.abc import Mapping
import itertools
from typing import Any

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
    ds_new = xr.open_zarr(session.store)
    ds_old = (
        preprocess.open_icechunk(icechunk_info.relpath)
        .isel(IRIDL_time=slice(0,2))
    )
    xarray.testing.assert_equal(ds_old, ds_new)
    assert ds_old.attrs == ds_new.attrs
    for var in itertools.chain(ds_old.coords, ds_old.data_vars):
        assert ds_old[var].attrs == ds_new[var].attrs
        assert_encodings_equal(ds_old[var].encoding, ds_new[var].encoding)

def assert_encodings_equal(d1: Mapping[str, Any], d2: Mapping[str, Any]):
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        v1 = d1[k]
        v2 = d2[k]
        # nan == nan is False, so we need a special case for it.
        if isinstance(v1, float) and np.isnan(v1):
            assert np.isnan(v2), f'different values for {k}: {v1} != {v2}'
        else:
            assert v1 == v2, f'different values for {k}: {v1} != {v2}'
