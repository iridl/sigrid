import numpy as np
import xarray as xr

from sigrid.preprocess import FileCoords, _FileOpener


def _write(path, name):
    xr.Dataset(
        {name: (('time', 'lat'), np.ones((2, 3), dtype='float32'))},
        coords={'lat': [0.0, 1.0, 2.0]},
    ).to_netcdf(path)


def _opener(rename_vars):
    return _FileOpener(
        original_time_dim=None,
        backend_kwargs=None,
        data_vars=None,
        drop_coords=None,
        expand_coords=[],
        aux_coords=[],
        rename_vars=rename_vars,
    )


def test_rename_vars_old_and_new_names(tmp_path):
    # Files from before and after a rename end up with the same variable name.
    old, new = tmp_path / 'old.nc', tmp_path / 'new.nc'
    _write(old, 'zg_200')
    _write(new, 'zg')
    opener = _opener({'zg_200': 'zg'})
    coords = FileCoords(np.datetime64('2020-01-01'), None, None, None)
    for path in (old, new):
        with opener.open(path, coords) as ds:
            assert list(ds.data_vars) == ['zg']


def test_rename_vars_default_is_no_change(tmp_path):
    path = tmp_path / 'f.nc'
    _write(path, 'zg_200')
    coords = FileCoords(np.datetime64('2020-01-01'), None, None, None)
    with _opener(None).open(path, coords) as ds:
        assert list(ds.data_vars) == ['zg_200']
