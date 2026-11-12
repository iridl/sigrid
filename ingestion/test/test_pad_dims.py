import numpy as np
import pytest
import xarray as xr

from sigrid import preprocess


def make_opener(pad_dims):
    return preprocess._FileOpener(
        original_time_dim=None,
        backend_kwargs=None,
        data_vars=None,
        drop_coords=['time'],
        expand_coords=[],
        aux_coords=[],
        pad_dims=pad_dims,
    )


@pytest.mark.parametrize('n_records', [64, 91, 92])
def test_pad_dims(tmp_path, n_records):
    # A file with n_records lead records, padded or trimmed to 91.
    data = np.arange(n_records * 2 * 3, dtype=np.float32).reshape(n_records, 2, 3)
    path = tmp_path / 'f.nc'
    xr.Dataset(
        {'x': (('time', 'lat', 'lon'), data)},
        coords={'time': np.arange(n_records), 'lat': [0.0, 1.0], 'lon': [0.0, 1.0, 2.0]},
    ).to_netcdf(path)

    coords = preprocess.FileCoords(np.datetime64('2026-10-03T00:00'), None, None, None)
    with make_opener({'time': 91}).open(path, coords) as ds:
        x = ds['x'].isel(IRIDL_time=0)
        assert x.sizes['time'] == 91
        assert x.dtype == np.float32
        kept = min(n_records, 91)
        np.testing.assert_array_equal(x.isel(time=slice(0, kept)).values, data[:kept])
        assert x.isel(time=slice(kept, None)).isnull().all()
