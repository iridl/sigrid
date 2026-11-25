import numpy as np
import pytest
import xarray as xr

from sigrid import preprocess
 

def make_opener(dim_lengths):
    return preprocess._FileOpener(
        original_time_dim=None,
        backend_kwargs=None,
        data_vars=None,
        drop_coords=['time'],
        expand_coords=[],
        aux_coords=[],
        dim_lengths=dim_lengths,
    )


def write_file(tmp_path, n_records):
    data = np.arange(n_records * 2 * 3, dtype=np.float32).reshape(n_records, 2, 3)
    path = tmp_path / 'f.nc'
    xr.Dataset(
        {'x': (('time', 'lat', 'lon'), data)},
        coords={'time': np.arange(n_records), 'lat': [0.0, 1.0], 'lon': [0.0, 1.0, 2.0]},
    ).to_netcdf(path)
    return path, data


COORDS = preprocess.FileCoords(np.datetime64('2026-10-03T00:00'), None, None, None)


@pytest.mark.parametrize('n_records', [64, 91, 92])
def test_dim_lengths(tmp_path, n_records, capsys):
    # A file with n_records lead records, padded or trimmed to 91.
    path, data = write_file(tmp_path, n_records)
    with make_opener({'time': 91}).open(path, COORDS) as ds:
        x = ds['x'].isel(IRIDL_time=0)
        assert x.sizes['time'] == 91
        assert x.dtype == np.float32
        kept = min(n_records, 91)
        np.testing.assert_array_equal(x.isel(time=slice(0, kept)).values, data[:kept])
        assert x.isel(time=slice(kept, None)).isnull().all()

    out = capsys.readouterr().out
    if n_records > 91:
        # The warning must identify file, dimension and both lengths.
        assert 'WARNING' in out
        assert str(path) in out and '"time"' in out and '92' in out and '91' in out
    else:
        assert out == ''


@pytest.mark.parametrize('n_days', [28, 30, 31])
def test_dim_lengths_climatology_months(tmp_path, n_days, capsys):
    # hindcast-climatology: months of 28..31 days padded to 31; never trims.
    path, data = write_file(tmp_path, n_days)
    with make_opener({'time': 31}).open(path, COORDS) as ds:
        x = ds['x'].isel(IRIDL_time=0)
        assert x.sizes['time'] == 31
        np.testing.assert_array_equal(x.isel(time=slice(0, n_days)).values, data)
        assert x.isel(time=slice(n_days, None)).isnull().all()
    assert capsys.readouterr().out == ''


def test_dim_lengths_missing_dim_raises(tmp_path):
    path, _ = write_file(tmp_path, 64)
    with pytest.raises(Exception, match='no dimension "nope"'):
        make_opener({'nope': 91}).open(path, COORDS)


def test_dim_lengths_unset_changes_nothing(tmp_path):
    # Regression: without the option the length is left as in the file.
    path, data = write_file(tmp_path, 64)
    with make_opener({}).open(path, COORDS) as ds:
        x = ds['x'].isel(IRIDL_time=0)
        assert x.sizes['time'] == 64
        np.testing.assert_array_equal(x.values, data)


def test_dim_lengths_through_descriptor(tmp_path):
    # Covers the constructor wiring (rename), which _FileOpener tests don't.
    path, data = write_file(tmp_path, 64)
    descriptor = preprocess.FileSetDescriptor(
        name='x',
        dir=tmp_path,
        catalog_path=preprocess.URLPath('x'),
        pattern=r'f\.nc$',
        drop_coords=['time'],
        dim_lengths={'time': 91},
    )
    assert descriptor.opener.dim_lengths == {'time': 91}
    with descriptor.opener.open(path, COORDS) as ds:
        assert ds['x'].isel(IRIDL_time=0).sizes['time'] == 91


def test_old_pad_dims_name_is_rejected(tmp_path):
    # Catalogs still using pad_dims must fail loudly, not be ignored.
    with pytest.raises(TypeError):
        preprocess.FileSetDescriptor(
            name='x', dir=tmp_path, catalog_path=preprocess.URLPath('x'),
            pattern=r'f\.nc$', pad_dims={'time': 91},
        )