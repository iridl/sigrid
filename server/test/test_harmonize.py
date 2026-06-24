import numpy as np
import pytest
import xarray as xr
import xarray.testing

import sigrid.harmonize as harmonize


@pytest.fixture
def make_dataset():
    """Factory for a small synthetic forecast-style dataset.

    Mirrors the dims/coords the harmonization pipeline expects (S, L, Y, X) with
    a single data variable whose ``units`` attribute is configurable, so tests
    can drive unit conversion and attribute standardization without any I/O.
    """
    def _make(
        varname: str = 'prcp',
        units: str | None = 'mm/day',
        n_s: int = 3,
        n_l: int = 4,
        ys: tuple[float, ...] = (0.0, 1.0),
        xs: tuple[float, ...] = (10.0, 11.0),
    ) -> xr.Dataset:
        s_vals = xr.date_range('1980-01-01', periods=n_s, freq='MS')
        attrs = {} if units is None else {'units': units}
        data = np.arange(
            n_s * n_l * len(ys) * len(xs), dtype='float64'
        ).reshape(n_s, n_l, len(ys), len(xs))
        return xr.Dataset(
            {varname: (('S', 'L', 'Y', 'X'), data, attrs)},
            coords={
                'S': s_vals,
                'L': np.arange(n_l),
                'Y': list(ys),
                'X': list(xs),
            },
        )
    return _make


# --- rename -----------------------------------------------------------------

def test_rename_present_keys(make_dataset):
    ds = make_dataset(varname='raw_prcp')
    out = harmonize.rename(ds, {'raw_prcp': 'prcp', 'X': 'lon'})
    assert 'prcp' in out.data_vars and 'raw_prcp' not in out.variables
    assert 'lon' in out.dims and 'X' not in out.dims


def test_rename_skips_absent_and_identity(make_dataset):
    ds = make_dataset(varname='prcp')
    # 'nope' isn't present; 'prcp'->'prcp' is identity. Neither should error.
    out = harmonize.rename(ds, {'nope': 'whatever', 'prcp': 'prcp'})
    assert set(out.variables) == set(ds.variables)


# --- drop_non_std -----------------------------------------------------------

def test_drop_non_std_drops_unlisted_vars(make_dataset):
    ds = make_dataset(varname='prcp')
    ds['extra'] = (('S',), np.zeros(ds.sizes['S']))
    std = {'prcp': {}, 'S': {}, 'L': {}, 'Y': {}, 'X': {}}
    out = harmonize.drop_non_std(ds, std, bare_dims=[])
    assert 'extra' not in out.variables
    assert 'prcp' in out.variables


def test_drop_non_std_raises_on_nonstandard_dim(make_dataset):
    ds = make_dataset(varname='prcp')
    # M is a dim with no coord var and isn't in standard_attrs or bare_dims.
    ds = ds.expand_dims({'M': 2})
    std = {'prcp': {}, 'S': {}, 'L': {}, 'Y': {}, 'X': {}}
    with pytest.raises(Exception, match='non standard dims'):
        harmonize.drop_non_std(ds, std, bare_dims=[])


def test_drop_non_std_allows_bare_dim(make_dataset):
    ds = make_dataset(varname='prcp')
    ds = ds.expand_dims({'M': 2})
    std = {'prcp': {}, 'S': {}, 'L': {}, 'Y': {}, 'X': {}}
    out = harmonize.drop_non_std(ds, std, bare_dims=['M'])
    assert 'M' in out.dims


# --- convert_units_da -------------------------------------------------------

def _da(value, units):
    da = xr.DataArray(np.array([value], dtype='float64'), dims=['x'], name='v')
    if units is not None:
        da.attrs['units'] = units
    return da


def test_convert_units_da_kelvin():
    da = _da(300.0, 'K')
    out = harmonize.convert_units_da(da, 'degree_Celsius')
    assert out.values[0] == pytest.approx(300.0 - 273.15)


def test_convert_units_da_same_units_unchanged():
    da = _da(5.0, 'mm/day')
    out = harmonize.convert_units_da(da, 'mm/day')
    assert out.values[0] == 5.0
    assert out.attrs['units'] == 'mm/day'


def test_convert_units_da_drop_units_keep_values():
    da = _da(5.0, 'K')
    out = harmonize.convert_units_da(da, None)
    assert out.values[0] == 5.0
    assert 'units' not in out.attrs


def test_convert_units_da_missing_original_raises():
    da = _da(5.0, None)
    with pytest.raises(Exception):
        harmonize.convert_units_da(da, 'degree_Celsius')


def test_convert_units_da_unknown_pair_raises():
    da = _da(1.0, 'parsecs')
    with pytest.raises(KeyError):
        harmonize.convert_units_da(da, 'mm/day')


# --- convert_units ----------------------------------------------------------

def test_convert_units(make_dataset):
    ds = make_dataset(varname='t2m', units='K')
    std = {'t2m': {'units': 'degree_Celsius'}}
    out = harmonize.convert_units(ds, std)
    xarray.testing.assert_allclose(out['t2m'], ds['t2m'] - 273.15)


# --- standardize_attrs ------------------------------------------------------

def test_standardize_attrs_replaces_and_drops_grib(make_dataset):
    ds = make_dataset(varname='prcp')
    ds['prcp'].attrs['provider_junk'] = 'remove me'
    ds.attrs['GRIB_centre'] = 'noise'
    ds.attrs['existing'] = 'keep'
    da_attrs = {
        'prcp': {
            'standard_name': 'lwe_precipitation_rate',
            'units': 'mm/day'
        },
        'S': {}, 'L': {}, 'Y': {}, 'X': {},
    }
    out = harmonize.standardize_attrs(
        ds, da_attrs=da_attrs, ds_attrs={'Conventions': 'CF-1.13'}, encodings={},
    )
    assert out['prcp'].attrs == {
        'standard_name': 'lwe_precipitation_rate',
        'units': 'mm/day',
    }
    assert out.attrs == {
        'Conventions': 'CF-1.13',
        'existing': 'keep'
    }


def test_standardize_attrs_applies_encodings(make_dataset):
    ds = make_dataset(varname='prcp')
    da_attrs = {'prcp': {}, 'S': {}, 'L': {}, 'Y': {}, 'X': {}}
    encodings = {'S': {'dtype': 'int32', 'units': 'hours since 1960-01-01'}}
    out = harmonize.standardize_attrs(ds, da_attrs, ds_attrs={}, encodings=encodings)
    assert out['S'].encoding['dtype'] == 'int32'
    assert out['S'].encoding['units'] == 'hours since 1960-01-01'


# --- add_target -------------------------------------------------------------

def test_add_target_monthly(make_dataset):
    ds = make_dataset(varname='prcp', n_s=2, n_l=3)
    out = harmonize.add_target(ds, lead_is_month=True)

    assert out['target_bnds'].dims == ('S', 'L', 'nbound')
    assert np.array_equal(
        out['target_bnds'].values,
        np.array([
            [['1980-01', '1980-02'], ['1980-02', '1980-03'], ['1980-03', '1980-04']],
            [['1980-02', '1980-03'], ['1980-03', '1980-04'], ['1980-04', '1980-05']],
        ], dtype=np.datetime64)
    )


# --- monthly_total ----------------------------------------------------------

def test_monthly_total_converts_rate(make_dataset):
    ds = make_dataset(varname='prcp', units='mm/day', n_s=2, n_l=2)
    ds = harmonize.add_target(ds, lead_is_month=True)
    out = harmonize.monthly_total(ds, ['prcp'])
    assert out['prcp'].attrs['units'] == 'mm'
    xarray.testing.assert_allclose(out['prcp'].isel(S=0, L=0), ds['prcp'].isel(S=0, L=0) * 31)
    xarray.testing.assert_allclose(out['prcp'].isel(S=0, L=1), ds['prcp'].isel(S=0, L=1) * 29)
    xarray.testing.assert_allclose(out['prcp'].isel(S=1, L=0), ds['prcp'].isel(S=1, L=0) * 29)
    xarray.testing.assert_allclose(out['prcp'].isel(S=1, L=1), ds['prcp'].isel(S=1, L=1) * 31)


def test_monthly_total_rejects_non_daily_rate(make_dataset):
    ds = make_dataset(varname='prcp', units='mm', n_s=1, n_l=1)
    ds = harmonize.add_target(ds, lead_is_month=True)
    with pytest.raises(Exception, match='rate per day'):
        harmonize.monthly_total(ds, ['prcp'])


# --- standardize --------------------------------------

def test_standardize(make_dataset):
    ds = make_dataset(varname='t2m', units='K', n_s=2, n_l=3)
    ds['junk'] = (('S',), np.zeros(ds.sizes['S']))
    config = harmonize.DatasetConfig(
        da_attrs={
            't2m': {'standard_name': 'air_temperature', 'units': 'degree_Celsius'},
            'S': {'standard_name': 'forecast_reference_time'},
            'L': {}, 'Y': {'units': 'degree_north'}, 'X': {'units': 'degree_east'},
            'target': {}, 'target_bnds': {},
        },
        ds_attrs={'Conventions': 'CF-1.13'},
        encodings={},
        bare_dims=['nbound'],
        lead_is_month=True,
    )
    out = harmonize.standardize(ds, config)
    # dropped, converted, target added, attrs standardized
    assert 'junk' not in out.variables
    np.testing.assert_allclose(out['t2m'].values, ds['t2m'].values - 273.15)
    assert 'target' in out.coords and 'target_bnds' in out.coords
    assert out['t2m'].attrs == {
        'standard_name': 'air_temperature', 'units': 'degree_Celsius',
    }
    assert out.attrs['Conventions'] == 'CF-1.13'
