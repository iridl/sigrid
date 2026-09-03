import dask.array
import numpy as np
import pytest
import xarray as xr

from sigrid.harmonize import grids

# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #

def gaussian_grid(n_lat, n_lon, y_ascending=True):
    mu, _ = np.polynomial.legendre.leggauss(n_lat)
    lat = np.degrees(np.arcsin(mu))
    if not y_ascending:
        lat = lat[::-1]
    lon = np.arange(n_lon) * 360.0 / n_lon
    return grids.make_grid(lat, lon, gaussian=True)

def nmme_grid(n_lat, n_lon):
    """NMME-style grid where Y extends from -90 to 90 inclusive. Grid sizes are
    regular except at the poles, where cells are truncated to avoid overlapping."""
    return grids.make_grid(
        np.linspace(-90, 90, n_lat),
        np.arange(n_lon) * 360.0 / n_lon
    )

def regular_grid(n_lat, n_lon):
    """Grid where all cells are the same size"""
    Y_bounds = np.linspace(-90, 90, n_lat + 1)
    Y = (Y_bounds[:-1] + Y_bounds[1:]) / 2
    X = np.arange(n_lon) * 360.0 / n_lon
    return grids.make_grid(Y, X)

def cell_areas(ds):
    lat_width = np.abs(np.diff(np.sin(np.radians(grids._bounds_for(ds, 'Y')))))
    lon_width = np.abs(np.diff(np.radians(grids._bounds_for(ds, 'X'))))
    return np.outer(lat_width, lon_width)

def integral(ds):
    da = next(iter(ds.data_vars.values()))
    return np.sum(da * cell_areas(ds))

def smooth_field(ds):
    """A smooth, strictly positive precip-like field (no sub-grid variance that
    would defeat a coarsening conservation check)."""
    la = np.radians(ds['Y'].values)[:, np.newaxis]
    lo = np.radians(ds['X'].values)[np.newaxis, :]
    f = 2.0 + np.cos(la) ** 2 + 0.5 * np.cos(la) * np.sin(2 * lo)
    ds = ds.copy()
    ds['pr'] = (('Y', 'X'), f)
    return ds



# --------------------------------------------------------------------------- #
# output shape + metadata
# --------------------------------------------------------------------------- #

def test_output_shape():
    src = smooth_field(gaussian_grid(48, 96))
    dst = nmme_grid(37, 72)
    out = grids.regrid_conservative(src, dst)
    assert out['pr'].shape == (37, 72)
    assert out['Y'].equals(dst['Y'])
    assert out['X'].equals(dst['X'])

def test_broadcast():
    src = smooth_field(gaussian_grid(48, 96)).expand_dims(M=3, S=2)
    dst = nmme_grid(37, 72)
    out = grids.regrid_conservative(src, dst)
    assert out['pr'].shape == (3, 2, 37, 72)

def test_attrs_preserved():
    src = smooth_field(gaussian_grid(48, 96)).expand_dims(M=3, S=2)
    src['pr'].attrs["units"] = "kg m-2 s-1"
    dst = nmme_grid(37, 72)
    out = grids.regrid_conservative(src, dst)
    assert out['pr'].attrs.get("units") == "kg m-2 s-1"


# --------------------------------------------------------------------------- #
# conservation: area-weighted global integral is preserved
# --------------------------------------------------------------------------- #

def test_fine_to_coarse_nmme_conserves():
    src = smooth_field(gaussian_grid(48, 96))
    dst = nmme_grid(37, 72)
    out = grids.regrid_conservative(src, dst)
    src_integral = integral(src)
    dst_integral = integral(out)
    assert abs(dst_integral - src_integral) / abs(src_integral) < 1e-12

def test_fine_to_coarse_regular_conserves():
    src = smooth_field(gaussian_grid(48, 96))
    dst = regular_grid(36, 72)
    out = grids.regrid_conservative(src, dst)
    src_integral = integral(src)
    dst_integral = integral(out)
    assert abs(dst_integral - src_integral) / abs(src_integral) < 1e-12

def test_coarse_to_fine_conserves():
    src = smooth_field(nmme_grid(37, 72))
    dst = nmme_grid(181, 360)
    out = grids.regrid_conservative(src, dst)
    src_integral = integral(src)
    dst_integral = integral(out)
    assert abs(dst_integral - src_integral) / abs(src_integral) < 1e-12

def test_same_grid_is_identity():
    dst = nmme_grid(48, 96)
    src = smooth_field(dst)
    out = grids.regrid_conservative(src, dst)
    np.testing.assert_allclose(out['pr'], src['pr'], rtol=1e-12)


# --------------------------------------------------------------------------- #
# a constant field regrids to the same constant
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("c", [0.0, 3.7, -2.5])
def test_constant_preserved(c):
    src = nmme_grid(48, 96)
    src['pr'] = (
        ('Y', 'X'),
        np.full(
            (len(src['Y']), len(src['X'])),
            c
        )
    )
    dst = nmme_grid(50, 98)
    out = grids.regrid_conservative(src, dst)
    np.testing.assert_allclose(out['pr'].values, c, atol=1e-12)


# --------------------------------------------------------------------------- #
# NaN guard
# --------------------------------------------------------------------------- #

def test_nan_input_raises():
    src = smooth_field(nmme_grid(181, 360))
    src['pr'][10, 20] = np.nan
    dst = nmme_grid(91, 180)
    with pytest.raises(ValueError, match="NaN"):
        grids.regrid_conservative(src, dst)

# --------------------------------------------------------------------------- #
# Dask chunking
# --------------------------------------------------------------------------- #

def test_broadcast_dims_chunked():
    one_slice = smooth_field(gaussian_grid(48, 96))
    src = xr.concat([one_slice] * 4, dim='L')
    src['pr'].data = dask.array.from_array(src['pr'].values, chunks=(2, 48, 96)) # pyright: ignore[reportArgumentType]
    dst = nmme_grid(37, 72)
    out = grids.regrid_conservative(src, dst)
    assert out.chunks is not None
    assert out['pr'].shape == (4, 37, 72)

def test_core_dims_chunked_fails():
    # If the array is chunked on X or Y, we must fail rather than giving an
    # incorrect result.
    src = smooth_field(gaussian_grid(48, 96))
    src['pr'].data = dask.array.from_array(src['pr'].values, chunks=(24, 48)) # pyright: ignore[reportArgumentType]
    dst = nmme_grid(37, 72)
    with pytest.raises(ValueError):
        grids.regrid_conservative(src, dst)

# --------------------------------------------------------------------------- #
# periodic longitude
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('start,step', [(0, 1), (0, 7), (1, 1), (1, 7)])
def test_lon_edges_close_the_circle(start, step):
    grid = grids.make_grid(
        np.linspace(-90, 90, 181),
        np.arange(start, 360, step)
    )
    lon_edges = grids.bounds_cf_to_1d(grid['X_bnds'].values)
    assert lon_edges[0] + 360 == lon_edges[-1]
    assert np.sum(np.abs(np.diff(lon_edges))) == 360
