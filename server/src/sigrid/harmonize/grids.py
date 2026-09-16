import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from . import Coords as C


def regrid_bilinear(
        ds_in: xr.Dataset,
        ds_out: xr.Dataset,
) -> xr.Dataset:
    """Bilinear interpolation with periodic longitude"""
    lon = ds_in[C.X]
    period = 360.
    # Pad one column on each side to avoid discontinuity at the seam
    left = ds_in.isel({C.X: -1}).assign_coords({C.X: lon[-1].item() - period})
    right = ds_in.isel({C.X: 0}).assign_coords({C.X: lon[0].item() + period})
    ds_in = xr.concat([left, ds_in, right], dim=C.X)
    ds_in = ds_in.interp({C.Y: ds_out[C.Y], C.X: ds_out[C.X]}, method='linear')
    return ds_in


def regrid_conservative(
        ds_in: xr.Dataset,
        ds_out: xr.Dataset,
) -> xr.Dataset:
    """First-order conservative regrid for lat-lon grids. ds_in is the Dataset
    to be regridded, and ds_out is a Dataset whose X and Y coordinates define
    the target grid. ds_out needn't have any data variables; they are ignored if
    present. Both ds_in and ds_out must have bounds coordinates associated with
    X and Y.

    If you want to regrid a Dataset that lacks bounds coordinates, use
    `ensure_bounds` to add them.
    """
    src_lat_bounds = _bounds_for(ds_in, C.Y)
    src_lon_bounds = _bounds_for(ds_in, C.X)
    dst_lat_bounds = _bounds_for(ds_out, C.Y)
    dst_lon_bounds = _bounds_for(ds_out, C.X)

    # Ensure that src lat and lon are both in increasing order. This simplifies
    # what follows.
    if ds_in[C.Y][0] > ds_in[C.Y][1]:
        ds_in = ds_in.isel({C.Y: slice(None, None, -1)})
        src_lat_bounds = src_lat_bounds[::-1]
    if ds_in[C.X][0] > ds_in[C.X][1]:
        ds_in = ds_in.isel({C.X: slice(None, None, -1)})
        src_lon_bounds = src_lon_bounds[::-1]

    lat_weights = _overlap_lat(src_lat_bounds, dst_lat_bounds)
    lon_weights = _overlap_lon(src_lon_bounds, dst_lon_bounds)

    # apply_ufunc consumes (lat, lon) and produces new axes determined by the
    # destination grid. Use temporary names to avoid colliding with the input
    # core dims, rename back to the original names.
    out: xr.Dataset = xr.apply_ufunc(
        _regrid_chunk,
        ds_in,
        input_core_dims=[[C.Y, C.X]],
        output_core_dims=[["__lat", "__lon"]],
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            'output_sizes': {
                "__lat": len(ds_out[C.Y]),
                "__lon": len(ds_out[C.X])
            }
        },
        kwargs={'lat_weights': lat_weights, 'lon_weights': lon_weights},
    ).rename({"__lat": C.Y, "__lon": C.X})

    out = out.assign_coords({
        name: ds_out[name] for name in (
            C.Y, ds_out[C.Y].attrs['bounds'],
            C.X, ds_out[C.X].attrs['bounds']
        )
    })
    return out


def make_grid(Y: ArrayLike, X: ArrayLike, gaussian: bool = False) -> xr.Dataset:
    return ensure_bounds(
        xr.Dataset({'Y': Y, 'X': X}),
        gaussian=gaussian
    )


def ensure_bounds(input: xr.DataArray | xr.Dataset, gaussian: bool = False) -> xr.Dataset:
    """Add X_bnds and Y_bnds coordinates if not already present. Use
    Gauss-Legendre quadrature if gaussian=True, otherwise use the midpoints
    between neighboring grid points, truncating at the poles to avoid cells that
    overlap each other.
    """

    input = input.copy()
    if isinstance(input, xr.DataArray):
        ds = input.to_dataset()
    else:
        ds = input

    if gaussian:
        lat_bounds_1d = _gaussian_lat_bounds(ds[C.Y])
    else:
        lat_bounds_1d = _regular_lat_bounds(ds[C.Y])
    lat_bounds_cf = bounds_1d_to_cf(lat_bounds_1d)
    bounds_var: str | None = ds[C.Y].attrs.get('bounds')
    if bounds_var is None:
        bounds_var = f'{C.Y}_bnds'
        ds[C.Y].attrs['bounds'] = bounds_var
        ds = ds.assign_coords({bounds_var: ((C.Y, C.nbound), lat_bounds_cf)})
    else:
        if not np.allclose(lat_bounds_cf, ds[bounds_var]):
            raise Exception('Latitude bounds are incompatible with latitudes')

    lon_bounds_cf = bounds_1d_to_cf(_regular_lon_bounds(ds[C.X]))
    bounds_var: str | None = ds[C.X].attrs.get('bounds')
    if bounds_var is None:
        bounds_var = f'{C.X}_bnds'
        ds[C.X].attrs['bounds'] = bounds_var
        ds = ds.assign_coords({bounds_var: ((C.X, C.nbound), lon_bounds_cf)})
    else:
        if not np.allclose(lon_bounds_cf, ds[bounds_var]):
            raise Exception('Longitude bounds are incompatible with longitudes')

    return ds


def _regrid_chunk(arr: np.ndarray, lat_weights: np.ndarray, lon_weights: np.ndarray) -> np.ndarray:
    if np.isnan(arr).any() and not np.isnan(arr).all():
        # Not bothering to handle NaNs until we need it.
        raise ValueError("Source contains mix of NaNs and non-NaNs")
    return lat_weights @ arr @ lon_weights.T


def _bounds_for(ds: xr.Dataset, var: str) -> np.ndarray:
    bounds_var = ds[var].attrs.get('bounds')
    if bounds_var is None or bounds_var not in ds:
        raise Exception(f"Missing bounds for {var}. Consider calling ensure_bounds first.")
    return bounds_cf_to_1d(ds[bounds_var].values)


def _regular_lat_bounds(lat: ArrayLike) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    midpoints = (lat[:-1] + lat[1:]) / 2
    bounds = np.concatenate([[2*lat[0] - midpoints[0]], midpoints, [2*lat[-1] - midpoints[-1]]])
    # Don't allow cells to go past the poles, because that would make polar
    # cells overlap, which would make conserving the mean meaningless. First and
    # last cells may thus come out smaller than the others.
    bounds = np.clip(bounds, -90.0, 90.0)
    return bounds


def _gaussian_lat_bounds(lat: ArrayLike, atol: float = 1e-8) -> np.ndarray:
    """Exact latitude cell bounds for a Gaussian grid, from quadrature weights.
    """
    # The cell measure is d(sin lat) = dmu, and the Gaussian weights are the
    # mu-widths of the cells, summing to 2. The edges in mu are therefore the
    # running partial sums of the weights starting at -1; arcsin maps them back to
    # latitude, landing on +-90 at the ends.
    lat = np.asarray(lat, dtype=float)
    n = len(lat)

    descending = lat[0] > lat[-1]
    if descending:
        lat = lat[::-1]

    mu_nodes, w = np.polynomial.legendre.leggauss(n)   # increasing mu, sum(w)=2
    if not np.allclose(np.sin(np.radians(lat)), mu_nodes, atol=atol):
        raise ValueError(
            f"latitudes are not a Gaussian grid to tol={atol}; got spacing that "
            "does not match Legendre nodes."
        )
    bounds_mu = np.concatenate([[-1.0], -1.0 + np.cumsum(w)])
    bounds_mu[-1] = 1.0  # fix rounding at pole
    bounds = np.degrees(np.arcsin(bounds_mu))

    if descending:
        bounds = bounds[::-1]

    return bounds


def _regular_lon_bounds(lon: ArrayLike, atol: float = 1e-8) -> np.ndarray:
    """Cell bounds for a periodic 1-D longitude coordinate. Uses the midpoint
    between grid points as the boundary.
    """
    lon = np.asarray(lon, dtype=float)
    descending = lon[-1] < lon[0]
    if descending:
        lon = lon[::-1]

    midpoints = (lon[:-1] + lon[1:]) / 2
    wrap = ((lon[0] + 360.0) + lon[-1]) / 2
    bounds = np.concatenate([[wrap - 360.0], midpoints, [wrap]])

    if descending:
        bounds = bounds[::-1]

    return bounds


def _overlap_lat(src_bounds: np.ndarray, dst_bounds: np.ndarray) -> np.ndarray:
    """Row-normalized (n_dst, n_src) latitude overlap matrix, assuming both edge
    arrays are in increasing order. Each dst row sums to 1."""
    # s[:-1] is a row vector of source cell lower edges. d[:-1, np.newaxis] is a
    # column vector of destination cell lower edges. When we apply np.maximum to
    # a (n_dst, 1) column and a (1, n_src) row, numpy broadcasts both to (n_dst,
    # n_src): the column is repeated across all n_src columns, the row across
    # all n_dst rows. So entry [i, j] of lo is the greater of destination cell
    # i's lower edge and source cell j's lower edge, i.e. the lower edge of the
    # overlap between those two cells. Similarly, hi is the upper edge of the
    # overlap; so hi - lo is the size of the intersection when they overlap, and
    # negative when they don't. We then clip negative numbers to 0.
    s = np.sin(np.radians(src_bounds))
    d = np.sin(np.radians(dst_bounds))
    lo = np.maximum(d[:-1, np.newaxis], s[:-1])
    hi = np.minimum(d[1:, np.newaxis], s[1:])
    ov = np.clip(hi - lo, 0.0, None)
    return ov / ov.sum(axis=1, keepdims=True)


def _overlap_lon(src_bounds: np.ndarray, dst_bounds: np.ndarray) -> np.ndarray:
    """Row-normalized overlap on a circle: each destination cell collects source
    overlap from the source cell and its +-period images, so a cell straddling
    the 0/360 seam contributes to destinations on both sides."""
    period = 360
    s = np.asarray(src_bounds, float)
    d = np.asarray(dst_bounds, float)
    ov = np.zeros((len(d) - 1, len(s) - 1))
    for shift in (-period, 0.0, period):
        lo = np.maximum(d[:-1, np.newaxis], s[np.newaxis, :-1] + shift)
        hi = np.minimum(d[1:, np.newaxis], s[np.newaxis, 1:] + shift)
        ov += np.clip(hi - lo, 0.0, None)
    return ov / ov.sum(axis=1, keepdims=True)


def bounds_1d_to_cf(simple_bounds: np.ndarray) -> np.ndarray:
    """Given an array of shape (n+1) representing the bounds of n cells, return
    an array of shape (n, 2) representing the same bounds in the format used by
    the CF Conventions."""
    return np.stack([simple_bounds[:-1], simple_bounds[1:]], axis=1)


def bounds_cf_to_1d(cf_bounds: np.ndarray) -> np.ndarray:
    """Given an array of shape (n, 2) representing cell bounds in the redundant
    format of the CF Conventions, return an array of shape (n+1) representing
    the unique bounds."""
    return np.concatenate([cf_bounds[:, 0], cf_bounds[-1:, 1]])
