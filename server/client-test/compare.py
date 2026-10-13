import time

import numpy as np
import xarray as xr


def compare_ds(ds1, ds2, atol):
    names = list(ds1.data_vars)
    assert len(names) == 1
    var1 = names[0]
    
    names = list(ds2.data_vars)
    assert len(names) == 1 or len(names) == 2
    if len(names) == 2:
        if 'target_bnds' in names:
            ds2 = ds2.assign_coords({'target_bnds': ds2['target_bnds']})
            names = list(ds2.data_vars)
            assert len(names) == 1
            var2 = names[0]
        else:
            raise Exception('2nd var should be target_bnds')
    else:
        var2 = names[0]

    da1 = ds1[var1]
    da2 = ds2[var2]

    all_same = True
    all_same &= compare_shape(da1, da2)
    if all_same:
        all_same &= compare_coords(da1, da2)
        all_same &= compare_data(da1, da2, atol)
    return all_same

def compare_coords(ds1, ds2):
    all_same = True
    for cname in sorted(set(ds1.coords) | set(ds2.coords)):
        if cname == 'target':
            continue
        c1 = ds1.coords.get(cname)
        c2 = ds2.coords.get(cname)
        if c1 is None:
            print(cname, 'absent', 'present')
            return False
        if c2 is None:
            print(cname, 'present', 'absent')
            return False
        if cname == 'S':
            # Just check that all sigrid values are also in iridl,
            # not vice versa, because of different handling of
            # missing values. compare_shape limits how many missing
            # values are tolerated.
            same = set(c1.values) <= set(c2.values)
        elif np.issubdtype(c1.dtype, np.floating):
            same = np.allclose(c1.values, c2.values, equal_nan=True)
        else:
            same = np.array_equal(c1.values, c2.values)
        if not same:
            print(cname, 'values differ:')
            print(c1.values)
            print(c2.values)
            all_same = False
    return all_same

def compare_shape(da1, da2):
    dims1 = sorted(
        (name, size) for name, size in da1.sizes.items()
        if name != 'S'
    )
    dims2 = sorted(
        (name, size) for name, size in da2.sizes.items()
        if name != 'S'
    )
    s1 = da1.sizes['S']
    s2 = da2.sizes['S']
    if dims1 == dims2:
        if s1 == s2:
            print('same dims')
            return True
        if (s2 - s1) / s2 < .2:
            print(f'mostly same dims, S close enough: {s1} vs {s2}')
            return True
        print(f'S lengths are too different: {s1} vs {s2}')
        return False
    else:
        print('different dims')
        print(dims1, f'S={s1}')
        print(dims2, f'S={s2}')
        return False

def compare_data(da1, da2, atol):
    # When there are missing S values, sigrid has gaps in S, while ingrid
    # has a regular S grid but some entries are all-NaN. In that case, pick
    # three S slices that are present in both to compare, and pick three
    # S slices that are missing from sigrid and verify that they're all-NaN
    # in ingrid.
    s_len = da1.sizes['S']
    all_same = True
    for i in (0, s_len // 2, s_len - 1):
        s = da1['S'].isel(S=i).values
        slice1 = da1.sel(S=s)
        slice2 = da2.sel(S=s)
        same = compare_slice(slice1, slice2, atol=atol)
        if same:
            print(f"S={s}: same")
        else:
            print(f"S={s}:")
            print('da1')
            if 'M' in da1.sizes: 
                print(slice1.isel(L=0, M=0).values)
            else:
                print(slice1.isel(L=0).values)
            print('da2')
            print(slice2.isel(L=0).values)
        all_same &= same
    if da1.sizes['S'] < da2.sizes['S']:
        missing_s = np.setdiff1d(da2['S'].values, da1['S'].values)
        missing_count = len(missing_s)
        if missing_count > 3:
            missing_s = missing_s[[0, missing_count // 2, missing_count - 1]]
        for s in missing_s:
            slice2 = da2.sel(S=s)
            if not slice2.isnull().all():
                print(f'S={s} is missing from da1 and non-NaN in da2')
                all_same = False
    return all_same

def compare_slice(da1, da2, atol):
    start = time.time()

    dim_chunk='L'
    if 'M' in da1.sizes and da1.sizes['M'] > da1.sizes['L']:
        dim_chunk='M'

    da1 = load_chunked(da1, dim=dim_chunk)
    print(f'da1 took {time.time() - start}s')
    start = time.time()
    da2 = load_chunked(da2, dim=dim_chunk)
    print(f'da2 took {time.time() - start}s')

    da1, da2 = xr.align(da1, da2, join='override')
    if np.allclose(da1, da2, equal_nan=True, atol=atol):
        return True
    nan_diff = np.logical_xor(da1.isnull(), da2.isnull())
    if nan_diff.any():
        idx = nan_diff.argmax(...)
        # np.array2string prints 32-bit floats with the appropriate number of
        # digits, unlike python's built-in formatting, which converts to a 64-bit
        # float and then formats that.
        labels = {dim: np.array2string(nan_diff[dim].values[i.item()]) for dim, i in idx.items()}
        print(f'Unaligned NaNs at {labels}')
    diff = np.abs(da1 - da2)
    idx = diff.argmax(...)
    max_diff = diff[idx]
    if max_diff > atol:
        labels = {dim: np.array2string(diff[dim].values[i.item()]) for dim, i in idx.items()}
        print(f'tolerance {atol}, max diff {np.array2string(max_diff)} between {np.array2string(da1[idx])} and {np.array2string(da2[idx])} at {labels}')
    return False

def fetch(url):
    ds = xr.open_dataset(url, decode_times=False)
    for coord in ds.variables.values():
        if coord.attrs.get("calendar") == "360":
            coord.attrs["calendar"] = "360_day"

    units = ds['L'].attrs.get('units')
    if units is None or 'months' in units:
        ds = xr.decode_cf(ds)
    else:
        ds = xr.decode_cf(ds, decode_timedelta=True)
        
    return ds
 
def parse_listfile(filename):
    paths = {}
    with open(filename) as f:
        url1 = None
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            if url1 is None:
                url1 = line
            else:
                paths[url1] = line
                url1 = None
    return paths

def load_chunked(da, dim='M', chunk_size=10):
    """Load a DataArray in chunks along `dim` to avoid oversized single requests
    (e.g. Ingrid's 413 Request Entity Too Large on very large slices)."""
    if dim not in da.dims or da.sizes[dim] <= chunk_size:
        da.load()
        return da

    pieces = []
    size = da.sizes[dim]
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        piece = da.isel({dim: slice(start, end)})
        piece.load()
        pieces.append(piece)
    return xr.concat(pieces, dim=dim)