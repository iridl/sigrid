import argparse
import sys
import time

import numpy as np
import recording_proxy
import xarray as xr

def compare(url1, url2, atol):
    try:
        ds1 = fetch(url1)
    except Exception as e:
        print('first failed', e)
        return
    
    try:
        ds2 = fetch(url2)
    except Exception as e:
        print('second failed', e)
        return

    ds2 = ds2.convert_calendar('standard', dim='S', align_on='date')
    compare_ds(ds1, ds2, atol)

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
    all_same &= compare_coords(da1, da2)
    all_same &= compare_data(da1, da2, atol)
    return all_same

def compare_coords(ds1, ds2):
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
        if not np.array_equal(c1.values, c2.values):
            print(cname, 'values differ:')
            print(c1.values)
            print(c2.values)
            return False
        #compare_attrs(c1, c2)
    return True

def compare_shape(da1, da2):
     dims1 = sorted(list(da1.sizes.items()))
     dims2 = sorted(list(da2.sizes.items()))
     if dims1 == dims2:
         print('same dims')
         return True
     else:
         print('different dims:')
         print(dims1)
         print(dims2)
         return False

def compare_data(da1, da2, atol):
    # Accomodating the fact that Ingrid typically has a regular S grid, even if
    # we have no files for some values of S, whereas pydap's S coordinate only
    # contains values of S for which files are present. Only compare data for dates
    # that exist in both datasets; rely on compare_shape to catch missing dates.
    s_len = da1.sizes['S']
    all_same = True
    for i in (0, s_len // 2, s_len - 1):
        s = da1['S'].isel(S=i).values
        same = compare_slice(da1.sel(S=s), da2.sel(S=s), atol=atol)
        print(f"S={s}: {same}")
        all_same &= same
    return all_same

def compare_slice(da1, da2, atol):
    start = time.time()
    a1 = da1.values
    print(f'da1 took {time.time() - start}s')
    start = time.time()
    a2 = da2.values
    print(f'da2 took {time.time() - start}s')
    return np.isclose(a1, a2, equal_nan=True, atol=atol).all()

def compare_attrs(da1, da2):
    for a in sorted(set(da1.attrs) | set(da2.attrs)):
        print(f"  {a:30.30} {str(da1.attrs.get(a, '')):20.20} {str(da2.attrs.get(a,'')):20.20}")

def fetch(url):
    ds = xr.open_dataset(url, decode_times=False)
    for name, coord in ds.variables.items():
        if coord.attrs.get("calendar") == "360":
            coord.attrs["calendar"] = "360_day"
    ds = xr.decode_cf(ds)
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
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root")
    parser.add_argument("reference_root")
    parser.add_argument("listfile")
    parser.add_argument("test_path")
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--atol", type=float, default=1e-8)
    
    args = parser.parse_args()
    path_mapping = parse_listfile(args.listfile)
    url1 = f'{args.test_root}/{args.test_path}'
    url2 = f'{args.reference_root}/{path_mapping[args.test_path]}'
    with recording_proxy.recording_proxy(
            "responses",
            args.record,
            prefixes=[args.reference_root],
            verbose=args.verbose,
    ):
        all_same = compare(url1, url2, args.atol)
    print(all_same)
    if all_same:
        sys.exit(0)
    else:
        sys.exit(1)
