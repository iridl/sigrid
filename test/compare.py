import argparse
import sys
import time

import numpy as np
import xarray as xr

def compare(url1, url2):
    target = False
    split_url1 = url1.split('/')
    if split_url1[-1] == 'target':
        url1 = '/'.join(split_url1[0:-1])
        target = True
    ds1 = fetch(url1, target=target)
    ds2 = fetch(url2)

    if isinstance(ds1, Exception):
        print('first failed', ds1)
        return
    if isinstance(ds2, Exception):
        print('second failed', ds2)
        return
    
    names = list(ds1.data_vars)
    assert len(names) == 1
    var1 = names[0]
    
    names = list(ds2.data_vars)
    assert len(names) == 1
    var2 = names[0]
    
    da1 = ds1[var1]
    da2 = ds2[var2]

    all_same = True
    all_same &= compare_shape(da1, da2)
    all_same &= compare_data(da1, da2)
    return all_same

def compare_coords(da1, da2):
    for cname in sorted(set(da1.coords) | set(da2.coords)):        
        c1 = da1.coords.get(cname)
        c2 = da2.coords.get(cname)
        print(cname)
        if c1 is None:
            print(cname, 'absent', 'present')
            continue
        if c2 is None:
            print(cname, 'present', 'absent')
            continue
        if np.array_equal(c1.values, c2.values):
            print('values are the same')
        else:
            print('values differ:')
            print(c1.values)
            print(c2.values)
        compare_attrs(c1, c2)

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

def compare_data(da1, da2):
    # Accomodating the fact that Ingrid typically has a regular S grid, even if
    # we have no files for some values of S, whereas pydap's S coordinate only
    # contains values of S for which files are present. Only compare data for dates
    # that exist in both datasets; rely on compare_shape to catch missing dates.
    da2 = da2.convert_calendar('standard', dim='S', align_on='date')
    s_len = da1.sizes['S']
    all_same = True
    if da1.name == 'target_bounds':
        da1 = da1['target_bnds'].dt.strftime("%Y%m%dT%H:%M")
        da2 = da2.dt.strftime("%Y%m%dT%H:%M")
        all_same = all([
            [
                [
                    da1.sel(S=s).isel(L=l, nbound=n).values == da2.sel(S=s).isel(L=l, nbound=n).values
                    for n in range(da1.sizes['nbound'])
                ]
                for l in range(da1.sizes['L'])
            ]
            for s in da1['S']
        ])
    else:
        for i in (0, s_len // 2, s_len - 1):
            s = da1['S'].isel(S=i).values
            same = compare_slice(da1.sel(S=s), da2.sel(S=s))
            print(f"S={s}: {same}")
            if not same:
                print(da1.sel(S=s))
                print(da2.sel(S=s))
                all_same = False
    return all_same

def compare_slice(da1, da2):
    start = time.time()
    a1 = da1.values
    print(f'da1 took {time.time() - start}s')
    start = time.time()
    a2 = da2.values
    print(f'da2 took {time.time() - start}s')
    return np.isclose(a1, a2, equal_nan=True).all()

def compare_attrs(da1, da2):
    for a in sorted(set(da1.attrs) | set(da2.attrs)):
        print(f"  {a:30.30} {str(da1.attrs.get(a, '')):20.20} {str(da2.attrs.get(a,'')):20.20}")

def fetch(url, target=False):
    try:
        if target:
            ds = xr.open_dataset(url, decode_times=False)['target_bnds'].to_dataset(name='target_bounds')
        else:
            ds = xr.open_dataset(url, decode_times=False)
        for name, coord in ds.variables.items():
            if coord.attrs.get("calendar") == "360":
                coord.attrs["calendar"] = "360_day"
        ds = xr.decode_cf(ds)
        return ds
    except Exception as e:
        return e

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
    
    args = parser.parse_args()
    path_mapping = parse_listfile(args.listfile)
    url1 = f'{args.test_root}/{args.test_path}'
    url2 = f'{args.reference_root}/{path_mapping[args.test_path]}'
    print(url1)
    print(url2)
    all_same = compare(url1, url2)
    print(all_same)
    if all_same:
        sys.exit(0)
    else:
        sys.exit(1)
