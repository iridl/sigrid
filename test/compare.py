import argparse
import time

import numpy as np
import xarray as xr

def compare(url1, url2, var1=None, var2=None, check_data=False):
    ds1 = fetch(url1)
    ds2 = fetch(url2)

    if isinstance(ds1, Exception):
        print('first failed', ds1)
        return
    if isinstance(ds2, Exception):
        print('second failed', ds2)
        return
    
    if var1 is None:
        names = list(ds1.data_vars)
        assert len(names) == 1, f"Specify var1 as one of {names}"
        var1 = names[0]
    
    if var2 is None:
        names = list(ds2.data_vars)
        assert len(names) == 1, f"Specify var2 as one of {names}"
        var2 = names[0]
    
    da1 = ds1[var1]
    da2 = ds2[var2]

    compare_shape(da1, da2)
    compare_attrs(da1, da2)
    compare_coords(da1, da2)
    if check_data:
        compare_data(da1, da2)

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
     else:
         print('different dims:')
         print(dims1)
         print(dims2)

def compare_data(da1, da2):
    l = da1.sizes['S']
    for i in (0, l // 2, l - 1):
        same = compare_slice(da1.isel(S=i), da2.isel(S=i))
        print(f"S={i}: {same}")

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

def fetch(url):
    try:
        ds = xr.open_dataset(url, decode_times=False)
        for name, coord in ds.coords.items():
            if coord.attrs.get("calendar") == "360":
                coord.attrs["calendar"] = "360_day"
        ds = xr.decode_cf(ds)
        return ds
    except Exception as e:
        return e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("url1")
    parser.add_argument("url2")
    parser.add_argument("--var1")
    parser.add_argument("--var2")
    args = parser.parse_args()

    compare(args.url1, args.url2, args.var1, args.var2)
