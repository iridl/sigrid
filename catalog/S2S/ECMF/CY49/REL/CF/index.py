import xarray as xr

import pydap_icechunk
import numpy as np

def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'S2S/ECMF/CY49/REL/CF/{varname}', decode_times=False)
    #ds[varname].attrs['coordinates'] = "valid_time"
    
    rename_map = {
        'time': 'S',
        'longitude': 'X',
        'latitude': 'Y',
        'step': 'L',
        'isobaricInhPa': 'P',
        'number': 'M',
        'valid_time': 'target',
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds})
    original_names = {
        't2m': '2t',
        'u10': '10u',
        'v10': '10v',
        'd2m': '2dt',
        }
    # setdefault is used as a check for the value of varname. 
    # If it does not exist in the dictionary, it means that both 
    # in the original file and in the DL index they have the same name, 
    # so it use the value of varname.
    ds = ds.rename({original_names.setdefault(varname, varname): varname})
    # Remove scalar values to avoid errors that cause
    #   the DataArrayProxy class -> __getitem__ in pydap_icechunk to fail,
    #   due to combinations of tuples and scalars being interpreted as variables;
    #   scalars are also not necessary
    scalar_coords = [c for c in ds.coords if ds.coords[c].ndim == 0]
    ds = ds.drop_vars(scalar_coords)
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    ds.attrs.pop('history', None) #There are too many history messages, and they cause exceptions and warnings when the data is served.
    print("Valores de L:", ds['L'].values[:5])
    print("Atributos de L:", ds['L'].attrs)

    # L should be in days 
    dic_conversion = {'days': 1, 'hours': 24, 'seconds': 86400}
    l_units = ds['L'].attrs.get('units')
    if l_units not in dic_conversion:
        raise ValueError(f"L unit not contemplated: '{l_units}'")
    # Assign the data type according to the resulting values from the conversion.
    converted = ds['L'].values / dic_conversion[l_units]
    converted = converted.astype(int) if (converted == converted.astype(int)).all() else converted

    ds = ds.assign_coords(L=('L', converted))
    ds['L'].attrs['units'] = 'days'

    #ds = ds.assign_coords(L=('L', range(ds.sizes['L'])))
    #Grid order 
    base_dims = ['S', 'L', 'Y', 'X']
    if 'P' in ds[varname].dims:
        base_dims.insert(2, 'P') 
    ds[varname] = ds[varname].transpose(*base_dims)
    #Invert Y from N-S to S-N
    if ds['Y'].values[0] > ds['Y'].values[-1]:
        ds = ds.isel(Y=slice(None, None, -1))
    #Force target as coordinate
    ds[varname].attrs['coordinates'] = "target"
    return ds
