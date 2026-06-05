import xarray as xr

import pydap_icechunk


def open(varname) -> xr.Dataset:
    ds = pydap_icechunk.open_icechunk(f'S2S/ECMF/CY49/REL/CF/{varname}', decode_times=False)
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

    orig_name = next(iter(ds.data_vars))
    ds = ds.rename({orig_name: varname})
    # Remove scalar values to avoid errors that cause
    #   the DataArrayProxy class -> __getitem__ in pydap_icechunk to fail,
    #   due to combinations of tuples and scalars being interpreted as variables;
    #   scalars are also not necessary
    scalar_coords = [c for c in ds.coords if ds.coords[c].ndim == 0]
    ds = ds.drop_vars(scalar_coords)
    # TODO overwrite the attrs wholesale rather than passing through what was saved in the zarr.
    ds.attrs.pop('history', None) #There are too many history messages, and they cause exceptions and warnings when the data is served.
    ds[varname] = ds[varname].assign_coords(L=('L', range(len(ds['L']))))
    return ds
