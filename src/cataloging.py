import xarray as xr

import pydap_icechunk


# Change the dictionary values should you want different names throughout your system
COORDS_NAMES = {
    'S': 'S',
    'L': 'L',
    'Y': 'Y',
    'X': 'X',
    'M': 'M',
    'target': 'target',
}
VARS_NAMES = {
    'prec': 'prec',
    'tref': 'tref',
    'sst': 'sst',
}
# Change the dictionary values 
# should you different time encoding throughout your system
ENCODING = {
    'cf_units': 'hours since 1960-01-01',
    'calendar': 'standard',
}


def encode_time(
    ds,
    cf_catalog=ENCODING['calendar'],
    cf_units=ENCODING['cf_units'],
):
    time_coords = [
        coord for coord in ds.coords
        if ds[coord].dtype in ['datetime64[ns]', 'timedelta64[ns]']
    ]
    for tc in time_coords:
        data, units, calendar = xr.coding.times.encode_cf_datetime(
            ds[tc], cf_units, cf_catalog
        )
        ds = ds.assign_coords({tc: (ds[tc].dims, data)})
        ds[tc].attrs['units'] = units
        ds[tc].attrs['calendar'] = calendar
    return ds


def S_L_to_target(S, L):
    return xr.DataArray(
        data=[
            xr.date_range(
                start=s.item(),
                # TODO may not be wise to simply rely on len(L)
                periods=len(L),
                freq='MS',
            )
            for s in S
        ],
        coords=dict(S=S, L=L),
        attrs={'long_name': 'target date'},
    )


def catalog(
    varname,
    varpath,
    original_names,
    #drop_variables=None,
    del_ds_attrs=None,
    lead_is_month=False,
    ):
    ds = pydap_icechunk.open_icechunk(
        f'{varpath}/{varname}',
        #drop_variables=drop_variables,
    )
    # Some varnames have scalar coordinates that break pydap
    ds = ds.drop_vars(
        [name for name, coord in ds.coords.items() if coord.dims == ()]
    )
    # Renaming
    for var in original_names:
        if var in COORDS_NAMES:
            ds = ds.rename({original_names[var]: COORDS_NAMES[var]})
        if var in VARS_NAMES and VARS_NAMES[var] == varname:
            ds = ds.rename({original_names[var]: varname})
    # Drop coords not standard
    ds = ds.drop_vars(
        [
            name
            for name, coord in ds.coords.items()
            if name not in COORDS_NAMES.values()
        ]
    )
    # Deleting buggy attributes
    for attr in del_ds_attrs:
        del ds.attrs[attr]
    if lead_is_month:
        # Set lead times
        ds = ds.assign_coords({
            COORDS_NAMES['L']: range(ds.sizes[COORDS_NAMES['L']])
        })
        # Set target
        ds = ds.assign_coords({
            COORDS_NAMES["target"]: S_L_to_target(
                ds[COORDS_NAMES['S']], ds[COORDS_NAMES['L']]
            )
        })
    # Encode time
    ds = encode_time(ds)
    # Force into coords
    ds[varname].attrs['coordinates'] = COORDS_NAMES["target"]
    return ds
