import xarray as xr

import pydap_icechunk


COORDS_NAMES = {
    'start': 'S',
    'lead': 'L',
    'latitude': 'Y',
    'longitude': 'X',
    'member': 'M',
    'forecast_time': 'target',
}
VARS_NAMES = {
    'precipitation': 'prec',
    'temperature': 'tref',
    'sea_surface_temperature': 'sst',
}
ENCODING = {
    'cf_units': 'hours since 1960-01-01',
    'calendar': 'standard',
}


def encode_time(
    ds,
    cf_catalogue=ENCODING['calendar'],
    cf_units=ENCODING['cf_units'],
):
    time_coords = [
        coord for coord in ds.coords
        if ds[coord].dtype in ['datetime64[ns]', 'timedelta64[ns]']
    ]
    for tc in time_coords:
        data, units, calendar = xr.coding.times.encode_cf_datetime(
            ds[tc], cf_units, cf_catalogue
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


def catalogue(
    varname,
    varpath,
    original_names,
    drop_variables=None,
    del_ds_attrs=None,
    lead_is_month=False,
    ):
    ds = pydap_icechunk.open_icechunk(
        f'{varpath}/{varname}',
        drop_variables=drop_variables,
    )
    # Some varnames have scalar coordinates that break
    ds = ds.drop_vars(
        [name for name, coord in ds.coords.items() if coord.dims == ()]
    )
    # Renaming
    for var in original_names:
        if var in COORDS_NAMES:
            ds = ds.rename({original_names[var]: COORDS_NAMES[var]})
        if var in VARS_NAMES and VARS_NAMES[var] == varname:
            ds = ds.rename({original_names[var]: varname})
    # Deleting buggy attributes
    for attr in del_ds_attrs:
        del ds.attrs[attr]
    # Set lead times
    ds = ds.assign_coords({
        COORDS_NAMES['lead']: range(ds.sizes[COORDS_NAMES['lead']])
    })
    if lead_is_month:
        # Set target
        ds = ds.assign_coords({
            COORDS_NAMES["forecast_time"]: S_L_to_target(
                ds[COORDS_NAMES['start']], ds[COORDS_NAMES['lead']]
            )
        })
    # Encode time
    ds = encode_time(ds)
    # Force into coords
    ds[varname].attrs['coordinates'] = COORDS_NAMES["forecast_time"]
    return ds
