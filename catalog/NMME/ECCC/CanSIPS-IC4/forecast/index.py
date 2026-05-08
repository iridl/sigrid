import xarray as xr

import cataloguing


def open(varname) -> xr.Dataset:
    return cataloguing.catalogue(
        varname,
        varpath='NMME/ECCC/CanSIPS-IC4/forecast',
        original_names={
            'start': 'IRIDL_time',
            'lead': 'step',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'member': 'number',
            'precipitation': 'prate',
            'temperature': 'avg_2t',
            'sea_surface_temperature': 'avg_sst',
        },
        drop_variables=["valid_time_expanded", "time"],
        del_ds_attrs=['history'],
        lead_is_month=True,
    )

def list_vars():
    return ['prec', 'tref', 'sst']
