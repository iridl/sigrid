import xarray as xr

import cataloguing


def open(varname) -> xr.Dataset:
    return cataloguing.catalogue(
        varname,
        varpath='NMME/ECCC/CanSIPS-IC4/forecast',
        original_names={
            'S': 'IRIDL_time',
            'L': 'step',
            'Y': 'latitude',
            'X': 'longitude',
            'M': 'number',
            'prec': 'prate',
            'tref': 'avg_2t',
            'sst': 'avg_sst',
        },
        #drop_variables=["valid_time_expanded", "time"],
        del_ds_attrs=['history'],
        lead_is_month=True,
    )

def list_vars():
    return ['prec', 'tref', 'sst']
