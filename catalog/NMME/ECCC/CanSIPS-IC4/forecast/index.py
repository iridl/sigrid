import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
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
    return [
        cataloging.VARS_NAMES['prec'],
        cataloging.VARS_NAMES['tref'],
        cataloging.VARS_NAMES['sst'],
    ]
