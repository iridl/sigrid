import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/ECCC/CanSIPS-IC4/forecast',
        # maps icechunk names (keys)to pydap-icechunk conventional names (values)
        # Values can not be changed,
        # Keys must correspond to icechunk names
        original_names={
            'IRIDL_time': 'S',
            'step': 'L',
            'latitude': 'Y',
            'longitude': 'X',
            'number': 'M',
            'prate': 'prcp',
            'avg_2t': 't2m',
            'avg_sst': 'sst',
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['prcp'],
        cataloging.VARS_NAMES['t2m'],
        cataloging.VARS_NAMES['sst'],
    ]
