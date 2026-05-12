import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/NASA-GMAO/GEOSS2S/forecast',
        # maps icechunk names to pydap-icechunk conventional names
        # Keys can not be changed,
        # Values must correspond to icechunk names
        original_names={
            'S': 'IRIDL_time',
            'L': 'time',
            'Y': 'latitude',
            'X': 'longitude',
            'prec': 'precip',
            'tref': 'tref',
            'sst': 'sst',
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['prec'],
        cataloging.VARS_NAMES['tref'],
        cataloging.VARS_NAMES['sst'],
    ]
