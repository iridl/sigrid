import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/NASA-GMAO/GEOSS2S/hindcast',
        # maps icechunk names to pydap-icechunk conventional names
        # Keys can not be changed,
        # Values must correspond to icechunk names
        original_names={
            'S': 'IRIDL_time',
            'L': 'time',
            'Y': 'latitude',
            'X': 'longitude',
            'prcp': 'precip',
            't2m': 't2m',
            'sst': 'sst',
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['prcp'],
        cataloging.VARS_NAMES['t2m'],
        cataloging.VARS_NAMES['sst'],
    ]
