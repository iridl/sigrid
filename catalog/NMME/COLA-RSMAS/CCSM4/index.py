import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/COLA-RSMAS/CCSM4',
        # maps icechunk names (keys)to pydap-icechunk conventional names (values)
        # Values can not be changed,
        # Keys must correspond to icechunk names
        original_names={
            'IRIDL_time': 'S',
            'TIME': 'L',
            'LAT': 'Y',
            'LON': 'X',
            'PRECIP_1X1': 'prcp',
            'T_REF_1X1': 't2m',
            'SST_1X1': 'sst',
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['prcp'],
        cataloging.VARS_NAMES['t2m'],
        cataloging.VARS_NAMES['sst'],
    ]
