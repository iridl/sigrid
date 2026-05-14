import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='SubC/EMC/GEFSv12/forecast',
        # maps icechunk names to pydap-icechunk conventional names
        # Keys can not be changed,
        # Values must correspond to icechunk names
        original_names={
            'S': 'IRIDL_time',
            'Y': 'lat',
            'X': 'lon',
            'pr': 'PRATE_P1_L1_GLL0',
        },
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['pr'],
    ]
    