import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='C3S/ECMWF/SEAS51/hindcast',
        # maps icechunk names to pydap-icechunk conventional names
        # Keys can not be changed,
        # Values must correspond to icechunk names
        original_names={
            'IRIDL_time': 'S',
            'step': 'L',
            'latitude': 'Y',
            'longitude': 'X',
            'number': 'M',
            'tprate': 'prcp',
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['prcp'],
        cataloging.VARS_NAMES['t2m'],
        cataloging.VARS_NAMES['sst'],
    ]
