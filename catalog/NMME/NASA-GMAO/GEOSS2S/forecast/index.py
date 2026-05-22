import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/NASA-GMAO/GEOSS2S/forecast',
        # maps icechunk names (keys)to pydap-icechunk conventional names (values)
        # Values can not be changed,
        # Keys must correspond to icechunk names
        original_names={
            'IRIDL_time': 'S',
            'time': 'L',
            'latitude': 'Y',
            'longitude': 'X',
            'precip': 'prcp',
            'tref': 't2m',
            'sst': 'sst',
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.NAMES['prcp'],
        cataloging.NAMES['t2m'],
        cataloging.NAMES['sst'],
    ]
