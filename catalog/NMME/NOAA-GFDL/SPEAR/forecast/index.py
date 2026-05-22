import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/NOAA-GFDL/SPEAR/forecast',
        # maps icechunk names (keys)to pydap-icechunk conventional names (values)
        # Values can not be changed,
        # Keys must correspond to icechunk names
        original_names={
            'IRIDL_time': 'S',
            'TIME': 'L',
            'LAT': 'Y',
            'LAT1': 'Y',
            'LON': 'X',
            'LON1': 'X',
            'PRECIP_1X1': 'prcp',
            'T_REF_1X1': 't2m',
            'SST_1X1': 'sst',
        },
        units={
            'prcp': 'mm/s',
            't2m': 'K',
            'sst': 'degree_Celsius'
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.NAMES['prcp'],
        cataloging.NAMES['t2m'],
        cataloging.NAMES['sst'],
    ]
    