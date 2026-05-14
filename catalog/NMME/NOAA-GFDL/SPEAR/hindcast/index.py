import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/NOAA-GFDL/SPEAR/hindcast',
        # maps icechunk names to pydap-icechunk conventional names
        # Keys can not be changed,
        # Values must correspond to icechunk names
        original_names={
            'S': 'IRIDL_time',
            'L': ['TIME', 'TIME1'],
            'Y': ['LAT', 'LAT1'],
            'X': ['LON', 'LON1'],
            'prcp': 'PRECIP_1X1',
            't2m': 'T_REF_1X1',
            'sst': 'SST_1X1',
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
        cataloging.VARS_NAMES['prcp'],
        cataloging.VARS_NAMES['t2m'],
        cataloging.VARS_NAMES['sst'],
    ]
    