import xarray as xr

import cataloging


def open(varname) -> xr.Dataset:
    return cataloging.catalog(
        varname,
        varpath='NMME/NOAA-GFDL/SPEAR/forecast',
        # maps icechunk names to pydap-icechunk conventional names
        # Keys can not be changed,
        # Values must correspond to icechunk names
        original_names={
            'S': 'IRIDL_time',
            'L': 'TIME',
            'Y': ['LAT', 'LAT1'],
            'X': ['LON', 'LON1'],
            'prec': 'PRECIP_1X1',
            'tref': 'T_REF_1X1',
            'sst': 'SST_1X1',
        },
        missing_units={
            'prec': 'mm/s',
            'tref': 'K',
            'sst': 'degree_Celsius'
        },
        lead_is_month=True,
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['prec'],
        cataloging.VARS_NAMES['tref'],
        cataloging.VARS_NAMES['sst'],
    ]
    