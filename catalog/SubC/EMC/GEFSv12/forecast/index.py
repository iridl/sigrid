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
            'L': 'time',
            'target': 'time_expanded',
            'Y': 'lat',
            'X': 'lon',
            'P': 'P',
            'pr': 'PRATE_P1_L1_GLL0',
            'tas': 'TMP_P1_L103_GLL0',
            'tasmax': 'TMAX_P1_L103_GLL0',
            'tasmin': 'TMIN_P1_L103_GLL0',
            'psl': 'PRMSL_P1_L101_GLL0',
            'uas': 'UGRD_P1_L103_GLL0',
            'vas': 'VGRD_P1_L103_GLL0',
            'zg': 'HGT_P1_L100_GLL0',
        },
        units={
            'P': 'hPa',
        }
    )

def list_vars():
    return [
        cataloging.VARS_NAMES['pr'],
    ]
    