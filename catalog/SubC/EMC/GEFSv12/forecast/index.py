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
            'IRIDL_time': 'S',
            'time': 'L',
            'time_expanded': 'target',
            'lat': 'Y',
            'lon': 'X',
            'P': 'P',
            'PRATE_P1_L1_GLL0': 'pr',
            'TMP_P1_L103_GLL0': 'tas',
            'TMAX_P1_L103_GLL0': 'tasmax',
            'TMIN_P1_L103_GLL0': 'tasmin',
            'PRMSL_P1_L101_GLL0': 'psl',
            'UGRD_P1_L103_GLL0': 'uas',
            'VGRD_P1_L103_GLL0': 'vas',
            'HGT_P1_L100_GLL0': 'zg',
        },
        units={
            'P': 'hPa',
        }
    )

def list_vars():
    return [
        cataloging.NAMES['pr'],
    ]
    