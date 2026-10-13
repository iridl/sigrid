import datetime
import os
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray.coding.times

import compare


# TODO yuck
listfile = Path(os.environ['COOKED_CATALOG_ROOT']).parent.parent / 'test/iridl-vs-ccsr.txt'
paths = compare.parse_listfile(listfile)

@pytest.mark.parametrize('test_path', paths.keys())
def test_one(proxy, server, test_path):
    reference_path = paths[test_path]
    url1 = f'{server.url}/{test_path}'
    url2 = f'http://iridl.ldeo.columbia.edu/{reference_path}'

    ds1 = compare.fetch(url1)
    ds2 = compare.fetch(url2)

    if 'S' in ds1.coords:
        time_dim = 'S'
    elif 'T' in ds1.coords:
        time_dim = 'T'
    else:
        assert False

    if 'L' in ds1:
        # Forecasts: Ingrid's L is to the midpoint of the month, pydap's is to
        # the start.
        if not np.issubdtype(ds1['L'].dtype, np.timedelta64) :
            ds2['L'] = ds2['L'] - 0.5
    else:
        # Obs
        if isinstance(ds2['T'].values[0], cftime.Datetime360Day):
            # Ingrid's T is to the midpoint of the month, pydap's is to the
            # start.
            ds2['T'] = ds2['T'] - datetime.timedelta(days=15) # pyright: ignore[reportOperatorIssue] (xarray stubs are wrong)
        elif ds2['T'].units == 'julian_day':
            ds2['T'] = (
                pd.to_datetime(ds2['T'].values, unit="D", origin="julian") -
                pd.Timedelta(hours=12)  # julian_day origin is noon
            )
        else:
            assert False  # Figure out other cases if and when we see them.

    # Convert Ingrid's 360_day calendar to standard
    ds2 = ds2.convert_calendar('standard', dim=time_dim, align_on='date')

    # Some S2S models in IRIDL combine different versions into a single dataset,
    # whereas PyDAP keeps them separate. Therefore, S in PyDAP could be smaller 
    # than in IRIDL.
    if 'S2S/ECMF/' in test_path :
        ds2 = ds2.sel({time_dim: ds1[time_dim]})
    # Cut them both off so I don't have tests failing or busting the response cache
    # because of updates. If we want to check for successful updates, we'll need
    # to write a separate set of tests for that.
    else:
        ds1 = ds1.sel({time_dim: slice(None, '2026-05-01')})
        ds2 = ds2.sel({time_dim: slice(None, '2026-05-01')})


    # IRIDL uses g2clib for GRIB decoding, while sigrid uses ecCodes. The two
    # libraries use different implementations of the GRIB bit-(un)packing
    # formula that can yield slighly different floating point numbers when
    # reading the same file. The magnitude of the difference between Ingrid and
    # Sigrid depends on the scaling factors used to encode each variable.
    temperature_vars = ('tos', 'tas', 'tasmax', 'tasmin')

    if (
        ('CanSIPS-IC4' in test_path or 'CFSv2' in test_path) and
        any(test_path.endswith(f'/{suff}') for suff in temperature_vars)
    ):
        atol = 1e-4
    else:
        atol=1e-8  # numpy's default

    # CCSM4 sst has a first S value that's non-contiguous with the rest. Ingrid
    # drops it, so drop it from pydap before comparing.
    if test_path == 'NMME/COLA-RSMAS/CCSM4/tos':
        ds1 = ds1.isel(S=slice(1, None))

    # SPEAR forecasts are missing some starts. Ingrid has a regular grid with
    # NaNs filled in for the missing forecasts, while pydap has an irregular
    # coordinate with no entries for the missing forecasts.
    if 'NMME/NOAA-GFDL/SPEAR/forecast/' in test_path:
        assert ds1.sizes['S'] >= 58
        ds2 = ds2.sel(S=ds1['S'])

    # Ingrid CFSv2 zg has only 200 hPa, while I added a couple other levels to
    # sigrid.
    if 'CFSv2' in test_path and 'zg' in test_path:
        ds1 = ds1.sel(P=[200])

    assert compare.compare_ds(ds1, ds2, time_dim, atol)
