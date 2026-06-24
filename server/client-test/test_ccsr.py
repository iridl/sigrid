import os
from pathlib import Path

import pytest

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

    # Cut them both off so I don't have tests failing or busting the response cache
    # because of updates. If we want to check for successful updates, we'll need
    # to write a separate set of tests for that.
    ds1 = ds1.sel(S=slice(None, '2026-05-01'))
    ds2 = ds2.sel(S=slice(None, '2026-05-01'))

    # Convert Ingrid's 360_day calendar to standard
    ds2 = ds2.convert_calendar('standard', dim='S', align_on='date')

    # Ingrid's L is to the midpoint of the month, pydap's is to the start.
    ds2['L'] = ds2['L'] - 0.5

    # CanSIPS t2m and sst differ in the fifth decimal place.
    # TODO why?
    if (
        'CanSIPS-IC4' in test_path and
        ('sst' in test_path or 't2m' in test_path)
    ):
        atol = 1e-4
    else:
        atol=1e-8  # numpy's default

    # CCSM4 sst has a first S value that's non-contiguous with the rest. Ingrid
    # drops it, so drop it from pydap before comparing.
    if test_path == 'NMME/COLA-RSMAS/CCSM4/sst':
        ds1 = ds1.isel(S=slice(1, None))

    # SPEAR forecasts are missing some starts. Ingrid has a regular grid with
    # NaNs filled in for the missing forecasts, while pydap has an irregular
    # coordinate with no entries for the missing forecasts.
    if 'NMME/NOAA-GFDL/SPEAR/forecast/' in test_path:
        assert ds1.sizes['S'] >= 58
        ds2 = ds2.sel(S=ds1['S'])

    assert compare.compare_ds(ds1, ds2, atol)
