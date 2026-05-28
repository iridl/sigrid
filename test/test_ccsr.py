import subprocess
import pytest
import re

import compare
import recording_proxy


@pytest.fixture(scope="session", autouse=True)
def server():
    proc = subprocess.Popen(
        [
            # -e default usually isn't necessary, but here it is,
            # I guess because the parent process is running in a
            # different environment (client).
            "pixi", "run", "-e", "default", "python", "-c",
            'from werkzeug.serving import run_simple; from app import app; run_simple("127.0.0.1", 0, app)'
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stderr is not None

    port_line_re = re.compile(r' \* Running on http://127\.0\.0\.1:(\d+)')
    port = None
    for line in proc.stderr:
        print(line, end='')
        match = port_line_re.match(line)
        if match:
            port = int(match.group(1))
            break
    assert port is not None

    yield f'http://127.0.0.1:{port}'

    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def proxy():
    with recording_proxy.recording_proxy(
            "responses",
            is_record=False,
            prefixes=['http://iridl.ldeo.columbia.edu'],
            verbose=True,
    ):
        yield


paths = compare.parse_listfile('../ccsr-config/test/iridl-vs-ccsr.txt')

@pytest.mark.parametrize('test_path', paths.keys())
def test_one(proxy, server, test_path):
    reference_path = paths[test_path]
    url1 = f'{server}/{test_path}'
    url2 = f'http://iridl.ldeo.columbia.edu/{reference_path}'

    ds1 = compare.fetch(url1)
    ds2 = compare.fetch(url2)
    ds2 = ds2.convert_calendar('standard', dim='S', align_on='date')

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
