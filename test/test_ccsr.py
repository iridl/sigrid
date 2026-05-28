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


def test_first(proxy, server):
    print('test starts')
    path_mapping = compare.parse_listfile('../ccsr-config/test/iridl-vs-ccsr.txt')
    test_path = 'NMME/COLA-RSMAS/CCSM4/t2m'
    url1 = f'{server}/{test_path}'
    url2 = f'http://iridl.ldeo.columbia.edu/{path_mapping[test_path]}'
    assert compare.compare(url1, url2)
