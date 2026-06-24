import os
import re
import subprocess
import threading
from dataclasses import dataclass, field

import pytest

import recording_proxy


def pytest_addoption(parser):
    # Add a custom command-line flag
    parser.addoption(
        "--record",
        action="store_true",
        help="Make live requests to ingrid and save the responses to replay later"
    )


@dataclass
class ServerHandle:
    url: str
    # Lines the server has written to stderr (including request logs and
    # tracebacks). Appended to by a background thread; the makereport hook reads
    # it to attach the server's output to failing tests.
    log: list[str] = field(default_factory=list)


@pytest.fixture(scope="session", autouse=True)
def server():
    log: list[str] = []
    proc = subprocess.Popen(
        [
            # -e default usually isn't necessary, but here it is,
            # I guess because the parent process is running in a
            # different environment (client).
            "pixi", "run", "-e", "default", "python", "-c",
            'from werkzeug.serving import run_simple; from sigrid.serve.app import app; run_simple("127.0.0.1", 0, app)'
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stderr is not None

    port_line_re = re.compile(r' \* Running on http://127\.0\.0\.1:(\d+)')
    port = None
    for line in proc.stderr:
        log.append(line)
        print(line, end='')
        match = port_line_re.match(line)
        if match:
            port = int(match.group(1))
            break
    assert port is not None

    # Keep draining stderr in the background. Without this, the server's
    # output after startup (including tracebacks) is never read, the pipe
    # buffer can eventually fill and block the server, and we lose the
    # diagnostics that make failures debuggable.
    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            log.append(line)

    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()

    yield ServerHandle(url=f'http://127.0.0.1:{port}', log=log)

    proc.terminate()
    proc.wait()
    drain_thread.join(timeout=5)


@pytest.fixture(scope="session")
def proxy(request):
    is_record = request.config.getoption('--record')
    response_dir = os.environ.get('RECORDING_PROXY_DIR', "recorded-responses")
    with recording_proxy.recording_proxy(
            response_dir,
            is_record=is_record,
            prefixes=['http://iridl.ldeo.columbia.edu'],
            verbose=True,
    ):
        yield


@pytest.fixture(autouse=True)
def _reset_server_log(server):
    server.log.clear()
    yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item):
    # Attach the server's stderr (request logs and, crucially, tracebacks)
    # produced during a failing test to its report. The server runs in a
    # separate process, so its stack isn't part of the test's stack.
    report = yield
    if report.when == "call" and report.failed:
        server = item.funcargs.get("server")
        if server is not None:
            output = "".join(server.log[:])
            if output.strip():
                report.sections.append(
                    ("Captured server log (background process)", output)
                )
    return report
