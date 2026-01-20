import os
import werkzeug

import pydap.wsgi.app

if __name__ == "__main__":
    path = os.environ['PYDAP_CATALOG_ROOT']
    assert path
    app = pydap.wsgi.app.DapServer(path)
    werkzeug.serving.run_simple(
        "localhost", 8001, app,
        # use_reloader=True,
        passthrough_errors=True,
    )
