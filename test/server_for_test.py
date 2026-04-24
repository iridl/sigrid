import logging
import os
import werkzeug

import pydap_icechunk

if __name__ == "__main__":
    logger = logging.getLogger("werkzeug")
    logger.setLevel(logging.DEBUG)
    catalog_path = os.environ['PYDAP_CATALOG_ROOT']
    assert catalog_path
    app = pydap_icechunk.Server(catalog_path)
    werkzeug.serving.run_simple(
        "localhost", 8001, app,
        use_reloader=True,
        passthrough_errors=True,
    )
