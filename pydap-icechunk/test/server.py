import logging
import os
import werkzeug

import pydap_icechunk.app

if __name__ == "__main__":
    logger = logging.getLogger("werkzeug")
    logger.setLevel(logging.DEBUG)
    werkzeug.serving.run_simple(
        "localhost",
        int(os.environ['PYDAP_PORT']),
        pydap_icechunk.app.app,
        #use_reloader=True,
        passthrough_errors=True,
    )
