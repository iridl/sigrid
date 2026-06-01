import logging
import os
import werkzeug

import app

if __name__ == "__main__":
    logger = logging.getLogger("werkzeug")
    logger.setLevel(logging.DEBUG)
    werkzeug.serving.run_simple(
        "localhost",
        int(os.environ['PYDAP_PORT']),
        app.app,
        #use_reloader=True,
        passthrough_errors=True,
    )
