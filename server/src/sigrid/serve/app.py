import os
from pathlib import Path

import icechunk

import sigrid.harmonize
import sigrid.serve

icechunk.initialize_logs()
# This application only reads, never writes, so this warning
# is unnecessary.
icechunk.set_logs_filter('[{message="The LocalFileSystem storage is not safe for concurrent commits}]=off')

catalog_root = Path(os.environ['COOKED_CATALOG_ROOT'])
catalog = sigrid.harmonize.Catalog(catalog_root)
app = sigrid.serve.Server(catalog)
