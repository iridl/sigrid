import icechunk

import sigrid.harmonize
import sigrid.serve

icechunk.initialize_logs()
# TODO this is an important warning that needs to be addressed.
icechunk.set_logs_filter('[{message="The LocalFileSystem storage is not safe for concurrent commits}]=off')

catalog = sigrid.harmonize.Catalog()
app = sigrid.serve.Server(catalog)
