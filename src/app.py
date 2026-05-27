import os
import pydap_icechunk
app = pydap_icechunk.Server(os.environ['PYDAP_CATALOG_ROOT'])