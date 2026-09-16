import os
import sys
from pathlib import Path

import sigrid.harmonize

if __name__ == '__main__':
    var = sys.argv[1]
    catalog_root = Path(os.environ['COOKED_CATALOG_ROOT'])
    catalog = sigrid.harmonize.Catalog(catalog_root)
    ds = catalog.open_variable(var)
    print(ds)