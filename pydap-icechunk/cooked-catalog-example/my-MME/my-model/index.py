# This is the index.py file for one model within the my-MME
# ensemble. Its transform function maps provider-specific variable
# names to the ensemble-standard names expected by the parent
# directory's transform function.

# This transformation could also have been performed in
# hindcast/index.py.  However, if you want to perform the same
# transformation on both hindcast and forecast variables, it is more
# convenient and less error-prone to define it once here, rather than
# duplicating it in the hindcast and forecast directories.

from cataloging import rename


var_names = {
    'IRIDL_time': 'S',
    'step': 'L',
    'latitude': 'Y',
    'longitude': 'X',
    'prate': 'prcp',
}

# The input value is the xarray.Dataset returned by the open()
# function in hindcast/index.py.
def transform(ds):
    ds = rename(ds, var_names)
    return(ds)
