# This is the index.py file for the hindcast section of my-model. As
# the lowest-level file in the hierarchy, it defines the functions
# open(varname) and list_vars().

import sigrid.harmonize as harmonize

# Maps user-visible variable name (the ensemble's standard name) to a path
# component that is used in the file path below.
store_names = {
    'prcp': 'precip',
}

# list_vars returns a list of strings that are valid values for the
# varname argument of open (see below).
def list_vars():
    return store_names.keys()

# open(varname) returns an xarray.Dataset representing the contents of
# the requested variable. In this example, it retrieves those contents
# from an icechunk store.
def open(varname):
    # The file path is relative to the value of the environment variable
    # ICECHUNK_ROOT
    return harmonize.open_icechunk(f'my-MME/my-model/hindcast/{store_names[varname]}')

