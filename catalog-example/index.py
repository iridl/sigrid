# This is the top-level index.py file. Its transform function is
# applied to every variable in the catalog. In this example, it adds a
# new attribute to every dataset.

def transform(ds):
    ds.attrs['Distributor']: 'my-server.example.com'
