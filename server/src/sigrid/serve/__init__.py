import webob
from webob.dec import wsgify

from sigrid.harmonize.catalog import Catalog

from . import opendap, ui


class Server:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    @wsgify
    def __call__(self, req: webob.Request):
        url_path = req.path_info
        if url_path.endswith(('.das', '.dds', '.dods')):
            return opendap.VariableHandler(self.catalog)
        return ui.CatalogBrowser(self.catalog)
