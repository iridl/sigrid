import abc
import importlib.util
from pathlib import Path
from typing import override
import icechunk
import os
import webob
from webob.dec import wsgify
from webob.exc import HTTPNotFound
import xarray as xr

from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType


orig_root = os.environ['PYDAP_ICECHUNK_ORIGINAL_ROOT']
icechunk_root = os.environ['PYDAP_ICECHUNK_PROCESSED_ROOT']
# TODO this must be available from the pydap config already?
catalog_root = os.environ['PYDAP_CATALOG_ROOT']


class XarrayHandler(BaseHandler, abc.ABC):
    def __init__(self, name: str):
        BaseHandler.__init__(self)
        self.name = name
        with self.open() as source:
            # Try to read one element from the dataset. If we're going to fail,
            # it's better to fail here, before the response headers have been sent,
            # than to fail after sending a 200 response. Success here doesn't
            # guarantee success for the rest of the dataset, but it catches simple
            # configuration errors like misconfigured icechunk virtual chunk directory.
            # TODO maybe instead of doing an extra read here, we can find a way to defer
            # sending the headers until the first chunk has been read successfully?
            da = next(iter(source.data_vars.values()))
            da.isel({dim: 0 for dim in da.dims}).data

            # TODO populate last-modified
            # self.additional_headers.append(
            #     (
            #         "Last-modified",
            #         (
            #             formatdate(
            #                 time.mktime(time.localtime(os.stat(filepath)[ST_MTIME]))
            #             )
            #         ),
            #     )
            # )

            # shortcuts
            vars = source.variables
            dims = source.dims

            # build dataset

            self.dataset = DatasetType(
                self.name, attributes=dict(source.attrs)
            )

            # add grids
            grids = [var for var in vars if var not in dims]
            for grid in grids:
                # make dimension a fully qualifying name
                dimensions = ["/" + str(dim) for dim in vars[grid].dims]
                self.dataset[grid] = BaseType(
                    str(grid),
                    DataArrayProxy(source[grid]),
                    dims=dimensions,
                    **vars[grid].attrs,
                )

            # TODO deal with groups
            # if len(source.groups) > 0:
            #     # start at root level
            #     path = source.path
            #     for vdim in source.dimensions:
            #         fqn_dims.update({path + vdim: vdim})  # fqn is unique
            #     fqn_dims = group_fqn(self.dataset, source, self.filepath, fqn_dims)

            vdims = [dim for dim in dims if dim in vars]
            for dim in vdims:
                data = vars[dim].data
                attributes = vars[dim].attrs
                self.dataset[dim] = BaseType(str(dim), data, None, attributes)
                # TODO deal with the type error when I deal with groups and
                # understand what's intended.
                self.dataset[dim].dims = ["/" + str(dim)] # type: ignore


    @abc.abstractmethod
    def open(self) -> xr.Dataset: ...


def open_icechunk(rel_path):
    storage = icechunk.local_filesystem_storage(Path(icechunk_root) / rel_path)
    repo = icechunk.Repository.open(
        storage,
        authorize_virtual_chunk_access={f'file://{orig_root}/': None}
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, zarr_format=3, decode_times=False)
    return ds


# Initialy defined this to provide a .view method to satisfy
# tostring_with_byteorder. I think that function's use of view is
# pointless and unnecessary, so I just made a dummy method that
# returns its argument unchanged. But then it turns out that method
# never gets called. The mere fact of being an instance of an
# unknown class is sufficient to make it follow a different code
# path that works, and is fast, and doesn't even attempt to call .view.
# One difference is that pydap.handlers.lib.wrap_arrayterator has
# an isinstance check. But that's not the whole story. Just wrapping
# the DataArray in Arrayterator instead of DataArrayProxy makes the
# code work, but slow. There's probably another relevant isinstance check
# somewhere else too. TODO
class DataArrayProxy:
    def __init__(self, da: xr.DataArray):
        self._da = da

    def __getattr__(self, name: str):
        return getattr(self._da, name)

    def __getitem__(self, index):
        return self._da[index].data


def ensure_trailing(s: str) -> str:
    if s.endswith('/'):
        return s
    return f'{s}/'


class Server:
    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)

    @wsgify
    def __call__(self, req: webob.Request):
        # Path() strips trailing slashes, so extract that info first
        dir_requested = req.path_info[-1] == '/'
        # path_info looks like an absolute path. Strip the leading / to
        # make it relative.
        relpath = Path(req.path_info).relative_to('/')
        abspath = self.catalog_path / relpath
        if dir_requested:
            # TODO directory listing
            return HTTPNotFound()
        file_path = abspath.parent / 'index.py'
        if file_path.is_file:
            varname = abspath.stem
            extension = abspath.suffix
            return CatalogFileHandler(file_path, varname, extension)
        return HTTPNotFound()


class CatalogFileHandler(XarrayHandler):
    def __init__(self, file_path, varname, extension):
        self.file_path = file_path
        self.varname = varname
        self.extension = extension
        super().__init__(varname)

    @override
    def open(self):
        spec = importlib.util.spec_from_file_location('catalog', self.file_path)
        assert spec is not None  # we already checked that it exists
        module = importlib.util.module_from_spec(spec)
        # Pyright says the loader could be None, but I don't see how that
        # could happen.
        assert spec.loader  
        spec.loader.exec_module(module)
        ds: xr.Dataset = module.vars[self.varname]()
        return ds