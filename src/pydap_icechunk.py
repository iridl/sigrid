from collections import OrderedDict
from pathlib import Path
import icechunk
import os
import re
import xarray as xr

from pydap.exceptions import OpenFileError
from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType


orig_root = os.environ['PYDAP_ICECHUNK_ORIGINAL_ROOT']
icechunk_root = os.environ['PYDAP_ICECHUNK_PROCESSED_ROOT']
# TODO this must be available from the pydap config already?
catalog_root = os.environ['PYDAP_CATALOG_ROOT']


class XarrayHandler(BaseHandler):
    """A handler for files that can be opened by xarray.
    """

    extensions = re.compile(r"^.*\.ncx$", re.IGNORECASE)

    def __init__(self, filepath):
        BaseHandler.__init__(self)

        self.filepath = filepath
        try:
            with self.open() as source:
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

                # Add dimensions when creating the DatasetType
                Dims = {}
                fqn_dims = OrderedDict()  # keep track of fully qualifying names of dims
                for dim in dims:
                    fqn_dims.update({"/" + str(dim): dim})
                    if source.sizes[dim] is None: # TODO is this how xarray represents unlimited dimension?
                        self.dataset.attributes["DODS_EXTRA"] = {
                            "Unlimited_Dimension": dim,
                        }
                    else:
                        Dims.update({dim: source.sizes[dim]})
                # build dataset

                name = os.path.split(filepath)[1]
                self.dataset = DatasetType(
                    name, dimensions=Dims, attributes=dict(source.attrs)
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
        except Exception as exc:
            raise
            message = "Unable to open file %s: %s" % (filepath, exc)
            raise OpenFileError(message)

    def open(self) -> xr.Dataset:
        return xr.open_dataset(self.filepath, decode_cf=False)


class IcechunkHandler(XarrayHandler):
    """A handler for icechunk stores.
    """

    extensions = re.compile(r"^.*\.icechunk$", re.IGNORECASE)

    def open(self) -> xr.Dataset:
        assert self.filepath.endswith('.icechunk')
        assert self.filepath.startswith(ensure_trailing(catalog_root))
        rel = self.filepath[len(ensure_trailing(catalog_root)):-len('.icechunk')]
        storage = icechunk.local_filesystem_storage(Path(icechunk_root) / rel)
        repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access={f'file://{orig_root}/': None})
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, zarr_format=3, decode_cf=False)
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


if __name__ == "__main__":
    import sys

    from werkzeug.serving import run_simple

    application = XarrayHandler(sys.argv[1])
    run_simple("localhost", 8001, application, use_reloader=True)
