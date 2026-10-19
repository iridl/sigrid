import abc
import warnings
from collections.abc import Iterable, Mapping
from typing import cast, override

import dask.array
import numpy as np
import webob
import xarray as xr
import xarray.coding.common
import xarray.conventions
from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType
from webob.dec import wsgify
from webob.exc import HTTPFound, HTTPNotFound

from sigrid import harmonize
from sigrid.harmonize import Coords
from sigrid.serve import pages


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
            da.isel({dim: 0 for dim in da.dims}).data  # noqa: B018

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

            assert all(isinstance(d, str) for d in source.dims)
            dims = cast(Iterable[str], source.dims)

            # Apply cf-encoding.
            # Check for NaNs in floats that will be converted to ints. If there
            # are none, add a _FillValue to suppress an xarray warning that's
            # raised regardless of whether NaNs are present or not.
            for cname, coord in source.coords.items():
                if (
                    np.issubdtype(coord.dtype, np.floating) and
                    np.issubdtype(coord.encoding.get('dtype'), np.integer) and
                    '_FillValue' not in coord.encoding and
                    coord.isnull().any()
                ):
                    raise Exception(f"Coordinate {cname} can't be encoded as {coord.encoding['dtype']} because it contains NaNs.")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.* floating point data as an integer dtype without any _FillValue .*', xarray.coding.common.SerializationWarning)
                vars, attrs = cast(
                    tuple[Mapping[str, xr.Variable], Mapping[str, str]],
                    xarray.conventions.cf_encoder(source.variables, source.attrs)
                )

            # build dataset

            self.dataset = DatasetType(
                self.name, attributes=attrs
            )

            # add grids
            grids = [var for var in vars if var not in dims]
            for grid in grids:
                # make dimension a fully qualifying name
                dimensions = ["/" + str(dim) for dim in vars[grid].dims]
                data = vars[grid].data
                if isinstance(data, dask.array.Array):
                    data = DaskArrayProxy(data)
                self.dataset[grid] = BaseType(
                    str(grid),
                    data,
                    dims=dimensions,
                    **vars[grid].attrs,
                )

            vdims = [dim for dim in dims if dim in vars]
            for dim in vdims:
                data = vars[dim].data
                attributes = vars[dim].attrs
                self.dataset[dim] = BaseType(str(dim), data, None, attributes)
                # TODO deal with the type error when I deal with groups and
                # understand what's intended.
                self.dataset[dim].dims = ["/" + str(dim)]  # pyright: ignore[reportAttributeAccessIssue]

    @abc.abstractmethod
    def open(self) -> xr.Dataset: ...


class DaskArrayProxy:
    """Adds a tobytes method to dask Array so that pydap can serialize it"""
    def __init__(self, arr: dask.array.Array):
        self._arr = arr

    def __getattr__(self, name: str):
        return getattr(self._arr, name)

    def __getitem__(self, index):
        return self.__class__(self._arr[index])

    def astype(self, t):
        return self.__class__(self._arr.astype(t))

    def view(self, x):
        return self.__class__(self._arr.view(x)) # pyright: ignore[reportAttributeAccessIssue] Missing from type defs?

    def tobytes(self):
        arr = self._arr.compute()
        bytes = arr.tobytes()
        return bytes


def ensure_trailing(s: str) -> str:
    if s.endswith('/'):
        return s
    return f'{s}/'


class Server:
    def __init__(self, catalog: harmonize.Catalog) -> None:
        self.catalog = catalog

    @wsgify
    def __call__(self, req: webob.Request):
        # Note: req.path is the full URL path, including the wsgi app mount point
        # (e.g. /data), while req.path_info is the part of the path after the
        # mount point.

        url_path = req.path_info
        assert url_path.startswith('/')

        if url_path.endswith('/'):
            listings = self.catalog.open_dataset_path(url_path)
            if listings is None:
                return HTTPNotFound()
            return self.dir(req.script_name, path_components(url_path), listings)
        else:
            parent_path, last_component = url_path.rsplit('/', maxsplit=1)
            if '.' in last_component:
                varname, extension = last_component.rsplit('.', maxsplit=1)
            else:
                varname = last_component
                extension = None
            catalog_path = f'{parent_path}/{varname}'
            ds = self.catalog.open_variable(catalog_path)
            if ds is None:
                # See if it's actually a dir and they just forgot the trailing slash
                if self.catalog.open_dataset(catalog_path + '/') is not None:
                    return HTTPFound(location=req.path + '/')
                return HTTPNotFound()

            # The web page needs the listings along the variable's path to
            # draw the catalog tree; OPeNDAP responses don't.
            page = None
            if extension is None:
                listings = self.catalog.open_dataset_path(parent_path + '/')
                assert listings is not None  # open_variable found the variable
                page = (req.script_name, path_components(parent_path + '/'), listings)
            return CatalogFileHandler(ds, varname, extension, page)  # TODO bad name--only handles variables.

    def dir(
        self,
        mount: str,
        components: list[str],
        listings: list[harmonize.DisplayDataset],
    ):
        """Return a directory listing."""
        return webob.Response(text=pages.render_dir(mount, components, listings))


def path_components(dataset_path: str) -> list[str]:
    """Split a dataset path such as /SubC/ECCC/ into ['SubC', 'ECCC']."""
    return [c for c in dataset_path.split('/') if c]


class CatalogFileHandler(XarrayHandler):
    def __init__(self, ds, varname, extension, page=None):
        # Populate data variables' "coordinates" attribute, otherwise variables
        # that xarray considers to be auxiliary vars turn into data vars
        # in the OPeNDAP response. Doing this here so that the attribute is
        # present both in OPeNDAP responses and in the web UI.
        for da in ds.data_vars.values():
            aux_coords = set(da.attrs.get('coordinates', '').split())
            if Coords.S in da.dims:
                aux_coords |= {Coords.target, Coords.target_bnds}
            if aux_coords:
                da.attrs['coordinates'] = ' '.join(aux_coords)

        self.ds = ds
        self.extension = extension
        # (mount, components, listings) for the web page; see Server.__call__.
        self.page = page
        super().__init__(varname)

    @override
    def __call__(self, environ, start_response):
        if self.extension:
            return super().__call__(environ, start_response)
        request = webob.Request(environ)

        # To help users understand what they will get via opendap, add the units
        # and calendar attributes that the response will have when datetimes
        # get cf-encoded.
        for coord in self.ds.coords.values():
            if np.issubdtype(coord.dtype, np.datetime64):
                coord.attrs['units'] = coord.encoding['units']
                coord.attrs['calendar'] = coord.encoding['calendar']
            elif np.issubdtype(coord.dtype, np.timedelta64) and 'units' in coord.encoding:
                # e.g. lead L, sent as integer hours
                coord.attrs['units'] = coord.encoding['units']

        assert self.page is not None
        mount, components, listings = self.page
        body = pages.render_var(
            mount, components, listings, self.name, self.ds, request.path_url,
        )
        response = webob.Response(text=body)
        return response(environ, start_response)

    @override
    def open(self):
        ds = self.ds.copy()

        # Scalar coordinates break pydap. TODO fix pydap.
        ds = ds.drop_vars(
            [name for name, coord in ds.coords.items() if coord.dims == ()]
        )

        # Attributes that contain quotes cause pydap to produce an invalid response.
        # TODO fix pydap, or at least figure out how to escape quotes before
        # passing them to pydap.
        ds.attrs = {
            k: v
            for k, v in ds.attrs.items()
            if not isinstance(v, str) or '"' not in v
        }

        return ds
