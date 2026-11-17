import warnings
from collections.abc import Iterable, Mapping
from typing import cast

import dask.array
import numpy as np
import webob
import xarray as xr
import xarray.coding.common
import xarray.conventions
from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType
from webob.dec import wsgify
from webob.exc import HTTPNotFound

from sigrid.harmonize import Coords


class DatasetHandler:
    """Uses a harmonize.Catalog to find the harmonize.Dataset corresponding to
    the request URL, and generates an OPeNDAP response for it.
    """

    def __init__(self, catalog):
        self.catalog = catalog

    @wsgify
    def __call__(self, request: webob.Request):
        url_path = request.path_info
        parent_path, last_component = url_path.rsplit('/', maxsplit=1)
        if not last_component.endswith(('.das', '.dds', '.dods')):
            return HTTPNotFound()
        var_name, _ = last_component.rsplit('.', maxsplit=1)
        ds = self.catalog.open_dataset(f'{parent_path}/{var_name}')

        if ds is None:
            return HTTPNotFound()

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

        # Populate data variables' "coordinates" attribute, otherwise variables
        # that xarray considers to be auxiliary vars turn into data vars
        # in the OPeNDAP response.
        for da in ds.data_vars.values():
            aux_coords = set(da.attrs.get('coordinates', '').split())
            if Coords.S in da.dims:
                aux_coords |= {Coords.target, Coords.target_bnds}
            if aux_coords:
                da.attrs['coordinates'] = ' '.join(aux_coords)

        return XarrayHandler(var_name, ds)


class XarrayHandler(BaseHandler):
    """Translates an xarray Dataset to a pydap Dataset, and (by inheriting from
    BaseHandler) translates that pydap Dataset to an OPeNDAP response.
    """
    def __init__(self, name: str, xr_ds: xr.Dataset):
        BaseHandler.__init__(self)
        # Try to read one element from the dataset. If we're going to fail,
        # it's better to fail here, before the response headers have been sent,
        # than to fail after sending a 200 response. Success here doesn't
        # guarantee success for the rest of the dataset, but it catches simple
        # configuration errors like misconfigured icechunk virtual chunk directory.
        # TODO maybe instead of doing an extra read here, we can find a way to defer
        # sending the headers until the first chunk has been read successfully?
        da = next(iter(xr_ds.data_vars.values()))
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

        assert all(isinstance(d, str) for d in xr_ds.dims)
        dims = cast(Iterable[str], xr_ds.dims)

        # Apply cf-encoding.
        # Check for NaNs in floats that will be converted to ints. If there
        # are none, add a _FillValue to suppress an xarray warning that's
        # raised regardless of whether NaNs are present or not.
        for cname, coord in xr_ds.coords.items():
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
                xarray.conventions.cf_encoder(xr_ds.variables, xr_ds.attrs)
            )

        # Build dataset. BaseHandler.__call__ expects to find this on self.

        self.dataset = DatasetType(
            name, attributes=attrs
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
