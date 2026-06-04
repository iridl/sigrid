import abc
from typing import Iterable, Mapping, cast, override

import dask.array
import jinja2
import numpy as np
from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType
import webob
from webob.dec import wsgify
from webob.exc import HTTPFound, HTTPNotFound
import xarray as xr
import xarray.conventions

import cataloging


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

            assert all(isinstance(d, str) for d in source.dims)
            dims = cast(Iterable[str], source.dims)

            # Apply cf-encoding
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
                self.dataset[dim].dims = ["/" + str(dim)] # type: ignore
    
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
    def __init__(self, catalog: cataloging.Catalog) -> None:
        self.catalog = catalog

    @wsgify
    def __call__(self, req: webob.Request):
        # Note: req.path is the full URL path, including the wsgi app mount point
        # (e.g. /data), while req.path_info is the part of the path after the
        # mount point.

        url_path = req.path_info
        assert url_path.startswith('/')

        if url_path.endswith('/'):
            node = self.catalog.open_dataset(url_path)
            if node is None:
                return HTTPNotFound()
            return self.dir(node)
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
                    return HTTPFound(location=catalog_path + '/')
                return HTTPNotFound()
            return CatalogFileHandler(ds, varname, extension)  # TODO bad name--only handles variables.

    def dir(self, dataset: cataloging.Dataset):
        """Return a directory listing."""
       

        context = {
            "dirs": [name for name, sub in dataset.subdatasets.items() if not sub.hidden],
            "vars": list(dataset.variables),
        }

        return webob.Response(
            body=self.dir_template.render(context),
        )

    dir_template = jinja2.Template("""
        <!-- OPeNDAP -->
        {% if dirs %}
        <h1>Datasets</h1>
        <table>
            <tbody>
                {% for x in dirs %}
                <tr><td><a href="{{ x }}/">{{ x }}</a></td></tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
        {% if vars %}
        <h1>Variables</h1>
        <table>
            <tbody>
                {% for x in vars %}
                <tr><td><a href="{{ x }}">{{ x }}</a></td></tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    """)


class CatalogFileHandler(XarrayHandler):
    def __init__(self, ds, varname, extension):
        self.ds = ds
        self.extension = extension
        super().__init__(varname)

    @override
    def __call__(self, environ, start_response):
        if self.extension:
            return super().__call__(environ, start_response)
        request = webob.Request(environ)

        # To help users understand what they will get via opendap, add the units
        # and calendar attributes that the response will have.
        for name, coord in self.ds.coords.items():
            if np.issubdtype(coord.dtype, np.datetime64):
                coord.attrs['units'] = coord.encoding['units']
                coord.attrs['calendar'] = coord.encoding['calendar']

        context = {
            'ds': self.ds,
            'url': request.url,
        }
        response = webob.Response(body=self.var_template.render(context))
        return response(environ, start_response)

    
    var_template = jinja2.Template(
        """
        <html><body> 
            <script>
                function copyURL() {
                    const pageUrl = window.location.href;
                    navigator.clipboard.writeText(pageUrl).then(() => {
                        const btn = document.getElementById('copyButton');
                        btn.innerText = "URL Copied";
                        setTimeout(() => btn.innerText = "Copy OPeNDAP URL", 2000);
                    }).catch(err => {
                        console.error('Failed to copy opendap URL: ', err);
                    });
                }
            </script>
        <button onclick="copyURL()" id="copyButton">Copy OPeNDAP URL</button>
        {{ds._repr_html_()}}
        </body><html>
        """
    )

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
