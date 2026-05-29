import abc
import importlib.util
from pathlib import Path
from typing import override
import dask.array
import icechunk
import os
import re
import jinja2
import webob
from webob.dec import wsgify
from webob.exc import HTTPForbidden, HTTPNotFound
import xarray as xr
import numpy as np

from pydap.handlers.lib import BaseHandler
from pydap.model import BaseType, DatasetType



orig_root = os.environ['PYDAP_ICECHUNK_ORIGINAL_ROOT']
icechunk_root = os.environ['PYDAP_ICECHUNK_PROCESSED_ROOT']
# TODO this must be available from the pydap config already?


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
                data = source[grid].data
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


def open_icechunk(rel_path, decode_times=True, drop_variables=None):
    abspath = Path(icechunk_root) / rel_path
    storage = icechunk.local_filesystem_storage(abspath)
    # Workaround for https://github.com/earth-mover/icechunk/issues/2105
    if not icechunk.Repository.exists(storage):
        raise Exception(f'No repository exists at {abspath}')
    try:
        repo = icechunk.Repository.open(
            storage,
            authorize_virtual_chunk_access={f'file://{orig_root}/': None}
        )
        session = repo.readonly_session("main")
        ds = xr.open_zarr(
            session.store,
            zarr_format=3,
            decode_times=decode_times,
            drop_variables=drop_variables,
        )
        return ds
    except Exception as e:
        e.add_note(f'When trying to open {abspath}')
        raise


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
    def __init__(self, catalog_path: Path | str):
        self.catalog_path = Path(catalog_path).resolve()

    @wsgify
    def __call__(self, req: webob.Request):
        # path_info looks like an absolute path. Strip the leading / to
        # make it relative.
        assert req.path_info[0] == '/'
        relpath = req.path_info[1:]

        # Check for trailing slash before Path() strips it
        is_dir = req.path_info[-1] == '/'

        abspath = self.catalog_path / relpath

        # Don't allow an attacker to escape from the catalog root
        # by using paths containing ".."
        try:
            abspath.resolve().relative_to(self.catalog_path)
        except ValueError:
            return HTTPForbidden()

        if is_dir:
            if abspath.exists():
                return self.dir(abspath)
            return HTTPNotFound()

        file_path = abspath.parent / 'index.py'
        if file_path.is_file():
            varname = abspath.stem
            extension = abspath.suffix
            return CatalogFileHandler(file_path, varname, extension)

        return HTTPNotFound()

    def dir(self, dir_path: Path):
        """Return a directory listing."""

        dirs = [
            d.name
            for d in dir_path.iterdir()
            if d.is_dir() and d.name != '__pycache__' and not (d / 'hidden').exists()
        ]
        dirs = sorted(dirs, key=alphanum_key)

        index_path = dir_path / "index.py"
        if index_path.exists():
            module = load_index(index_path)
            vars = module.list_vars()
            vars = sorted(vars, key=alphanum_key)
        else:
            vars = []
        

        context = {
            "dirs": dirs,
            "vars": vars,
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
    def __init__(self, file_path, varname, extension):
        self.file_path = file_path
        self.varname = varname
        self.extension = extension
        super().__init__(varname)

    @override
    def __call__(self, environ, start_response):
        if self.extension:
            return super().__call__(environ, start_response)
        request = webob.Request(environ)
        ds_orig = self.open()
        ds_decoded = xr.decode_cf(ds_orig)
        for name, coord in ds_decoded.coords.items():
            # For some reason, decode_cf causes aux coords (e.g. target)
            # to get unloaded, so we see "..." instead of values in the UI.
            # Fix that by explicitly reloading them.
            coord.load()
            # Xarray's CF decoding removes the units and calendar attributes.
            # Put them back for display purposes.
            if np.issubdtype(coord.dtype, np.datetime64):
                for attr in ('units', 'calendar'):
                    coord.attrs[attr] = ds_orig[name].attrs[attr]
        context = {
            'ds': ds_decoded,
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
        module = load_index(self.file_path)
        ds: xr.Dataset = module.open(self.varname)
        return ds


def load_index(file_path):
    spec = importlib.util.spec_from_file_location('catalog', file_path)
    assert spec is not None  # we already checked that it exists
    module = importlib.util.module_from_spec(spec)
    # Pyright says the loader could be None, but I don't see how that
    # could happen.
    assert spec.loader  
    spec.loader.exec_module(module)
    return module


# Vendored from pydap to avoid a spurious dependency on gunicorn
def alphanum_key(s):
    """Parse a string, returning a list of string and number chunks.

        >>> alphanum_key("z23a")
        ['z', 23, 'a']

    Useful for sorting names in a natural way.

    From http://nedbatchelder.com/blog/200712.html#e20071211T054956

    """

    def tryint(s):
        try:
            return int(s)
        except Exception:
            return s

    return [tryint(c) for c in re.split("([0-9]+)", s)]


