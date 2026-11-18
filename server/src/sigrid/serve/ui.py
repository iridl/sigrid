import jinja2
import numpy as np
import webob
import xarray as xr
from webob.dec import wsgify
from webob.exc import HTTPFound, HTTPNotFound

from sigrid.harmonize import Coords, catalog


class CatalogBrowser:
    def __init__(self, catalog: catalog.Catalog) -> None:
        self.catalog = catalog

    @wsgify
    def __call__(self, request: webob.Request) -> webob.Response:
        # req.path is the full URL path, including the wsgi app mount point
        # (e.g. /data on forecast.ccsr), while req.path_info is the part of the
        # path after the mount point.
        url_path = request.path_info
        if url_path.endswith('/'):
            group = self.catalog.open_group(url_path)
            if group is None:
                return HTTPNotFound()
            return group_response(request, group)
        else:
            parent_path, varname = url_path.rsplit('/', maxsplit=1)
            catalog_path = f'{parent_path}/{varname}'
            ds = self.catalog.open_variable(catalog_path)
            if ds is None:
                # See if it's actually a dir and they just forgot the trailing slash
                if self.catalog.open_group(catalog_path + '/') is not None:
                    return HTTPFound(location=request.path + '/')
                return HTTPNotFound()
            return variable_response(ds)


def group_response(request: webob.Request, group: catalog.Group) -> webob.Response:
    """Generate a directory listing page."""

    components = request.path_info.rstrip('/').split('/')
    prefixes = [components[:i] for i in range(1, len(components) + 1)]
    paths = ['/'.join(prefix) for prefix in prefixes]
    # Add trailing slash, except to the first (empty) prefix because that would
    # make it an absolute path.
    paths = paths[:1] + [f'{path}/' for path in paths[1:]]
    breadcrumbs = (
        [
            f'<a href="{request.relative_url(path, to_application=True)}">{prefix[-1] or 'Top'}</a>'
            for path, prefix in zip(paths, prefixes)
        ]
    )
    trail = ' &rarr; '.join(breadcrumbs)
    context = {
        'breadcrumbs': trail,
        'group': group,
    }

    return webob.Response(
        body=dir_template.render(context),
    )

dir_template = jinja2.Template("""
    <!-- OPeNDAP -->
    {{ breadcrumbs }}
    <p>
    {{ group.summary.description }}
    <p>
    {{ group.documentation }}
    {% if group.groups %}
    <h1>Groups</h1>
    <table>
        <tbody>
            {% for x in group.groups %}
            <tr><td><a href="{{ x.name }}/">{{ x.name }}</a></td></tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
    {% if group.variables %}
    <h1>Variables</h1>
    <table>
        <tbody>
            {% for x in group.variables %}
            <tr>
                <td><a href="{{ x.name }}">{{ x.name }}</a></td>
                <td>{{ x.description }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% endif %}
""")


def variable_response(ds: xr.Dataset) -> webob.Response:
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

    # To help users understand what they will get via opendap, add the units
    # and calendar attributes that the response will have when datetimes
    # get cf-encoded.
    for coord in ds.coords.values():
        if np.issubdtype(coord.dtype, np.datetime64):
            coord.attrs['units'] = coord.encoding['units']
            coord.attrs['calendar'] = coord.encoding['calendar']

    context = {
        'ds': ds,
    }
    return webob.Response(body=var_template.render(context))


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
