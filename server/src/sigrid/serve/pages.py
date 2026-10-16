"""HTML pages for browsing the catalog.

Every page has the same frame: a site header, the catalog tree on the left
and the page content on the right. The tree is opened along the path to the
current page, so it shows where the page sits and its neighbours.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import jinja2
import numpy as np
import xarray as xr

from sigrid.harmonize import DisplayDataset

SITE_NAME = 'CCSR Data Library'


@dataclass
class TreeItem:
    name: str
    href: str
    is_dir: bool = True
    current: bool = False
    # Only filled for datasets on the path to the current page.
    children: list['TreeItem'] = field(default_factory=list)


@dataclass
class GridRow:
    name: str
    long_name: str
    size: int
    values: str
    units: str


def render_dir(mount: str, components: Sequence[str], listings: Sequence[DisplayDataset]) -> str:
    """Page for a dataset (a folder in the catalog).

    mount is the URL path the app is served under, e.g. /data.
    components are the path components below the mount.
    listings[i] is the listing of the dataset at components[:i].
    """
    listing = listings[-1]
    return env.get_template('dir').render(
        title=components[-1] if components else root_name(mount),
        description=listing.description,
        links=listing.links,
        dirs=listing.subdatasets,
        vars=listing.variables,
        **frame(mount, components, listings),
    )


def render_var(
    mount: str,
    components: Sequence[str],
    listings: Sequence[DisplayDataset],
    varname: str,
    ds: xr.Dataset,
    url: str,
) -> str:
    """Page for a variable. components and listings are those of its parent dataset."""
    da = ds[varname]
    return env.get_template('var').render(
        title=varname,
        varname=varname,
        long_name=da.attrs.get('long_name', ''),
        units=da.attrs.get('units', ''),
        url=url,
        grid_rows=[grid_row(ds, str(dim)) for dim in da.dims],
        attr_groups=attr_groups(ds, varname),
        **frame(mount, components, listings, varname),
    )


def root_name(mount: str) -> str:
    return mount.strip('/').split('/')[-1] or 'data'


def frame(
    mount: str,
    components: Sequence[str],
    listings: Sequence[DisplayDataset],
    varname: str | None = None,
) -> dict:
    """Template values shared by all pages: site name, tree and breadcrumb."""
    root_href = mount.rstrip('/') + '/'
    crumbs = [(root_name(mount), root_href)]
    href = root_href
    for component in components:
        href += component + '/'
        crumbs.append((component, href))
    if varname is not None:
        crumbs.append((varname, href + varname))
    return {
        'site_name': SITE_NAME,
        'root_href': root_href,
        'tree': build_tree(mount, components, listings, varname),
        'crumbs': crumbs,
    }


def build_tree(
    mount: str,
    components: Sequence[str],
    listings: Sequence[DisplayDataset],
    varname: str | None = None,
) -> TreeItem:
    def build(i: int, href: str) -> list[TreeItem]:
        listing = listings[i]
        on_path = components[i] if i < len(components) else None
        names = list(listing.subdatasets)
        # Hidden datasets are left out of listings, but a page inside one
        # still shows its own path.
        if on_path is not None and on_path not in names:
            names.append(on_path)
        items = []
        for name in names:
            item = TreeItem(name + '/', f'{href}{name}/')
            if name == on_path:
                item.children = build(i + 1, item.href)
                item.current = i == len(components) - 1 and varname is None
            items.append(item)
        for var in listing.variables:
            items.append(TreeItem(
                var, href + var, is_dir=False,
                current=i == len(components) and var == varname,
            ))
        return items

    root_href = mount.rstrip('/') + '/'
    root = TreeItem(
        root_name(mount) + '/', root_href,
        current=not components and varname is None,
    )
    root.children = build(0, root_href)
    return root


def grid_row(ds: xr.Dataset, dim: str) -> GridRow:
    size = ds.sizes[dim]
    if dim not in ds.coords:
        # A dimension without a coordinate variable, e.g. lead L.
        return GridRow(dim, '', size, '', '')
    coord = ds[dim]
    return GridRow(
        dim,
        coord.attrs.get('long_name', ''),
        size,
        describe_values(coord.values),
        coord.attrs.get('units', ''),
    )


def describe_values(values: np.ndarray) -> str:
    """Short text for a coordinate's values: all of them if there are few,
    otherwise first to last, plus the step if it is constant."""
    if values.size == 0:
        return ''
    if values.size <= 6:
        return ', '.join(format_value(v) for v in values)
    text = f'{format_value(values[0])} to {format_value(values[-1])}'
    if np.issubdtype(values.dtype, np.number):
        steps = np.diff(values)
        if np.allclose(steps, steps[0]):
            text += f' by {format_value(steps[0])}'
    return text


def format_value(v) -> str:
    if isinstance(v, np.datetime64):
        # Dates at midnight are shown without the time.
        return np.datetime_as_string(v, unit='s').removesuffix('T00:00:00')
    if isinstance(v, (float, np.floating)):
        return f'{v:g}'
    return str(v)


def attr_groups(ds: xr.Dataset, varname: str) -> list[tuple[str, dict]]:
    """Attributes of the variable, then its coordinates, then the dataset."""
    groups = [(varname, ds[varname].attrs)]
    groups += [(str(name), coord.attrs) for name, coord in ds.coords.items()]
    groups.append(('Global', ds.attrs))
    return [(name, attrs) for name, attrs in groups if attrs]


env = jinja2.Environment(
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
    loader=jinja2.DictLoader({
'base': """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} · {{ site_name }}</title>
<style>
:root {
  --bg: #ffffff; --fg: #18212b; --muted: #5a6673; --rule: #d9dee3;
  --accent: #1f5f8b; --accent-soft: #e4eef5; --code-bg: #eef1f4;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14191e; --fg: #e3e8ed; --muted: #98a4b0; --rule: #2e3943;
    --accent: #7fb6dc; --accent-soft: #1e3242; --code-bg: #222b33;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--accent); }
code, pre, .tree { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.88em; }
.site { padding: 10px 20px; border-bottom: 2px solid var(--accent); }
.site a { color: var(--fg); font-weight: 600; text-decoration: none; }
.panes { display: grid; grid-template-columns: minmax(200px, 280px) minmax(0, 1fr); }
.tree { padding: 14px 16px; border-right: 1px solid var(--rule); overflow-x: auto; }
.tree summary { display: none; }
.tree ul { list-style: none; margin: 0; padding-left: 14px; }
.tree > details > ul { padding-left: 0; }
.tree li { padding: 1px 0; white-space: nowrap; }
.tree a { text-decoration: none; }
.tree a.var { color: var(--muted); }
.tree a[aria-current] { background: var(--accent-soft); color: var(--fg); border-radius: 3px; padding: 0 4px; }
main { padding: 14px 24px 40px; min-width: 0; max-width: 980px; }
.crumbs { margin: 0 0 6px; color: var(--muted); overflow-wrap: anywhere; }
h1 { margin: 0 0 4px; font-size: 1.6rem; }
h2 { margin: 24px 0 8px; font-size: 1.1rem; }
.sub { margin: 0; color: var(--muted); }
.desc { max-width: 70ch; }
.links { margin: 8px 0 0; padding-left: 20px; max-width: 70ch; }
.links li { margin-bottom: 2px; overflow-wrap: anywhere; }
table { border-collapse: collapse; }
td, th { text-align: left; padding: 5px 14px 5px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 500; }
.num { font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.listhead { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; justify-content: space-between; max-width: 600px; }
#filter { font: inherit; padding: 3px 8px; border: 1px solid var(--rule); border-radius: 4px; background: var(--bg); color: var(--fg); }
.access { padding: 12px 14px; border: 1px solid var(--rule); border-radius: 6px; margin-top: 16px; }
.access h2 { margin-top: 0; }
.urlrow { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.urlrow code, pre { background: var(--code-bg); padding: 4px 8px; border-radius: 4px; overflow-x: auto; }
.urlrow code { flex: 1; min-width: 0; white-space: nowrap; }
pre { padding: 8px 10px; margin: 10px 0 0; }
button { font: inherit; font-size: 0.85rem; padding: 4px 10px; border: 1px solid var(--accent); color: var(--accent); background: var(--bg); border-radius: 4px; cursor: pointer; }
details.attrs { margin-top: 20px; }
details.attrs summary { cursor: pointer; color: var(--accent); }
@media (max-width: 700px) {
  .panes { grid-template-columns: minmax(0, 1fr); }
  .tree { border-right: 0; border-bottom: 1px solid var(--rule); }
  .tree summary { display: list-item; cursor: pointer; color: var(--muted); font-family: system-ui, sans-serif; }
  main { padding: 14px 16px 40px; }
}
</style>
</head>
<body>
{% block marker %}{% endblock %}
<header class="site"><a href="{{ root_href }}">{{ site_name }}</a></header>
<div class="panes">
<nav class="tree" aria-label="Catalog">
<details open>
<summary>Catalog</summary>
{% macro tree_list(items) %}
<ul>
{% for item in items %}
<li><a href="{{ item.href }}" class="{{ 'dir' if item.is_dir else 'var' }}"{% if item.current %} aria-current="page"{% endif %}>{{ item.name }}</a>
{% if item.children %}{{ tree_list(item.children) }}{% endif %}
</li>
{% endfor %}
</ul>
{% endmacro %}
{{ tree_list([tree]) }}
</details>
</nav>
<main>
<p class="crumbs">
{% for name, href in crumbs %}{% if not loop.first %} / {% endif %}{% if loop.last %}{{ name }}{% else %}<a href="{{ href }}">{{ name }}</a>{% endif %}{% endfor %}
</p>
{% block main %}{% endblock %}
</main>
</div>
{% block script %}{% endblock %}
</body>
</html>
""",

'dir': """{% extends 'base' %}
{% block marker %}<!-- OPeNDAP -->{% endblock %}
{% block main %}
<h1>{{ title }}</h1>
{% if description %}
<p class="desc">{{ description }}</p>
{% endif %}
{% if links %}
<ul class="links">
{% for link in links %}
<li><a href="{{ link.href }}">{{ link.description or link.name or link.href }}</a></li>
{% endfor %}
</ul>
{% endif %}
{% if dirs %}
<div class="listhead">
<h2>Datasets</h2>
<input type="search" id="filter" placeholder="Filter" aria-label="Filter datasets">
</div>
<table id="datasets">
<tbody>
{% for x in dirs %}
<tr><td><a href="{{ x }}/">{{ x }}</a></td></tr>
{% endfor %}
</tbody>
</table>
{% endif %}
{% if vars %}
<h2>Variables</h2>
<table>
<tbody>
{% for x in vars %}
<tr><td><a href="{{ x }}">{{ x }}</a></td></tr>
{% endfor %}
</tbody>
</table>
{% endif %}
{% endblock %}
{% block script %}
{% if dirs %}
<script>
const filter = document.getElementById('filter');
filter.addEventListener('input', () => {
  const q = filter.value.toLowerCase();
  for (const row of document.querySelectorAll('#datasets tr')) {
    row.hidden = !row.textContent.toLowerCase().includes(q);
  }
});
</script>
{% endif %}
{% endblock %}
""",

'var': """{% extends 'base' %}
{% block main %}
<h1>{{ long_name or varname }}</h1>
<p class="sub"><code>{{ varname }}</code>{% if units %} · {{ units }}{% endif %}</p>
<section class="access">
<h2>Access</h2>
<div class="urlrow"><code id="url">{{ url }}</code><button type="button" id="copy">Copy OPeNDAP URL</button></div>
<pre id="snippet">import xarray as xr
ds = xr.open_dataset("<span class="url">{{ url }}</span>")</pre>
<p>OPeNDAP metadata: <a href="{{ varname }}.das">{{ varname }}.das</a> · <a href="{{ varname }}.dds">{{ varname }}.dds</a></p>
</section>
<h2>Grids</h2>
<div class="scroll">
<table>
<thead><tr><th>Name</th><th>Long name</th><th>Size</th><th>Values</th><th>Units</th></tr></thead>
<tbody>
{% for row in grid_rows %}
<tr><td><code>{{ row.name }}</code></td><td>{{ row.long_name }}</td><td class="num">{{ row.size }}</td><td class="num">{{ row.values }}</td><td>{{ row.units }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
<details class="attrs">
<summary>All attributes</summary>
<pre>
{% for name, attrs in attr_groups %}
{{ name }}
{% for k, v in attrs.items() %}
    {{ k }}: {{ v }}
{% endfor %}
{% endfor %}
</pre>
</details>
{% endblock %}
{% block script %}
<script>
// Show the URL as the browser sees it, which can differ from what the
// server sees behind a proxy (e.g. https vs http).
const url = window.location.origin + window.location.pathname;
document.getElementById('url').textContent = url;
document.querySelector('#snippet .url').textContent = url;
const copy = document.getElementById('copy');
copy.addEventListener('click', () => {
  navigator.clipboard.writeText(url).then(() => {
    copy.textContent = 'URL copied';
    setTimeout(() => copy.textContent = 'Copy OPeNDAP URL', 2000);
  }).catch(err => {
    console.error('Failed to copy OPeNDAP URL: ', err);
  });
});
</script>
{% endblock %}
""",
    }),
)
