"""HTML pages for browsing the catalog.

Every page has the same frame: a site header, the catalog tree on the left
and the page content on the right. The tree is opened along the path to the
current page, so it shows where the page sits and its neighbours. In narrow
windows the tree is hidden and the breadcrumb is the only navigation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jinja2
import numpy as np
import xarray as xr

from sigrid.harmonize import Coords, DisplayDataset

SITE_NAME = 'CCSR Data Library'

# Coordinates with more elements than this get a summary only, so a page
# never loads a large array.
MAX_SUMMARY_SIZE = 1_000_000

ONE_HOUR = np.timedelta64(1, 'h')


@dataclass
class TreeItem:
    name: str
    href: str
    is_dir: bool = True
    current: bool = False
    # Only filled for datasets on the path to the current page.
    children: list['TreeItem'] = field(default_factory=list)


@dataclass
class VarInfo:
    """One row of a variable page: the data variable or a coordinate variable."""
    name: str
    dims: str
    dtype: str = ''
    long_name: str = ''
    summary: str = ''
    units: str = ''
    attrs: Mapping = field(default_factory=dict)
    # Shown in the row's drawer, under the attributes.
    values_label: str = ''
    values: str = ''


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
        vars=[
            (var, listing.variable_attrs.get(var, {}))
            for var in listing.variables
        ],
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
        data_var=data_var_info(ds, varname),
        coords=coord_infos(ds, varname),
        global_attrs=ds.attrs,
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


def data_var_info(ds: xr.Dataset, varname: str) -> VarInfo:
    da = ds[varname]
    return VarInfo(
        name=varname,
        dims=dims_text(da),
        dtype=served_dtype(da),
        long_name=da.attrs.get('long_name', ''),
        units=da.attrs.get('units', ''),
        attrs=da.attrs,
    )


def coord_infos(ds: xr.Dataset, varname: str) -> list[VarInfo]:
    """The variable's dimension coordinates in its dimension order, then the
    other coordinates (e.g. target, target_bnds)."""
    da = ds[varname]
    infos = []
    for dim in map(str, da.dims):
        if dim in ds.coords:
            infos.append(coord_info(ds, dim))
        else:
            infos.append(VarInfo(
                name=dim, dims=f'({dim}: {ds.sizes[dim]})',
                summary='No coordinate variable',
            ))
    infos += [
        coord_info(ds, str(name)) for name in ds.coords
        if name not in da.dims
    ]
    return infos


def coord_info(ds: xr.Dataset, name: str) -> VarInfo:
    coord = ds[name]
    info = VarInfo(
        name=name,
        dims=dims_text(coord),
        dtype=served_dtype(coord),
        long_name=coord.attrs.get('long_name', ''),
        units=coord.attrs.get('units', ''),
        attrs=coord.attrs,
    )
    if coord.size == 0 or coord.size > MAX_SUMMARY_SIZE:
        return info
    values = coord.values
    if coord.ndim == 1:
        info.summary = describe_values(values)
        info.values_label = 'Values'
        info.values = ', '.join(format_value(v) for v in values)
        return info
    # Multi-dimensional times, e.g. target (S, L) and target_bnds (S, L, nbound),
    # are shown as offsets from S or target when the offsets don't depend on S.
    offsets = relative_offsets(ds, coord)
    if offsets is not None:
        base, offset_texts, summary = offsets
        info.summary = summary
        info.values_label = f'{name} − {base}'
        info.values = ', '.join(offset_texts)
    else:
        info.summary = f'{format_value(values.min())} to {format_value(values.max())}'
    return info


def relative_offsets(ds: xr.Dataset, coord: xr.DataArray):
    """For a datetime coordinate that is S or target plus offsets that are the
    same for every S, return (base name, offsets along the other dims, summary).
    Otherwise None."""
    if not np.issubdtype(coord.dtype, np.datetime64) or Coords.S not in coord.dims:
        return None
    if Coords.target in ds.coords and ds[Coords.target].dims == coord.dims[:-1]:
        # Bounds: one offset from target per bound.
        base = Coords.target
    elif Coords.S in ds.coords:
        base = Coords.S
    else:
        return None
    # Exact offsets (e.g. days) first, then whole months for monthly data,
    # whose offsets in days vary with the length of the month.
    for offset, fmt in (
        (coord - ds[base], format_value),
        (month_offset(coord, ds[base]), format_months),
    ):
        if offset is None:
            continue
        first = offset.isel({Coords.S: 0})
        if not bool((offset == first).all()):
            continue
        first = first.values
        if base == Coords.target:
            # Offsets per bound must also be the same for every other element.
            per_bound = first.reshape(-1, first.shape[-1])
            if not (per_bound == per_bound[0]).all():
                continue
            parts = [
                base if o == np.zeros((), o.dtype) else f'{base} + {fmt(o)}'
                for o in per_bound[0]
            ]
            return base, [fmt(o) for o in per_bound[0]], '[' + ', '.join(parts) + ']'
        offsets = first.ravel()
        return base, [fmt(o) for o in offsets], f'{base} + ({describe_offsets(offsets, fmt)})'
    return None


def month_offset(coord: xr.DataArray, base: xr.DataArray) -> xr.DataArray | None:
    """coord - base in whole months, if both fall on the first of a month."""
    def months(da):
        values = da.values
        month = values.astype('datetime64[M]')
        if not (values.astype('datetime64[D]') == month.astype('datetime64[D]')).all():
            return None
        return xr.DataArray(month.astype(np.int64), dims=da.dims)
    coord_months, base_months = months(coord), months(base)
    if coord_months is None or base_months is None:
        return None
    return coord_months - base_months


def describe_offsets(offsets: np.ndarray, fmt) -> str:
    if len(offsets) == 1:
        return fmt(offsets[0])
    text = f'{fmt(offsets[0])} to {fmt(offsets[-1])}'
    steps = np.unique(np.diff(offsets))
    if len(steps) == 1:
        text += f', by {fmt(steps[0])}'
    return text


def format_months(n) -> str:
    return f'{n} month' if abs(n) == 1 else f'{n} months'


def dims_text(da: xr.DataArray) -> str:
    return '(' + ', '.join(f'{dim}: {size}' for dim, size in da.sizes.items()) + ')'


def served_dtype(da: xr.DataArray) -> str:
    """The type OPeNDAP sends, which differs from xarray's for encoded times."""
    return str(np.dtype(da.encoding.get('dtype', da.dtype)))


def describe_values(values: np.ndarray) -> str:
    """Short text for a coordinate's values: all of them if there are few,
    otherwise first to last and the spacing."""
    if values.size == 0:
        return ''
    if values.size <= 6:
        return ', '.join(format_value(v) for v in values)
    text = f'{format_value(values[0])} to {format_value(values[-1])}'
    spacing = describe_spacing(values)
    if spacing:
        text += ', ' + spacing
    return text


def describe_spacing(values: np.ndarray) -> str:
    if np.issubdtype(values.dtype, np.datetime64):
        months = values.astype('datetime64[M]')
        first_of_month = (values.astype('datetime64[D]') == months.astype('datetime64[D]')).all()
        if first_of_month and (np.diff(months) == np.timedelta64(1, 'M')).all():
            return 'every month'
        steps, counts = np.unique(np.diff(values), return_counts=True)
        if len(steps) <= 3:
            return 'every ' + ' or '.join(format_value(s) for s in steps)
        # Regular starts with some missing, e.g. daily with a few gaps.
        if counts[0] >= 0.8 * counts.sum():
            gaps = counts.sum() - counts[0]
            return f'every {format_value(steps[0])}, with {gaps} longer gaps'
        return 'irregular spacing'
    if np.issubdtype(values.dtype, np.timedelta64):
        steps = np.unique(np.diff(values))
        return f'by {format_value(steps[0])}' if len(steps) == 1 else ''
    if np.issubdtype(values.dtype, np.number):
        steps = np.diff(values)
        if np.allclose(steps, steps[0]):
            return f'by {format_value(steps[0])}'
    return ''


def format_value(v) -> str:
    if isinstance(v, np.datetime64):
        # Dates at midnight are shown without the time.
        return np.datetime_as_string(v, unit='s').removesuffix('T00:00:00')
    if isinstance(v, np.timedelta64):
        return format_timedelta(v)
    if isinstance(v, (float, np.floating)):
        return f'{v:g}'
    return str(v)


def format_timedelta(v: np.timedelta64) -> str:
    hours = v / ONE_HOUR
    if hours % 24 == 0:
        days = int(hours // 24)
        return f'{days} day' if abs(days) == 1 else f'{days} days'
    if hours == int(hours):
        return f'{int(hours)} hour' if abs(hours) == 1 else f'{int(hours)} hours'
    return str(v)


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
code, pre, .tree, .mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.88em; }
.site { padding: 10px 20px; border-bottom: 2px solid var(--accent); }
.site a { color: var(--fg); font-weight: 600; text-decoration: none; }
.panes { display: grid; grid-template-columns: minmax(200px, 280px) minmax(0, 1fr); }
.tree { padding: 14px 16px; border-right: 1px solid var(--rule); overflow-x: auto; }
.tree ul { list-style: none; margin: 0; padding-left: 14px; }
.tree > ul { padding-left: 0; }
.tree li { padding: 1px 0; white-space: nowrap; }
.tree a { text-decoration: none; }
.tree a.var { color: var(--muted); }
.tree a[aria-current] { background: var(--accent-soft); color: var(--fg); border-radius: 3px; padding: 0 4px; }
main { padding: 14px 24px 40px; min-width: 0; max-width: 1100px; }
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
.vars { border-top: 1px solid var(--rule); }
.var { border-bottom: 1px solid var(--rule); }
.var > summary, .var > .row { display: flex; flex-wrap: wrap; gap: 2px 14px; padding: 6px 0 6px 18px; position: relative; }
.var > summary { cursor: pointer; list-style: none; }
.var > summary::-webkit-details-marker { display: none; }
.var > summary::before { content: "▸"; position: absolute; left: 2px; color: var(--muted); }
.var[open] > summary::before { content: "▾"; }
.var .n { flex: 0 0 7em; font-weight: 600; }
.var .d { flex: 0 0 13em; color: var(--muted); }
.var .t { flex: 0 0 5em; color: var(--muted); }
.var .ln { flex: 1 1 11em; }
.var .s { flex: 2 1 16em; font-variant-numeric: tabular-nums; }
.var .u { flex: 0 1 11em; color: var(--muted); }
.var .body { padding: 0 0 10px 18px; }
dl.attrs { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 2px 14px; margin: 4px 0 0; }
dl.attrs dt { color: var(--muted); }
dl.attrs dd { margin: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.vlabel { margin: 10px 0 0; color: var(--muted); }
pre.values { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 16em; overflow-y: auto; margin-top: 4px; }
.none { margin: 4px 0 0; color: var(--muted); }
/* Below this width the tree takes too much room; the breadcrumb remains. */
@media (max-width: 900px) {
  .panes { grid-template-columns: minmax(0, 1fr); }
  .tree { display: none; }
  main { padding: 14px 16px 40px; }
}
</style>
</head>
<body>
{% block marker %}{% endblock %}
<header class="site"><a href="{{ root_href }}">{{ site_name }}</a></header>
<div class="panes">
<nav class="tree" aria-label="Catalog">
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
<div class="scroll">
<table>
<thead><tr><th>Name</th><th>Long name</th><th>Units</th></tr></thead>
<tbody>
{% for x, attrs in vars %}
<tr><td><a href="{{ x }}">{{ x }}</a></td><td>{{ attrs.long_name }}</td><td>{{ attrs.units }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
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
{% macro var_row(v) %}
{% set cells %}<span class="n mono">{{ v.name }}</span><span class="d mono">{{ v.dims }}</span><span class="t mono">{{ v.dtype }}</span><span class="ln">{{ v.long_name }}</span><span class="s">{{ v.summary }}</span><span class="u">{{ v.units }}</span>{% endset %}
{% if v.attrs or v.values %}
<details class="var">
<summary>{{ cells }}</summary>
<div class="body">
{% if v.attrs %}
<dl class="attrs">
{% for k, val in v.attrs.items() %}
<dt>{{ k }}</dt><dd>{{ val }}</dd>
{% endfor %}
</dl>
{% else %}
<p class="none">No attributes</p>
{% endif %}
{% if v.values %}
<p class="vlabel">{{ v.values_label }}</p>
<pre class="values">{{ v.values }}</pre>
{% endif %}
</div>
</details>
{% else %}
<div class="var"><div class="row">{{ cells }}</div></div>
{% endif %}
{% endmacro %}
{% block main %}
<h1>{{ long_name or varname }}</h1>
<p class="sub"><code>{{ varname }}</code>{% if units %} · {{ units }}{% endif %}</p>
<section class="access">
<h2>Access</h2>
<div class="urlrow"><code id="url">{{ url }}</code><button type="button" id="copy">Copy OPeNDAP URL</button></div>
<pre id="snippet">import xarray as xr
ds = xr.open_dataset("<span class="url">{{ url }}</span>")</pre>
</section>
<h2>Data variable</h2>
<div class="vars">
{{ var_row(data_var) }}
</div>
<h2>Coordinates</h2>
<div class="vars">
{% for c in coords %}
{{ var_row(c) }}
{% endfor %}
</div>
{% if global_attrs %}
<div class="vars" style="margin-top: 20px">
<details class="var">
<summary><span class="n">Other attributes</span></summary>
<div class="body">
<dl class="attrs">
{% for k, val in global_attrs.items() %}
<dt>{{ k }}</dt><dd>{{ val }}</dd>
{% endfor %}
</dl>
</div>
</details>
</div>
{% endif %}
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
