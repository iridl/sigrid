import numpy as np
import xarray as xr

from sigrid.harmonize import DisplayDataset
from sigrid.serve import pages


def test_build_tree_opens_path_to_variable():
    listings = [
        DisplayDataset(['NMME', 'SubC'], []),
        DisplayDataset(['ECCC', 'EMC'], []),
        DisplayDataset([], ['ua', 'tas']),
    ]
    root = pages.build_tree('/data', ['SubC', 'ECCC'], listings, 'ua')
    assert root.href == '/data/'
    assert [i.name for i in root.children] == ['NMME/', 'SubC/']
    nmme, subc = root.children
    assert nmme.children == []
    eccc, emc = subc.children
    assert emc.children == []
    assert [(i.href, i.current) for i in eccc.children] == [
        ('/data/SubC/ECCC/ua', True),
        ('/data/SubC/ECCC/tas', False),
    ]
    assert not eccc.current


def test_build_tree_marks_current_dataset():
    listings = [DisplayDataset(['SubC'], []), DisplayDataset(['ECCC'], [])]
    root = pages.build_tree('', ['SubC'], listings)
    assert root.href == '/'
    assert root.children[0].current
    assert not root.current


def test_build_tree_shows_hidden_dataset_on_path():
    # The root listing leaves out the hidden dataset S2S.
    listings = [DisplayDataset(['SubC'], []), DisplayDataset([], [])]
    root = pages.build_tree('/data', ['S2S'], listings)
    assert [i.name for i in root.children] == ['SubC/', 'S2S/']


def test_describe_values():
    assert pages.describe_values(np.array([850, 200, 100])) == '850, 200, 100'
    assert pages.describe_values(np.arange(-90.0, 91.0)) == '-90 to 90, by 1'
    assert pages.describe_values(np.array([0, 1, 2, 3, 4, 5, 10])) == '0 to 10'


def test_describe_values_date_spacing():
    # Twice-weekly starts, as in SubC GEPS8.
    dates = np.array(['2024-06-13', '2024-06-17', '2024-06-20', '2024-06-24',
                      '2024-06-27', '2024-07-01', '2024-07-04'], dtype='datetime64[ns]')
    assert pages.describe_values(dates) == '2024-06-13 to 2024-07-04, every 3 days or 4 days'
    weekly = np.arange('2024-01-01', '2024-03-01', 7, dtype='datetime64[D]').astype('datetime64[ns]')
    assert pages.describe_values(weekly).endswith(', every 7 days')
    daily = np.arange('2024-01-01', '2024-03-01', dtype='datetime64[D]')
    gappy = np.delete(daily, [5, 10, 11, 20, 30, 31, 32]).astype('datetime64[ns]')
    assert pages.describe_values(gappy).endswith(', every 1 day, with 4 longer gaps')
    monthly = np.arange('1991-01', '1992-01', dtype='datetime64[M]').astype('datetime64[ns]')
    assert pages.describe_values(monthly) == '1991-01-01 to 1991-12-01, monthly'
    assert pages.describe_values(np.delete(monthly, [4])) == '1991-01-01 to 1991-12-01, monthly, with 1 gap'
    leads = np.arange(1, 40).astype('timedelta64[D]').astype('timedelta64[ns]')
    assert pages.describe_values(leads) == '1 day to 39 days, by 1 day'
    assert pages.format_value(np.timedelta64(6, 'h')) == '6 hours'


def daily_forecast(lead_coord=True):
    """A small dataset shaped like a SubC daily forecast."""
    S = np.array(['2024-06-13', '2024-06-17', '2024-06-20'], dtype='datetime64[ns]')
    L = np.arange(1, 8).astype('timedelta64[D]').astype('timedelta64[ns]')
    target = S[:, None] + L[None, :]
    bnds = np.stack([target, target + np.timedelta64(1, 'D')], axis=2)
    ds = xr.Dataset(
        {'ua': (('S', 'L', 'X'), np.zeros((3, 7, 4), 'float32'),
                {'long_name': 'Eastward wind', 'units': 'm s-1'})},
        coords={
            'S': ('S', S, {'long_name': 'Forecast start time'}),
            'X': ('X', [0.0, 90.0, 180.0, 270.0], {'long_name': 'Longitude', 'units': 'degree_east'}),
            'target': (('S', 'L'), target, {'long_name': 'Forecast target period'}),
            'target_bnds': (('S', 'L', 'nbound'), bnds),
        },
        attrs={'Conventions': 'CF-1.13'},
    )
    if lead_coord:
        ds = ds.assign_coords(L=('L', L, {'long_name': 'Lead'}))
    ds.S.encoding.update(dtype='int32')
    return ds


def test_coord_infos_show_valid_times():
    infos = {i.name: i for i in pages.coord_infos(daily_forecast(), 'ua')}
    assert list(infos) == ['S', 'L', 'X', 'target', 'target_bnds']
    assert infos['S'].dtype == 'int32'
    assert infos['L'].summary == '1 day to 7 days, by 1 day'
    assert infos['target'].summary == '2024-06-14 to 2024-06-27'
    assert infos['target'].values.splitlines()[0] == '2024-06-13: 2024-06-14 to 2024-06-20, every 1 day'
    assert infos['target_bnds'].summary == '[target, target + 1 day]'
    assert infos['X'].values == '0, 90, 180, 270'


def test_coord_infos_dimension_without_coordinate():
    infos = {i.name: i for i in pages.coord_infos(daily_forecast(lead_coord=False), 'ua')}
    assert infos['L'].summary == 'No coordinate variable'
    assert infos['L'].dims == '(L: 7)'


def test_render_var():
    listings = [DisplayDataset(['f'], []), DisplayDataset([], ['ua'])]
    html = pages.render_var('/data', ['f'], listings, 'ua', daily_forecast(), 'https://x/data/f/ua')
    assert 'Coordinates' in html and 'Other attributes' in html
    assert '.das' not in html and '.dds' not in html
    assert '2024-06-14 to 2024-06-27' in html


def test_render_dir_variable_table():
    listings = [
        DisplayDataset(['f'], []),
        DisplayDataset([], ['pr'], variable_attrs={'pr': {'long_name': 'Total precipitation', 'units': 'mm/day'}}),
    ]
    html = pages.render_dir('/data', ['f'], listings)
    assert '<td><a href="pr">pr</a></td><td>Total precipitation</td><td>mm/day</td>' in html


def test_render_dir_shows_links():
    listings = [
        DisplayDataset(['SPEAR'], []),
        DisplayDataset(['forecast'], [], None, [
            {'name': 'Journal Paper', 'description': 'SPEAR <paper>', 'href': 'https://doi.org/x'},
            {'name': 'site', 'href': 'https://example.org/'},
        ]),
    ]
    html = pages.render_dir('/data', ['SPEAR'], listings)
    assert '<a href="https://doi.org/x">SPEAR &lt;paper&gt;</a>' in html
    assert '<a href="https://example.org/">site</a>' in html


def test_coord_infos_monthly_bounds_in_months():
    # Monthly starts and leads, as in C3S and NMME: bounds in days vary
    # with month length, so they are shown in months.
    S = np.array(['2024-01', '2024-02', '2024-03'], dtype='datetime64[M]')
    target = (S[:, None] + np.arange(3)).astype('datetime64[ns]')
    bnds = np.stack([target, (S[:, None] + np.arange(1, 4)).astype('datetime64[ns]')], axis=2)
    ds = xr.Dataset(
        {'tas': (('S', 'L'), np.zeros((3, 3), 'float32'))},
        coords={
            'S': ('S', S.astype('datetime64[ns]')),
            'L': ('L', [0, 1, 2]),
            'target': (('S', 'L'), target),
            'target_bnds': (('S', 'L', 'nbound'), bnds),
        },
    )
    infos = {i.name: i for i in pages.coord_infos(ds, 'tas')}
    assert infos['S'].summary == '2024-01-01, 2024-02-01, 2024-03-01'
    assert infos['target'].summary == '2024-01-01 to 2024-05-01'
    assert infos['target'].values.splitlines()[0] == '2024-01-01: 2024-01-01, 2024-02-01, 2024-03-01'
    assert infos['target_bnds'].summary == '[target, target + 1 month]'


def test_handler_shows_served_units_for_times():
    # The page shows the units OPeNDAP sends for encoded times and leads.
    from sigrid.serve import CatalogFileHandler
    ds = daily_forecast()
    ds.S.encoding.update(units='hours since 1960-01-01', calendar='standard')
    ds.target.encoding.update(units='hours since 1960-01-01', calendar='standard')
    ds.target_bnds.encoding.update(units='hours since 1960-01-01', calendar='standard')
    ds.L.encoding.update(units='hours', dtype='int32')
    listings = [DisplayDataset(['f'], []), DisplayDataset([], ['ua'])]
    handler = CatalogFileHandler(ds, 'ua', None, ('/data', ['f'], listings))
    environ = {'REQUEST_METHOD': 'GET', 'PATH_INFO': '/f/ua', 'SCRIPT_NAME': '/data',
               'SERVER_NAME': 'x', 'SERVER_PORT': '80', 'wsgi.url_scheme': 'http'}
    body = b''.join(handler(environ, lambda status, headers: None)).decode()
    assert '<dt>units</dt><dd>hours</dd>' in body
    assert '<dt>units</dt><dd>hours since 1960-01-01</dd>' in body
