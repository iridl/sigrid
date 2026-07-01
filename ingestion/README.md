# Tools for converting climate forecasts to Icechunk

## One-time setup
- Make sure you have a `.env` file in the parent directory. See ../README.md.
  For running on forecast.ccsr and writing production datasets, no changes are
  necessary. For test runs you can change `ICECHUNK_ROOT` to point to a
  directory of your own.

## Create or update an icechunk store
```
pixi run preprocess NMME/COLA-RSMAS/CCSM4/PRATE_SFC
```
where `NMME/COLA-RSMAS/CCSM4/` is the relative path, below
`$COOKED_CATALOG_ROOT` (defined in ../.env), of a directory containing an
`index.py` file, and `PRATE_SFC` is the name of one of the entries in
`dataset['vars']` within that file.

This command creates/updates the icechunk store at
`$ICECHUNK_ROOT/NMME/COLA-RSMAS/CCSM4/PRATE_SFC`, where `$ICECHUNK_ROOT` is
defined in `../.env`.

Use option --parallel n where n is the maximum number of processes to execute
tasks asynchronously. This will speedup the icechunking, particularly useful if
creating a large new set.

To inspect an existing store, in an interactive python session,
```
>>> from sigrid import preprocess
>>> preprocess.open_icechunk('NMME/COLA-RSMAS/CCSM4/PRATE_SFC')
```

## Catalog format (index.py)
Example
```
dataset = {
    'original_time_dim': 'time',
    'expand_coords': 'valid_time',
    'dir': 'NMME/ECCC/CanSIPS-IC4/forecast',
    'vars': {
        'PRATE_SFC': {
            # 2026022800_cansips_forecast_raw_nmme_latlon-1x1_PRATE_SFC_0_2026-03_allmembers.grib2
            'pattern': r'\d\d\d\d\d\d\d\d\d\d_cansips_forecast_raw_nmme_latlon-1x1_PRATE_SFC_0_(?P<year>\d\d\d\d)-(?P<month>\d\d)_allmembers.grib2$',
        },
        'WTMP_SFC': {
            'pattern': r'\d\d\d\d\d\d\d\d\d\d_cansips_forecast_raw_nmme_latlon-1x1_WTMP_SFC_0_(?P<year>\d\d\d\d)-(?P<month>\d\d)_allmembers.grib2$',
        },
    }
}
```
`original_time_dim` (optional): the file-internal name of the primary time
dimension if it's represented inside the file. (In many cases it's only
represented in the filename, not as a dimension inside the file.)

`expand_coords` (optional): list of the names of file-internal coordinate
variables that vary with the primary time dimension. This is common for target
period variables: if target is represented inside each file as a 1-dimensional
coordinate depending only on lead time, listing it in `expand_coords` will cause
it to become a 2-dimensional coordinate depending on S and L in the icechunk
store.

`aux_coords` (optional): list of variables in (nc) files other than the desired
`vars` (e.g. TIME_bnds) that will be reset as coordinates.

`dir`: path to the root directory of the file set, relative to `$ORIG_DATA_ROOT`
(defined in ../.env).

`parse_match` (optional): if the standard mechanism for mapping filenames to
coordinates can't handle the patterns, you can override it here. Value is a
function that takes a dictionary of named captures from the regex, and returns a
`FileCoords` object (see definition in `src/sigrid/preprocess.py`). Not usually needed; see
NMME/NASA-GMAO/GEOSS2S for an example.

`vars`: a dictionary where keys (left of the colon) are icechunk internal variable names,
values (right of the colon) are dictionaries structured as follows:

- `pattern`: a regex pattern containing named capture groups for the
  parts of the filename that should be used to determine coordinate values.
  The supported names are `year`, `month`, `day` (optional, defaults to 1),
  `member`, and `pressure`.

- Other configuration items that are permitted at the `dataset` level are also
  permitted here. They will override the defaults set under `dataset`.
