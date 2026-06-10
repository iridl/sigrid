# pydap-icechunk
An opendap server for icechunk stores, based on pydap and xarray. It supports
transforming the icechunk data with xarray before sending it to the user.

## Initial setup

```cp dot-env-example .env```

Change `PYDAP_PORT` so you don't collide with other developers.

## Inspecting an icechunk store

```
> pixi run python
>>> import cataloging as cat
>>> ds = cat.open_icechunk('NMME/ECCC/CanSIPS-IC4/hindcast/prcp')
>>> ds
<xarray.Dataset> Size: 52GB
Dimensions:        (IRIDL_time: 414, number: 40, step: 12, latitude: 181,
                    longitude: 360)
Coordinates:
  * IRIDL_time     (IRIDL_time) datetime64[ns] 3kB 1990-01-01 ... 2024-06-01
    time           (IRIDL_time) datetime64[ns] 3kB ...
  * number         (number) int64 320B 1 2 3 4 5 6 7 8 ... 34 35 36 37 38 39 40
    step_expanded  (IRIDL_time, step) timedelta64[ns] 40kB ...
    valid_time     (step) datetime64[ns] 96B ...
  * latitude       (latitude) float64 1kB -90.0 -89.0 -88.0 ... 88.0 89.0 90.0
  * longitude      (longitude) float64 3kB 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0
    surface        float64 8B ...
Dimensions without coordinates: step
Data variables:
    prate          (IRIDL_time, number, step, latitude, longitude) float32 52GB ...
Attributes:
    GRIB_edition:            2
    GRIB_centre:             cwao
    GRIB_centreDescription:  Canadian Meteorological Service - Montreal
    GRIB_subCentre:          0
    Conventions:             CF-1.7
    institution:             Canadian Meteorological Service - Montreal
    history:                 2026-04-28T18:33 GRIB to CDM+CF via cfgrib-0.9.1...
```

## Writing a catalog entry

A catalog consists of a set of files named `index.py` arranged in a directory
hierarchy, the root of which is indicated by the environment variable
`CATALOG_ROOT`. For an example, see `catalog-example` in this repository.

An `index.py` file at the lowest level of the hierarchy should define two
functions with the signature `open(varname)` and `list_vars()`. `open` should
take a string (the name of a variable) and return a dataset. `open` typically
calls `cataloging.open_icechunk` with a path to an icechunk store, which is
interpreted relative to the value of the environment variable `ICECHUNK_ROOT`.
`list_vars` should return a list of the variable names that are valid to be
requested from `open`.

`index.py` files at higher levels of the hierarchy should define a function with
the signature `transform(ds)`, which takes an `xarray.Dataset` as its argument,
and returns another `xarray.Dataset`. The first `transform` function that is
found along the path from the leaf node up to the root is called on the return
value of `open`, and then each successive `transform` function is called on the
return value of the previous one, in bottom-up order.

The purpose of the `transform` functions is to support the joint use of datasets
from different providers by mapping disparate provider-specific file formats,
array structures, and metadata conventions into a common structure.
`cataloging.py` defines a number of utility functions that may be useful for
such transformations.

## Manual testing
```
pixi run python test/server.py
```
then visit http://localhost:8081 in a browser; or to test OPeNDAP functionality,
in another terminal (note that this uses a separate pixi environment called
`client`):
```
> pixi run -e client python
>>> import xarray as xr
>>> ds = xr.open_dataset('http://localhost:8001/NMME/COLA-RSMAS/CCSM4/sst', engine='netcdf4')
>>> ds
<xarray.Dataset> Size: 17GB                                                        
Dimensions:  (S: 533, M: 10, L: 12, Y: 181, X: 360)
Coordinates:                                                                       
  * S        (S) datetime64[ns] 4kB 1981-10-01 1982-01-01 ... 2026-04-01
  * M        (M) int32 40B 1 2 3 4 5 6 7 8 9 10
  * L        (L) int32 48B 0 1 2 3 4 5 6 7 8 9 10 11           
  * Y        (Y) float32 724B -90.0 -89.0 -88.0 -87.0 ... 87.0 88.0 89.0 90.0
  * X        (X) float32 1kB 0.0 1.0 2.0 3.0 4.0 ... 356.0 357.0 358.0 359.0
Data variables:                          
    sst      (S, M, L, Y, X) float32 17GB ...                  
    target   (S, L) object 51kB ...
Attributes:
    Comment:      Dughong Min (dmin@rsmas.miami.edu) and Ben Kirtman (bkirtma...
    Conventions:  CF-1.0                                                           
    Created:      Thu Sep 18 13:30:01 EDT 2014
    Generator:    NCL v.5.0                                                        
    References:   Ben P. Kirtman, Dughong Min. (2009) Multimodel Ensemble ENS...
    Title:        CCSM4.0 National Multi-Model Ensembles(NMME) project
```
