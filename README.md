# pydap-icechunk
An opendap server for icechunk stores, based on pydap and xarray. It supports transforming the
icechunk data with xarray before sending it to the user.

## Initial setup
```cp dot-env-example .env```
Change PYDAP_PORT so you don't collide with other developers.

## Inspecting an icechunk store
```
> pixi run python
>>> import pydap_icechunk
>>> ds = pydap_icechunk.open_icechunk('NMME/ECCC/CanSIPS-IC4/hindcast/prec')
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
>>>

## Testing
```
pixi run python test/server_for_test.py
```
then in another terminal, in python:
```
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
>>> ds['sst']
<xarray.DataArray 'sst' (S: 533, M: 10, L: 12, Y: 181, X: 360)> Size: 17GB
[4167633600 values with dtype=float32] 
Coordinates:                                                                       
  * S        (S) datetime64[ns] 4kB 1981-10-01 1982-01-01 ... 2026-04-01
  * M        (M) int32 40B 1 2 3 4 5 6 7 8 9 10                         
  * L        (L) int32 48B 0 1 2 3 4 5 6 7 8 9 10 11
  * Y        (Y) float32 724B -90.0 -89.0 -88.0 -87.0 ... 87.0 88.0 89.0 90.0
  * X        (X) float32 1kB 0.0 1.0 2.0 3.0 4.0 ... 356.0 357.0 358.0 359.0 
Attributes:                                                                        
    history:        From ccsm4_0_cfsrr_Fcst.E1.pop.h.nday1.1982-09-01
    long_name:      sea surface temperature (SST)                    
    long_name_mod:  T=01-SEP-1982 00:00:01-OCT-1982 00:00@AVE
    spatial_op:     Conservative remapping: 1st order: destarea: NCL: ./map_g...
    standard_name:  sea surface temperature
    units:          degreeC
```
