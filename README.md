# sigrid-cdl: the Sigrid Climate Data Library suite

Sigrid is a collection of software components for building and operating a
library of analysis-ready climate data. The current focus is on seasonal and
seasonal-to-subseasonal climate forecasts, and on observational data that are
used to validate and calibrate such forecasts.

There has not yet been a stable release of Sigrid. It is changing rapidly, and
there should be no expectation of backwards compatibility until we announce
otherwise.

This project is named Sigrid because it is (in some ways) a **S**uccessor to
**I**n**grid**. [Ingrid](https://bitbucket.org/iridl/ingrid/) is the software on
which the IRI Data Library (199x-2026) was built. Sigrid follows some of the
architectural principles developed in Ingrid, but shares no code with it.

## Components
- `ingestion`: a library of utilities for writing scripts that periodically
download data files from elsewhere.
- `icechunker`: a tool for combining a large set of NetCDF, GRIB, or TIFF files
  into an [icechunk](https://github.com/earth-mover/icechunk) store.
- `cooked-catalog`: a library of utilities for harmonizing the metadata and
  structure of gridded climate data from different providers, to facilitate
  interoperability, e.g. for the construction of multi-model ensembles or the
  comparison of forecasts with observations.
- `pydap-icechunk`: a server for exposing a "cooked data catalog" (see above)
  via the [OPeNDAP protocol](https://en.wikipedia.org/wiki/OPeNDAP). 

![architecture diagram](./architecture.svg)

## Initial setup for development/testing

- Check out this repo and your site's catalog repo as sibling directories. In
the case of forecast.ccsr.columbia.edu, the first Sigrid site, the catalog repo is
called ccsr-config.
- In this repo, copy `dot-env-example` to a new file called `.env`

        cp dot-env-example .env

- Edit the paths in `.env` to match your local configuration. Relative
paths and environment variables are not supported, so use absolute paths. Change
`PYDAP_PORT` to a unique number so you don't collide with other developers on
the same server.

