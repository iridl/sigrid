import argparse
import concurrent.futures
from concurrent.futures import Executor, Future
from dataclasses import dataclass
import enum
import importlib.util
import itertools
import os
from pathlib import Path
import re
from types import TracebackType
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence, TypeVar, cast
import warnings

import icechunk
import icechunk.session
import numpy as np
import tqdm
import xarray as xr
import xarray.conventions
import zarr
from zarr.errors import GroupNotFoundError, ZarrUserWarning


# In my opinion, using unsanctioned codecs makes the icechunk store inappropriate to
# expose publicly, because not all clients will have the libraries required to read it,
# but it's fine for internal use because we control the client.
warnings.filterwarnings("ignore", category=ZarrUserWarning, message='.*Numcodecs.*')

ALLOWED_ERROR_RATE = .1
PROGRESS_PERIOD = 1  # How often to print a progress update, in seconds

type Pathy = str | os.PathLike[str]

class TopConfig:
    def __init__(
            self,
            orig_root: Pathy,
            icechunk_root: Pathy,
            raw_catalog_root: Pathy,
    ) -> None:
        self.orig_root = Path(orig_root)
        self.icechunk_root = Path(icechunk_root)
        self.raw_catalog_root = Path(raw_catalog_root)

def config_from_env() -> TopConfig:
    config_vars: dict[str, str] = {}
    with open('../.env') as f:
        for line in f:
            if line.lstrip().startswith('#') or line.strip() == '':
                continue
            k, v = line.rstrip().split('=', 1)
            config_vars[k] = v
    config = TopConfig(
        config_vars['ORIG_DATA_ROOT'],
        config_vars['ICECHUNK_ROOT'],
        config_vars['RAW_CATALOG_ROOT']
    )
    return  config

class FileType(enum.StrEnum):
    nc3 = enum.auto()
    nc4 = enum.auto()

class URLPath:
    path: str
    def __init__(self, arg: "URLPath | str") -> None:
        if isinstance(arg, URLPath):
            self.path = arg.path
        else:
            self.path = arg.rstrip('/')

    @property
    def is_absolute(self) -> bool:
        return self.path.startswith('/')

    def __truediv__(self, arg: "URLPath | str") -> "URLPath":
        if isinstance(arg, URLPath):
            suffix = arg.path
        else:
            suffix = arg
        if suffix.startswith('/'):
            raise URLPathException("Can't append an absolute URLPath")
        return URLPath(f'{self.path}/{suffix}')

    def __str__(self) -> str:
        return self.path

class URLPathException(Exception):
    pass

@dataclass
class IcechunkInfo:
    relpath: str

@dataclass
class FileSetCatalog:
    _raw_catalog_root: Path
    _data_root: Path

    def get_entry(self, path_arg: str | URLPath) -> tuple["FileSetDescriptor", IcechunkInfo]:
        path_str = str(path_arg)
        dir, var_name = path_str.rsplit('/', maxsplit=1)
        index_path = self._raw_catalog_root / dir / 'index.py'
        spec = importlib.util.spec_from_file_location('catalog', index_path)
        assert spec  # TODO
        assert spec.loader  # TODO
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        defaults = dict(module.dataset)
        vars = defaults.pop('vars')
        var_dict = defaults | vars[var_name]
        kwargs = dict(
            var_dict,
            dir=self._data_root / var_dict['dir'],
            name=var_name,
            catalog_path = URLPath(path_str)
        )
        return FileSetDescriptor(**kwargs), IcechunkInfo(path_str)


class FileCoords(NamedTuple):
    t: np.datetime64
    m: int | None
    p: int | None


class DatasetCoords(NamedTuple):
    # T is Iterable, not Sequence, to leave open the possibility of not
    # enumerating all the T values at once, which might involve listing
    # millions of files.
    T: Iterable[np.datetime64]
    M: Sequence[int] | None
    P: Sequence[int] | None

T = TypeVar('T')

ABBREV_MONTH = ['', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

class FileSetDescriptor:
    """Instructions for combining a set of files into a higher-dimension array"""
    def __init__(
            self,
            *,
            name: str,
            dir: Path,
            catalog_path: URLPath,
            pattern: str,
            original_time_dim: str | None = None,
            parse_match : Callable[[dict[str, str]],FileCoords] | None = None,
            backend_kwargs: dict[str, dict[str, str]] | None = None,
            drop_vars: str | Sequence[str] = (),
            expand_coords: str | Sequence[str] = (),
            aux_coords: str | Sequence[str] = (),
    ) -> None:
        # TODO constructor args need to be validated
        self.name = name
        self.dir = dir
        self.catalog_path = catalog_path
        self._matcher = re.compile(pattern)
        self.parse_match = parse_match or default_parse_match
        self.backend_kwargs = backend_kwargs

        if isinstance(drop_vars, str):
            drop_vars = [drop_vars]
        if isinstance(expand_coords, str):
            expand_coords = [expand_coords]
        if isinstance(aux_coords, str):
            aux_coords = [aux_coords]
        self.opener = _FileOpener(
            original_time_dim, drop_vars, expand_coords, aux_coords
        )

    def parse_path(self, path: str | Path) -> FileCoords | None:
        match = self._matcher.search(str(path))
        if not match:
            return None
        return self.parse_match(match.groupdict())


@dataclass
class _FileOpener:
    original_time_dim: str | None
    drop_vars: Sequence[str]
    expand_coords: Sequence[str]
    aux_coords: Sequence[str]

    def open(self, path: Path, file_coords: FileCoords) -> xr.Dataset:
        """Use as a context manager or call close() on the dataset when finished with it."""
        ds = open_one_file(path)

        if self.drop_vars:
            ds = ds.drop_vars(self.drop_vars)

        # Data vars get expanded along IRIDL_time, coords don't unless they're
        # explicitly listed in expand_coords.
        # Note: this has no effect on which variables appear as coordinates
        # in the pydap response. That's controlled by the data vars' "coordinates"
        # attributes, which we set elsewhere.
        if self.aux_coords:
            ds = ds.set_coords(self.aux_coords)

        if file_coords.p is not None:
            ds = ds.expand_dims(P=[file_coords.p])
        if file_coords.m is not None:
            ds = ds.expand_dims(M=[file_coords.m])

        t_dim = self.original_time_dim
        if t_dim is None:
            ds = expand(ds, 'IRIDL_time', self.expand_coords)
        else:
            if t_dim in ds.dims:
                for v in ds.data_vars.values():
                    if v.dims[0] != t_dim:
                        raise Exception(f'{t_dim} is not the outermost dimension')
            elif t_dim in ds.coords:
                # promote scalar coordinate to a dimension coordinate
                assert ds.coords[t_dim].shape == ()
                ds = expand(ds, t_dim, self.expand_coords)
            else:
                raise Exception(f'No dimension named "{t_dim}" is present')

            # It's conceivable that someone might put multiple starts in a file,
            # but I haven't encountered that yet.
            assert len(ds[t_dim]) == 1, "Don't know how to handle multiple initializations in a single file."

            # Rename the dimension, but not the existing coordinate variable.
            ds = ds.rename_dims({t_dim: 'IRIDL_time'})

        # Now the IRIDL_time dimension is guaranteed to exist. Give it a coordinate variable.
        ds = ds.assign_coords({'IRIDL_time': ('IRIDL_time', [file_coords.t])})

        # The presence of a scalar coordinate causes to_zarr with region= to fail
        # TODO scalar coordinates that vary from file to file should be converted
        # into array coordinates with the appropriate dimensions; scalar coordinates
        # that are constant across the datasets should become scalar coordinates on
        # the dataset.
        ds = ds.drop_vars([c for c in ds.coords if ds.coords[c].dims == ()])

        # Make sure there's only one index for the IRIDL_time dimension. Having multiple
        # indexes on the same dimension causes problems with region=
        ds = ds.drop_indexes(
            (
                name for name, coord in coords_of(ds).items()
                if 'IRIDL_time' in coord.dims and coord.name != 'IRIDL_time'
            ),
            errors='ignore',
        )

        # No GMAO, we're not going to waste our disk space and bandwidth
        # storing your forecasts to nine decimal digits of precision.
        # TODO this goes against our general policy of sticking as closely
        # as possible to the provider's representation. If most providers
        # provide 32-bit data, how much does it really harm us if one uses
        # 64 bits? Pydap responses should be 32-bit, but we could do the
        # rounding in pydap instead?
        # Caution: this causes the data variables to be loaded into memory.
        for varname in ds.data_vars:
            if ds[varname].dtype == np.dtype('float64'):
                ds[varname] = ds[varname].astype(np.float32)

        return ds

class FileSetListing:
    def __init__(self, descriptor: FileSetDescriptor) -> None:
        # Scan all the file names to determine the T, M, and P coordinates.
        # TODO We only really need to do this the first time.
        # Once we've created the icechunk store, we can get the shape from
        # that, and only scan files that we actually need to read.
        self._paths: dict[FileCoords, Path] = {}
        for path in descriptor.dir.rglob('*'):
            if path.is_file():
                coords = descriptor.parse_path(str(path))
                if coords is not None:
                    self._paths[coords] = path
        self.coords = assemble_coords(self._paths.keys())

    def list_times(self, first: np.datetime64 | None = None) -> Iterable[np.datetime64]:
        vals = self.coords.T
        if first is not None:
            vals = [v for v in vals if v >= first]
        return vals
    
    def get_path(self, coords: FileCoords) -> Path | None:
        return self._paths.get(coords)


def default_parse_match(values: dict[str, str]) -> FileCoords:
    t = m = p = None
    if 'year' in values or 'month' in values or 'day' in values:
        if not ('year' in values and 'month' in values):
            raise Exception('Path contains only a partial date')
        year = int(values['year'])
        try:
            month = int(values['month'])
        except ValueError:
            month = ABBREV_MONTH.index(values['month'].lower())
        if 'day' in values:
            day = int(values['day'])
        else:
            day = 1
        t = np.datetime64(f'{year}-{month:02}-{day:02}')
    else:
        raise Exception('No date was retrieved from path')
    if 'member' in values:
        m = int(values['member'])
    if 'pressure' in values:
        p = int(values['pressure'])
    return FileCoords(t, m, p)
    

def assemble_coords(coords: Iterable[FileCoords]) -> DatasetCoords:
    t_vals: set[np.datetime64] = set()
    m_vals: set[int | None] = set()
    p_vals: set[int | None] = set()
    for coord in coords:
        t_vals.add(coord.t)
        m_vals.add(coord.m)
        p_vals.add(coord.p)
    def helper[T: (int, str)](x: set[T | None], reverse: bool = False) -> list[T] | None:
        if None in x:
            assert x == {None}
            return None
        return sorted(cast(set[T], x), reverse=reverse)
    return DatasetCoords(sorted(t_vals), helper(m_vals), helper(p_vals, reverse=True))

def expand(ds: xr.Dataset, dim: str, expand_coords: Iterable[str]) -> xr.Dataset:
    ds = ds.expand_dims(dim)
    for c in expand_coords:
        expanded = ds[c].expand_dims(dim)
        new_name = f'{c}_expanded'
        ds = ds.assign_coords({new_name: expanded})
        ds = ds.drop_vars(c)
        for other in coords_of(ds).values():
            if other.attrs.get('bounds') == c:
                other.attrs['bounds'] = new_name
    return ds

def update(
        session: icechunk.session.Session,
        descriptor: FileSetDescriptor,
        *,
        limit: int | None,
        first: np.datetime64 | None,
        parallel: int
) -> int:
    listing = FileSetListing(descriptor)
    try:
        existing = xr.open_zarr(session.store, zarr_format=3)
    # Icechunk store throws one error, Zarr store throws a different one.
    except (GroupNotFoundError, FileNotFoundError):
        # Exceptions thrown from inside an exception handler get confusing, so don't try creating the 
        # new store here.
        existing = None

    if existing is None:
        initialize(session, descriptor.opener, listing)
        existing = xr.open_zarr(session.store, zarr_format=3)

    times_to_fetch = (
        t for t in listing.list_times(first=first)
        if t not in existing['IRIDL_time']
        # TODO if we have the T-slice but it's missing some P or M, fill in gaps?
    )
    times_to_fetch = list(itertools.islice(times_to_fetch, limit))

    if len(times_to_fetch) == 0:
        return 0
        
    if len(existing['IRIDL_time']) > 0:
        last_old = existing['IRIDL_time'][-1]
        first_new = times_to_fetch[0]
        if first_new < last_old:
            raise Exception("Inserting prior to an existing time slice is not yet implemented.")

    # Resize arrays to prepare them to receive the new chunks. Typically done
    # with xarray + dask, but I'm dropping down to zarr to avoid dask.
    # Any array that has the IRIDL_time dimension needs to be extended.
    zgroup = zarr.open_group(session.store, mode='r+', zarr_format=3)
    for name in existing.variables: # includes both coord and data vars
        assert isinstance(name, str)
        if 'IRIDL_time' in existing[name].dims:
            assert existing[name].dims[0] == 'IRIDL_time'
            z = zgroup[name]
            assert isinstance(z, zarr.Array)
            new_shape = ((z.shape[0] + len(times_to_fetch)),) + z.shape[1:]
            z.resize(new_shape)

    # Populate the time coordinate variable with the new values.
    # I tried to do this with xr.Dataset.to_zarr(append_dim=), but that wiped out the
    # zarr group's attributes.
    # TODO are you sure? There was a separate problem causing attributes not to be saved.
    # Possibly relevant: https://github.com/pydata/xarray/issues/8755
    new_times = xr.Variable('IRIDL_time', times_to_fetch, encoding=existing['IRIDL_time'].encoding)
    new_times_encoded = xarray.conventions.encode_cf_variable(new_times)
    z_arr = zgroup['IRIDL_time']
    assert isinstance(z_arr, zarr.Array)
    z_arr[-len(new_times_encoded):] = new_times_encoded.values
    # refresh our xarray view of the existing array to reflect the extended time coord
    existing = xr.open_zarr(session.store, zarr_format=3)

    # Write new values
    fork_session = session.fork()
    if parallel > 0:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=parallel)
    else:
        # for debugging
        executor = SyncExecutor()
    print('Starting writes')
    try:
        futures = [
            executor.submit(
                write_one_file_slice,
                fork_session,
                descriptor.opener,
                auto_detect_region(file_coords, existing),
                file_coords,
                path,
            )
            for t in times_to_fetch
            for m in listing.coords.M or [None]
            for p in listing.coords.P or [None]
            if (path := listing.get_path(file_coords := FileCoords(t, m, p))) is not None
        ]
        total_count = len(futures)
        success_count = 0
        failure_count = 0
        not_done = futures
        with tqdm.tqdm(total=total_count) as pbar:
            while not_done:
                done, not_done = concurrent.futures.wait(not_done, timeout=PROGRESS_PERIOD)
                for f in done:
                    if e := f.exception():
                        failure_count += 1
                        print(e)
                    else:
                        success_count += 1
                        session.merge(f.result())
                completed_count = success_count + failure_count
                pbar.update(len(done))
                if (
                    completed_count > 0 and
                    failure_count / completed_count > ALLOWED_ERROR_RATE
                ):
                    raise Exception('Too many failures')

    finally:
        executor.shutdown(cancel_futures=True, wait=False)
    return success_count


class SyncExecutor(Executor):
    def submit[**P, T](
        self,
        fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future[T] = Future()
        try:
            print(fn, args, kwargs)
            # Executes immediately in the current thread
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def __enter__(self) -> "SyncExecutor":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        self.shutdown()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        pass


def write_one_file_slice(session: icechunk.session.ForkSession, opener: _FileOpener, region: Mapping[str, slice], file_coords: FileCoords, path: Path):
    with opener.open(path, file_coords) as ds:
        # Working around an xarray bug: when updating an existing zarr store, it
        # uses the _FillValue it finds in the store, and doesn't tolerate the ds
        # having one even if it's the same as the one in the store. So we keep this
        # attribute in initialize, which creates the store, but drop it here when
        # updating.
        for v in ds.variables:
            ds[v].attrs.pop('_FillValue', None)
            ds[v].attrs.pop('missing_value', None)
        ds.to_zarr(session.store, region=region, zarr_format=3, consolidated=False)
    return session


def initialize(session: icechunk.session.Session, opener: _FileOpener, listing: FileSetListing) -> None:
    t = next(iter(listing.coords.T))
    file_slices: list[list[xr.Dataset]] = [
        [
            # Out of laziness, I'm assuming for now that the initial
            # t slice has no missing files. If we need to deal with
            # that situation, we can insert empty slices, or move to
            # filling in values one file at a time.
            opener.open(
                raise_if_null(
                    listing.get_path(FileCoords(t, m, p)),
                    'Missing file in initial time slice',
                ),
                FileCoords(t, m, p)
            )            
            for p in listing.coords.P or [None]
        ]
        for m in listing.coords.M or [None]
    ]
    # Combine a list of lists of (m, p) slices into a single t slice.
    # Note that we test for coord is None, not for len(slice) == 1,
    # because there are cases where we want to keep a dimension that has
    # length 1 (e.g. SubC GEFSv12 zg).
    if listing.coords.P is None:
        m_slices = [m_slice[0] for m_slice in file_slices]
    else:
        m_slices = [
            xr.concat(m_slice, dim='P', coords='minimal')
            for m_slice in file_slices
        ]
    if listing.coords.M is None:
        t_slice = m_slices[0]
    else:
        t_slice = xr.concat(m_slices, dim='M', coords='minimal')

    one_file_slice = file_slices[0][0]
    encoding = {
        varname: {'chunks': tuple(da.sizes[dim] for dim in da.dims)}
        for varname, da in one_file_slice.data_vars.items()
    }
    t_slice.to_zarr(session.store, consolidated=False, encoding=encoding)


def open_one_file(path: Path) -> xr.Dataset:
    decode_coords_opt = True
    mask_and_scale_opt = False
    kwargs = {}
    if path.suffix == '.tif':
        decode_coords_opt = 'all'
        mask_and_scale_opt = True
        kwargs.update({'band_as_variable': True})
    try:
        result = xr.open_dataset(
            path,
            # Pass CF _FillValue and datetime/timedelta attributes through to
            # icechunk unchanged.
            mask_and_scale=mask_and_scale_opt,
            decode_times=False,
            # Note: decode_coords doesn't control decoding of coordinate values,
            # it controls which variables become coordinates as opposed to data
            # variables. That's important because expand_dims affects data vars
            # only.
            decode_coords=decode_coords_opt,
            **kwargs,
        )
        return result
    except Exception as e:
        e.add_note(f'While attempting to open {path}')
        raise


def get_repo(storage_factory: Callable[[str], icechunk.Storage], dir: Pathy, orig_root: Path | None = None) -> icechunk.Repository:
    storage = storage_factory(str(dir))
    if icechunk.Repository.exists(storage):
        repo = icechunk.Repository.open(storage)
    else:
        repo_config = icechunk.RepositoryConfig.default()
        print(f'Creating {dir}')
        repo = icechunk.Repository.create(
            storage,
            repo_config,
        )
        repo.save_config()

    return repo


def coords_of(ds: xr.Dataset | xr.DataArray):
    """A version of the coords attribute with a better type hint"""
    return cast(Mapping[str, xr.DataArray], ds.coords)


def raise_if_null(x: T | None, message: str) -> T:
    if x is None:
        raise Exception(message)
    return x


def auto_detect_region(slice_coords: FileCoords, dest: xr.Dataset):
    # Adapted from xarray's implementation of region='auto', but this is much faster
    # for the cases we encounter.
    externals = {'IRIDL_time': slice_coords.t, 'M': slice_coords.m, 'P': slice_coords.p}
    region: Mapping[str, slice] = {}
    for dim in dest.dims:
        # Xarray's type hints are inconsistent. Keys of Dataset.dims are Hashable,
        # but to_zarr(region=) only accepts str.
        assert isinstance(dim, str)
        val = externals.get(dim)
        if val is None:
            region[dim] = slice(None)
        else:
            index = dest.get_index(dim)
            idxs = index.get_indexer([val]) # type: ignore TODO

            if (idxs == -1).any():
                raise KeyError(
                    f"Not all values of coordinate '{dim}' in the new array were"
                    " found in the original store. Writing to a zarr region slice"
                    " requires that no dimensions or metadata are changed by the write."
                )

            if (np.diff(idxs) != 1).any():
                raise ValueError(
                    f"The auto-detected region of coordinate '{dim}' for writing new data"
                    " to the original store had non-contiguous indices. Writing to a zarr"
                    " region slice requires that the new data constitute a contiguous subset"
                    " of the original store."
                )
            region[dim] = slice(idxs[0], idxs[-1] + 1)
    return region


def open_icechunk(rel_path: str, decode_times: bool = True, decode_cf: bool = True):
    """Handy util for debugging in a REPL"""
    c = config_from_env()
    storage = icechunk.local_filesystem_storage(str(c.icechunk_root / rel_path))
    # Workaround for https://github.com/earth-mover/icechunk/issues/2105
    if not icechunk.Repository.exists(storage):
        raise Exception(f'No repository exists at {storage}')
    repo = icechunk.Repository.open(
        storage,
        authorize_virtual_chunk_access={f'file://{c.orig_root}/': None}
    )
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, zarr_format=3, decode_times=decode_times, decode_cf=decode_cf)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("var")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--first", type=np.datetime64)
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    top_config = config_from_env()

    raw_cat = FileSetCatalog(
        top_config.raw_catalog_root,
        top_config.orig_root
    )
    descriptor, icechunk_info = raw_cat.get_entry(args.var)
    repo = get_repo(
        icechunk.local_filesystem_storage,
        top_config.icechunk_root / icechunk_info.relpath,
        top_config.orig_root
    )
    session = repo.writable_session('main')
    new_count = update(session, descriptor, limit=args.limit, first=args.first, parallel=args.parallel)
    if new_count:
        session.commit(f'update from {descriptor.dir}')
    print(xr.open_zarr(session.store))


if __name__ == '__main__':
    main()