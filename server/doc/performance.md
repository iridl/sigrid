# Performance notes

## Miscellaneous

Tests run from home, particularly over VPN, are generally bottlenecked by bandwidth.
When tuning server performance, send test queries from localhost, not over the internet.

## Memory allocation

We were seeing highly variable performance. The same request sometimes took a few seconds
and sometimes timed out after 60 seconds. Even at times when queries were slow, the disk
queue length (`aqu-sz` in `iostat`) was near zero and gunicorn queue length was zero
(`ss -xl 'src = /run/pydap/pydap.sock'`).

I used yappi to profile a program that processes the following test query, which originally came
from pycpt:
```
/NMME/ECCC/CanSIPS-IC4/hindcast/tos.dods?tos%5b15:12:413%5d%5b0:39%5d%5b3:5%5d%5b30:150%5d%5b0:180%5d
```
and found that more than half of the total time was being spent in `NDBuffer.__setitem__`.
That's bizarre, because all that method does is call `ndarray.__setitem__`, which is a native
function that just copies values from one array to another (at least when the src and dest
arrays have the same shape, dtype, and byte order, which was the case here).

Claude suggested instrumenting the method as follows:
```python
import resource, time
from zarr.core.buffer.cpu import NDBuffer

KEYS = ("pswpin", "pswpout", "pgscan_direct", "allocstall_normal", "allocstall_movable",
        "compact_stall", "thp_fault_alloc", "numa_pages_migrated")

def vm():
    d = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, v = line.split()
            if k in KEYS:
                d[k] = int(v)
    return d

_orig = NDBuffer.__setitem__

def timed(self, key, value):
    r0, v0, t0 = resource.getrusage(resource.RUSAGE_THREAD), vm(), time.perf_counter()
    _orig(self, key, value)
    dt = time.perf_counter() - t0
    r1, v1 = resource.getrusage(resource.RUSAGE_THREAD), vm()
    delta = {k: v1[k] - v0[k] for k in v1 if v1[k] != v0[k]}
    print(f"{dt*1e3:7.1f} ms sys {1e3*(r1.ru_stime-r0.ru_stime):7.1f} "
          f"minflt {r1.ru_minflt-r0.ru_minflt} majflt {r1.ru_majflt-r0.ru_majflt} {delta}")

NDBuffer.__setitem__ = timed
```
which resulted in some lines like
```
1612.2 ms sys  1590.7 minflt 424 majflt 0 {'allocstall_movable': 84, 'numa_pages_migrated': 1323, 'compact_stall': 168, 'thp_fault_alloc': 488}
1051.1 ms sys  1031.4 minflt 394 majflt 0 {'allocstall_movable': 277, 'pgscan_direct': 1024, 'numa_pages_migrated': 150, 'compact_stall': 558, 'thp_fault_alloc': 479}
```
which show that (a) a single call to `__setitem__` sometimes takes over a second and a half, most
of which is spent in the kernel, and (b) the kernel was doing a lot of THP compaction and reclamation
during that time (though not all of it was necessarily because of this program).

Running `echo defer > /sys/kernel/mm/transparent_hugepage/defrag` eliminated those stalls and sped up
the program by a factor of 5x. Prior to doing that, it was in mode `madvise`.

Interpretation: the cause of the slow requests was a combination of the following:
- The program allocates a lot of temporary arrays,
- The program uses `madvise(MADV_HUGEPAGE)` when allocating arrays, which tells the kernel that it
  should allocate pages of 2MB each rather than the usual 4KB,
- Memory was highly fragmented at that moment, so while there was plenty of free memory available,
  there were no 2MB contiguous blocks of it,
- The kernel was configured such that when the program made a hugepage request, and no free huge pages
  were available, the request would hang while the kernel tried to rearrange memory to create
  a large enough contiguous block.

Changing the defrag strategy to `defer` changed the kernel's behavior so that now, instead of
pausing the program while compaction occurs, it now immediately gives the program an allocation
made up of normal 4KB pages, and triggers compaction in the background.

Changing the defrag strategy was the quickest fix, but in time we should also look into other
aspects of the problem:
- The server should pre-allocate buffers and reuse them, rather than allocating and releasing
  many new buffers. The Zarr library supports this, but xarray doesn't, or not easily, so this
  may involve dropping xarray (or only using it for its CF-encoding functionality) and using Zarr
  directly.
- Understand why memory is so fragmented and compaction fails. I have a feeling it may have to do with
  the large rsync jobs we run, since they result in a large amout of memory used for page cache.
  This is handy when running the same rsync job multiple times (subsequent runs go much faster),
  but in general we don't need to do that, so maybe we should call rsync in a way that makes it not
  cache filesystem pages. Claude mentions a tool called `nocache` that calls `posix_fadvise(DONTNEED)`.
  Alternatively or in addition, we might tune the kernel to start freeing page cache earlier
  (`vm.min_free_kbytes`?). Also, the server is NUMA, and page cache is per-node, so pinning
  rsync (or whatever else is causing the memory fragmentation) to one node at least protects
  the other node from the fragmentation.
