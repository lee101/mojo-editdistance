# mojo-editdistance

A Mojo implementation of unit-cost Levenshtein distance with a tested subset of
the Python API from [`editdistance`](https://pypi.org/project/editdistance/).
It is intended for workloads where the strings or token sequences are large
enough for a compiled bit-parallel kernel to matter.

The import name and all four top-level callables in `editdistance` 0.8.1 match:

| function | status |
| --- | --- |
| `eval(a, b)` | covered |
| `distance(a, b)` | covered |
| `eval_criterion(a, b, thr)` | covered |
| `distance_le_than(a, b, thr)` | covered |

Strings, bytes, bytearrays, and indexable sequences of hashable objects are
covered. Tests compare every function directly with the conda-forge
`editdistance` 0.8.1 package, including Unicode and lone surrogates, embedded
zero bytes, NumPy arrays, hash collisions, thresholds and invalid threshold
errors, SIMD tails, 64-bit word boundaries, and the long-pattern fallback. As
in upstream 0.8.1, generic elements are compared by hash token, and a zero
threshold returns `False` for two non-empty sequences, even when equal.

## Install

The repository pins the tested Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-editdistance.so`. Importing from the
checkout also rebuilds a missing or stale library automatically.

## Usage

From the repository checkout, the module is a replacement for the covered
upstream API:

```python
import editdistance

editdistance.eval("kitten", "sitting")             # 3
editdistance.distance([1, 2, 3], [1, 4, 3])       # 1
editdistance.eval_criterion("banana", "bahama", 2) # True
```

Run the example from this checkout with:

```bash
pixi run python -c \
  'import editdistance; print(editdistance.eval("kitten", "sitting"))'
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-147-generic, and Python 3.13.14. Each entry is the best mean time
from five repetitions; result parity is checked before timing. All seven rows
below are from one `pixi run bench` run in the verification pass (load average
18-31 on 72 CPUs); see `bench/verify.md` for that run, the five other runs, and
the figures the table carried before.

| case | mojo-editdistance | editdistance 0.8.1 | relative |
| --- | ---: | ---: | ---: |
| eval ASCII 64 vs 64 | 2.1 us | 4.1 us | 1.93x faster |
| eval ASCII 1,024 vs 1,024 | 148.0 us | 2848.3 us | 19.25x faster |
| eval ASCII 64 vs 100,000 | 496.7 us | 5748.1 us | 11.57x faster |
| eval Unicode 1,000 vs 1,000 | 144.8 us | 2779.0 us | 19.19x faster |
| eval integer lists 1,000 vs 1,000 | 339.1 us | 2658.9 us | 7.84x faster |
| criterion 2,000 chars, threshold 2 | 22.1 us | 161.3 us | 7.29x faster |
| criterion 2,000 chars, threshold 16 | 28.2 us | 752.6 us | 26.65x faster |

Both sides run the same case in one process: the same argument objects are
passed to each implementation, both return the same result, which is checked
before any timing, and both are wrapped in the same `time.perf_counter` loop.
One asymmetry is worth stating because it is the only place the two sides run
different amounts of work: the criterion kernels strip common prefix and
suffix, and `editdistance` 0.8.1 never does. The threshold 2 case shares a
1,988-character suffix, so the trimmed problem this port solves there is 12 by 12
while upstream fills the full 2,000-wide band. Timing this port on the same
inputs with that shared run destroyed, or with no shared run at all, moved the
row by at most 6% in either direction (measured in `bench/verify.md`), so the
row is not driven by the trimming, but it does not isolate the kernel either.

The eval rows move by several percent between runs on this shared machine. The
previous version of this table made claims that measurement did not support,
and they have been removed: a threshold sweep from 1.04x at 8 to 1.17x at 64,
and a "previous kernels" column. The sweep was not reproduced in the
verification pass, and the previous-tree library it was measured against was
never kept. The one previous-kernel figure that could be rebuilt was in error:
rebuilding the kernel from the commit before the unroll pass (which predates the
SIMD pass as well) and interleaving it with this checkout's library gives 30.1 us
against 28.6 us at threshold 16 (1.05x, not 1.07x), and the same runs move the
1,024-token eval rows by 7-8% (1.076-1.080x, not "within noise"). Details in
`bench/verify.md`.

Wrong kernels are easy to benchmark: an early attempt in this pass wrote mask
words past the end of the table for one-word patterns and still returned
plausible distances, so `bench/differential.py` now compares thousands of random
pairs across every kernel path with `editdistance` 0.8.1 before any number is
trusted.

Short ASCII strings encode once to compact byte buffers, while bytes pairs use
their existing storage directly across the FFI boundary. When the shorter ASCII
or bytes input fits one 64-bit word, that path also handles arbitrarily long
text without allocating NumPy token, hash-table, or scratch buffers. Its Myers
kernel builds a stack-resident character mask table and clears it with
native-width SIMD stores. Short non-ASCII strings use UTF-32 buffers and SIMD
pattern scans with a scalar tail. The benchmark validates that both
implementations return the same result before timing them.

### Unrolling pass (single-threaded kernels)

The kernels are branch-limited rather than throughput-limited, so the loop
carried dependency in the bit-vector recurrence is what sets the cost. What
landed:

- The criterion band kernel no longer re-tests `best > infinity` for every
  cell. A cell is exact unless its true distance exceeds the threshold, and any
  cell computed from a capped input is above the threshold too, so capping and
  not capping agree on `distance <= threshold` and on whether an entire band can
  still reach it. The row still reports `row[bm]` exactly when the answer is
  within the threshold. Checked against a reference over 122,645 random and
  exhaustive sequence/threshold pairs.
- `a[a_start + i - 1]` is loaded once per band row instead of once per cell.

Tested, measured, and reverted because the benchmark did not show a gain at the
real caller sizes: a dedicated single-register loop for one-word bit patterns
(its deferred-carry register version broke parity, and the plain version won
only inside noise), merging the two table clears in the bit kernel (11-13%
slower at 1,024 tokens), an explicitly hoisted `index * words` base (0.997-1.010x
in the interleaved measurement), and the same row-invariant hoist in the full
Wagner-Fischer fallback (1.000x).

## Parallel pass: measured, not shipped

`max.algorithm.parallelize` was tried behind a size threshold and removed again:
it loses to the serial kernel at every size this package's callers reach, so the
shipped library is still single-threaded. `max` was added to `pixi.toml` for the
experiment and then removed, so the dependencies are unchanged.

Toolchain: `mojo ==1.2.0.dev2026092605` with `max ==26.7.0.dev2026092605`, the
pair `MOJO_NOTES.md` section 5b pins together. `from max.algorithm import
parallelize` compiled and ran from the ctypes-loaded shared library; only the
`std.algorithm` and `std.threading` forms are broken, and none were used.

Where parallel work could go. The Myers bit-vector kernels (short strings, and
patterns up to 1,024 tokens) carry `vp`/`vn` from one text position to the next,
so their text scan cannot be split. The only independent work is the
Wagner-Fischer fallback used above 1,024 tokens, which fills one row at a time
with the cells of a row independent of each other. The experiment therefore
split each row's columns across workers with one `parallelize` call per row, and
compared it against `_distance_dp` on identical input. Common-prefix and suffix
trimming was left out of the parallel variant so the measured cost is the DP
itself.

Serial cost first (`bench/sizes.py`, `eval` of two random strings of equal
length, best of five batches):

| n | serial |
| ---: | ---: |
| 1,024 | 169 us |
| 2,048 | 7.8 ms |
| 4,096 | 30.8 ms |
| 8,192 | 141 ms |
| 16,384 | 495 ms |

The crossover measurement had to clear two obstacles before it could be trusted,
and neither could be removed on this toolchain:

- **The per-call `parallelize` overhead dominates at these sizes.** An empty
  body at 2, 4, 8 and 16 workers measured 4.7-5.7 ms per call on this host,
  independent of the work item count in that run, and slow in the
  same way whether the calls came from a tight loop inside one shared-library
  function or from a Python loop over ctypes calls; at one worker the same
  shape costs about 0.05 us per 16,000 items. That is more than the whole
  serial DP costs up to about n=1,024, so splitting a row must pay the
  overhead before any cell work is saved. The same loop measured 2.1-3.4 s per
  call in a later run under heavier host load, so the absolute figure is not
  stable.
- **No usable row barrier was found.** Variants with `parallelize(func, items,
  workers)` called once per row, compared with the serial kernel on the same
  input: the first reproduction matched at 1 and 2 workers and differed at 3, 4,
  8 and 16 (serial 111 against 110-115); that small case rerun per worker count
  then failed on 14 of its 80 invocations (five runs of 16 worker counts), with
  a different wrong
  value on each run of the same input. The pattern is consistent with
  row-to-row order not being enforced - the next row sometimes reads a cell its
  worker has not written yet - but that was not proven by inspection. No other
  export in this toolchain's `max.algorithm` synchronises worker tasks: it
  exports `parallelize` (two overloads) and nothing else. `std.threading` does
  not exist on 1.2.0 (`MOJO_NOTES.md` section 4), so there is no barrier to
  build on.

Measured rows, the DP kernel on 128-token random integer sequences on the
experimental build (one run, host load average ~15 on 72 cores), serial versus
the same kernel at 1, 2, 4, 8 and 16 workers:

| configuration | n=128 | result |
| --- | ---: | --- |
| serial Wagner-Fischer | 54-92 us | matches reference |
| 1 worker `parallelize` per row | 74-105 us | matches; 0.81-1.24x of serial |
| 2 workers | ~2.1 s per call | the slowest launch case, and later the same build also disagreed with the serial kernel (173-175 against 111) |
| 4 workers | wrong result (112 against 111) | |
| 8 workers | wrong result (102 against 111); one run also took 3.4 s | |
| 16 workers | wrong result (113-114 against 111) | |

The 2-worker figure is per call in a tight loop on this host under load (~95
runnable processes); it is not a per-row cost. The one-worker row was measured
twice: in one run it beat the serial kernel (74 us against 92 us), in another it
lost (105 us against 54 us), so it is parity within noise, not a gain. Nothing
here approaches a win: the best case is one worker at parity, and every higher
count is both slower and wrong. Under the phase rule a parallel path that loses
to serial at the sizes real callers pass is worse than none, so it was reverted
rather than shipped with a threshold. A size threshold could not have saved it
either - the defect is the missing barrier, not the threshold - and sizes above
1,024 tokens are only reached by `eval`, which already runs in linear time on
the Myers kernel at those sizes. A correct barrier would still not make the
split pay: the per-call cost above exceeds the whole serial DP at n=2,048, so
only much larger cell counts could amortise it, and the result would say more
about thread-pool overhead than about this kernel.

Before and after for the shipped library are therefore identical: the README
table above still stands, and `pixi run build && pixi run test` passes with 30
tests on the unchanged serial kernels. `bench/sizes.py` is new and exists so a
future attempt can reuse the serial curve and the size sweep.

Two techniques from the brief do not apply here. There is no floating-point
accumulation in this package - every score is an integer - so multi-accumulator
reduction has nothing to act on, and the one SIMD `reduce_add` is an integer
popcount whose order is irrelevant. Compile-time unrolling via `@parameter for`
is rejected by the pinned compiler, and every hot loop here has a runtime trip
count anyway; `MOJO_NOTES.md` records both facts and the `@__parameter` spelling
that does work for compile-time counts.

## How it works

Outside the short-pair fast path, Python converts sequence elements to
contiguous `int64` tokens. Strings use their Unicode code points, bytes use byte
values, and generic sequences use the per-element hashes used by upstream.
NumPy-owned buffers and scratch arrays cross the C ABI as validated pointers;
the Mojo wrappers reconstruct them from addresses but neither own nor free
Python memory.

For patterns up to 1,024 tokens, Mojo builds a call-local open-addressed token
map and runs a multiword Myers bit-vector algorithm. This processes 64 pattern
positions per machine word. Longer patterns use a two-row-equivalent,
single-buffer Wagner-Fischer fallback with common-prefix and common-suffix
trimming, keeping auxiliary memory linear in the shorter input. Criterion calls
use a threshold-width DP band, so small thresholds avoid filling the rest of
the matrix.

There is no threaded or GPU path. The kernels have no usable thread barrier on
this toolchain - `max.algorithm` exports only `parallelize`, and
`std.threading` does not exist on 1.2.0 - so the Wagner-Fischer fallback could
not split a row correctly, and the Myers kernels are sequential by construction.
Even with a barrier, `parallelize` measured about 5 ms per call regardless of the work
size, above the serial kernel's cost for the sizes this package is used at. The
measurement, including the sizes swept and the wrong results that motivated the
revert, is in "Parallel pass" above. The GPU phase reached the same conclusion
from measured transfer cost rather than from the threading workaround; it is in
"GPU pass" below.

The exported Mojo functions live in one compilation unit to keep build cost
fixed. They use a C ABI and are loaded by Python through `ctypes`.

## GPU pass: measured, not shipped

The gate was run first, against `bench/analysis.md`. That file did not exist
before this pass and has been reconstructed from the kernels in
`src/editdistance.mojo`; it lists each shape's arithmetic intensity and the
argument for and against a device path.

**Gates.** `nvidia-smi --query-gpu=memory.free` reported 4,860-5,962 MiB over the
session, above the 4,000 MiB floor, so the phase was not skipped. The largest
candidate needs 384 KiB of device memory, far inside the 2 GB cap. **No device
allocation or launch was made from the shipped library**: nothing in `src/` or
`python/` imports `max`, and `pixi.toml`/`pixi.lock` are unchanged.

**What was actually exercised.** `bench/gpu_probe.py` ran in a scratch
`mojo==1.2.0.dev2026092605` + `max==26.7.0.dev2026092605` environment (the
repository's own pixi environment does not carry `max`, so `from max.gpu ...` fails
there; see `MOJO_NOTES.md` section 5b). It constructed a `DeviceContext` on the
`NVIDIA GeForce RTX 5090`, copies a 1,000,003-element buffer in, launches one
`_inc` kernel across a 3,907-block grid (last block partial), copies back, and
reports `tail ok: True`. The tail is explicit and checked, not assumed.

**Why nothing ships.** The two kernels real callers reach are on either side of
the 2-operations-per-byte gate, and neither side produces a shippable win:

| path | intensity | verdict |
| --- | ---: | --- |
| bit kernel, 64 vs 100,000 | 1.12 ops/byte | **CPU only: memory-bound** |
| bit kernel, 1,024 vs 1,024 | 256 ops/byte | 139-170 us CPU vs ~16 us transfer; but the text scan carries `vp`/`vn` per position, so the required "one launch, copies outside the loop" shape does not exist for it |
| criterion, 2,000 / threshold 16 | 8.0 ops/byte | whole call is 22-30 us; the round trip alone is ~30 us |
| Wagner-Fischer 16,384 vs 16,384 | 4,096 ops/byte | transfer is ~17 us against 508-536 ms CPU, but this is the fallback path: real callers of that size are routed to the bit kernel, and the DP has no row parallelism |

The measured host<->device round trip (copy in + copy out + sync, best of 20) was
128 KiB: 16 us; 512 KiB: 42 us; 2 MiB: 123 us; 8 MiB: 419 us; 32 MiB: 1155 us;
128 MiB: 4179 us - about 8 GB/s both ways, i.e. host `memcpy` cost. A kernel launch
plus sync round trip measured 6.8-12.7 us.

A caveat that is part of the record: the sub-MiB round-trip numbers are bimodal on
this shared host. Twenty cold invocations of the 128 KiB trip gave 12 runs of
8.1-10.7 us and 8 runs of 309-2270 us, with GPU memory free unchanged and a kernel
making no allocations; the table above quotes the fast mode. The slow mode makes
the GPU case worse, never better, so no decision here depends on the choice.

CPU comparison for the same calls, in this checkout
(`pixi run python bench/gpu_sizes.py`):

| case | CPU today |
| --- | ---: |
| eval ASCII 64 vs 100,000 | 492 us |
| eval ASCII 1,024 vs 1,024 | 139-170 us |
| criterion 2,000, threshold 16 | 22-31 us |
| eval integer lists 4,096 vs 4,096 (no trim) | 33.3 ms |
| eval integer lists 16,384 vs 16,384 (no trim) | 536 ms |

**Before and after are identical.** No shipped code changed, so the benchmark table
above still stands unchanged; this pass adds `bench/analysis.md`,
`bench/gpu_probe.py`, `bench/gpu_sizes.py` and `tests/test_gpu.py`, plus a
`gpu-probe` task in `pixi.toml` that is not part of `build` or `test`.

**Test results.** `pixi run test` reports `31 passed, 3 skipped`: the 30 unchanged
tests plus one CPU-fallback test pass, and the 3 GPU tests are **skipped** because
`EDITDISTANCE_GPU_PROBE` is not set in the repository environment (which has no
`max`).
Setting `EDITDISTANCE_GPU_PROBE` to a `mojo` binary from a `max` environment
runs all 4 GPU-file tests against the real device. On a host without a GPU the
probe exits 2 and the tests skip with "no usable GPU device" rather than fail.
To reproduce:

```bash
pixi init /tmp/gpu && cd /tmp/gpu
pixi add 'mojo ==1.2.0.dev2026092605' 'max ==26.7.0.dev2026092605'
cd /path/to/mojo-editdistance
EDITDISTANCE_GPU_PROBE=/tmp/gpu/.pixi/envs/default/bin/mojo pixi run test
```

`MOJO_NOTES.md` section 5's GPU sample is stale on this toolchain and the probe
uses the shapes the compiler accepts: `enqueue_create_buffer[DType.int64](n)` (not
`[Int64]`), `grid_dim=`/`block_dim=` keyword arguments, `block_idx.x` /
`thread_idx.x` (not `.value`, and `Int(block_idx)` is host-only), kernel arguments
as fixed-width `Int64` (`Int` is not `DevicePassable`), and `enqueue_copy(dst, src)`.
`bench/gpu_probe.py` documents each of these at the point it matters.

**One CPU number worth flagging.** The 16,384 by 16,384 DP fallback was
re-measured in the verification pass: `eval` on random integer arrays with no
common prefix or suffix took 510.4 ms, and `bench/sizes.py` reports
495,481.7 us (495.5 ms) for the same size. The 8,192 row in the parallel-pass
table said 126 ms, which would make the doubling 4x rather than 2x; re-running
`bench/sizes.py` gives 140,679.4 us for 8,192 and 495,481.7 us for 16,384, a
3.5x doubling. The 126 ms figure is withdrawn and `bench/sizes.py`'s measured
141 ms is used instead. The earlier standalone "faster" attempt is still
recorded as wrong: it returned 15,544 where the plain loop returns 11,374, so
it was not faster, it was incorrect. The DP cost is therefore measured, not
assumed, and nothing here derives a GPU win from it.

## Not covered

Compatibility is limited to the four functions and input categories listed
above. Iterables without `len()` and integer indexing are not accepted. This
project only implements insertion, deletion, and substitution with cost one;
it does not provide weighted costs, edit scripts or alignments,
Damerau-Levenshtein transpositions, approximate substring search, or a batch
matrix API. The upstream extension's non-public `bycython.eval_dp` helper is
outside the compatibility surface.

## License

MIT.
