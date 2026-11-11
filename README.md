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
from five repetitions; result parity is checked before timing. The two right
columns are the same case on the previous source tree and on the current
kernels; `EDITDISTANCE_LIBRARY=<lib> pixi run python bench/bench.py` selects the
library, so both sides run without a rebuild in between.

| case | mojo-editdistance | editdistance 0.8.1 | relative | previous kernels | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| eval ASCII 64 vs 64 | 1.9 us | 3.7 us | 1.93x faster | 2.0 us | none |
| eval ASCII 1,024 vs 1,024 | 149.9 us | 2719.0 us | 18.14x faster | 161.0 us | within noise |
| eval ASCII 64 vs 100,000 | 475.8 us | 5394.4 us | 11.34x faster | 490.5 us | within noise |
| eval Unicode 1,000 vs 1,000 | 148.3 us | 2699.4 us | 18.20x faster | 160.7 us | within noise |
| eval integer lists 1,000 vs 1,000 | 323.7 us | 2561.1 us | 7.91x faster | 349.7 us | within noise |
| criterion 2,000 chars, threshold 2 | 21.6 us | 164.5 us | 7.63x faster | 23.2 us | none |
| criterion 2,000 chars, threshold 16 | 27.9 us | 754.9 us | 27.05x faster | 31.1 us | 1.07x faster |

The eval rows and threshold 2 move by several percent between runs on this
shared machine, and no run showed a repeatable gap for them, so they are
reported as unchanged. The interleaved measurement in `bench/ab.py` (both
libraries in one process, alternating batches) separates them cleanly: threshold
16 is 1.061x, and every other row is between 0.997x and 1.010x. A sweep over
thresholds also shows the gain widening with the band width - about 1.04x at 8,
1.09x at 16, 1.13x at 32 and 1.17x at 64 - because the branch that was removed
ran once per band cell.

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
| 8,192 | 126 ms |
| 16,384 | 500 ms |

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
revert, is in "Parallel pass" above. A GPU path would face the same sequencing
and would not be justified.

The exported Mojo functions live in one compilation unit to keep build cost
fixed. They use a C ABI and are loaded by Python through `ctypes`.

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
