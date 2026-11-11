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

There is no threaded or GPU path. Myers advances through the text with a loop-
carried state, while each Wagner-Fischer cell depends on its left, upper, and
diagonal neighbors. No current kernel combines enough independent work with
high arithmetic intensity to offset thread, GPU launch, and transfer overhead,
so a GPU path would not be justified.

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
