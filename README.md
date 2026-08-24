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
Linux 6.8.0-136-generic, and Python 3.13.14. Each entry is the best mean time
from five repetitions; result parity is checked before timing.

| case | mojo-editdistance | editdistance 0.8.1 | relative |
| --- | ---: | ---: | ---: |
| eval ASCII 64 vs 64 | 1.9 us | 3.6 us | 1.89x faster |
| eval ASCII 1,024 vs 1,024 | 154.7 us | 2681.3 us | 17.33x faster |
| eval ASCII 64 vs 100,000 | 396.7 us | 5364.9 us | 13.52x faster |
| eval Unicode 1,000 vs 1,000 | 154.7 us | 2745.8 us | 17.75x faster |
| eval integer lists 1,000 vs 1,000 | 345.6 us | 2599.0 us | 7.52x faster |
| criterion 2,000 chars, threshold 2 | 23.5 us | 167.9 us | 7.13x faster |

Short ASCII strings encode once to compact byte buffers, while bytes pairs use
their existing storage directly across the FFI boundary. When the shorter ASCII
or bytes input fits one 64-bit word, that path also handles arbitrarily long
text without allocating NumPy token, hash-table, or scratch buffers. Its Myers
kernel builds a stack-resident character mask table and clears it with
native-width SIMD stores. Short non-ASCII strings use UTF-32 buffers and SIMD
pattern scans with a scalar tail. The benchmark validates that both
implementations return the same result before timing them.

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
