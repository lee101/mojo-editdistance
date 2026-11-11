# Kernel analysis: is any kernel compute-heavy enough for a GPU?

This file did not exist before the GPU phase. It is reconstructed from the shipped
kernels in `src/editdistance.mojo`, not invented: every operation count below is
read off the recurrence in that file, and every timing is measured in this
checkout on the machine named below.

"Bytes" means the thread's `int64` token buffers crossing the FFI per call
(`8 * (n + m)`) - what would have to be copied to a device each way. "Operations"
stands in for flops: this package has no floating point, every score is an integer.
The GPU gate in `prompts/accel.md` is roughly 2 operations per byte.

## Arithmetic intensity of each kernel shape

| shape | bytes | operations | ops/byte | where it comes from |
| --- | ---: | ---: | ---: | --- |
| bit kernel, 1,024 vs 1,024 | 16,384 | 4,194,304 | 256.0 | `_distance_bit`, `words = 16` |
| bit kernel, 64 vs 100,000 | 800,512 | 900,000 | 1.12 | `_distance_bit` |
| criterion, 2,000 chars, threshold 16 | 32,000 | 256,000 | 8.0 | `_within`, band width `2*16 + 2` |
| DP fallback, 16,384 vs 16,384 | 262,144 | 1,073,741,824 | 4,096 | `_distance_dp`, ~4 ops/cell |
| DP fallback, 100,000 vs 100,000 | 1,600,000 | 40,000,000,000 | 25,000 | `_distance_dp`, ~4 ops/cell |

The two working kernels sit on opposite sides of the gate. The bit kernel is
`O(n + m*n/64)`: for a short pattern against long text its bytes and its operations
both grow with `m`, at about 1.1 operations per byte - **below the gate**. For equal
lengths its work grows with `n*m` while bytes grow with `n`, so 1,024 vs 1,024 is
256 operations per byte. The criterion band is compute-bound but tiny. The
Wagner-Fischer fallback is the most compute-bound shape of all, and it is the path
taken above 1,024 tokens - where the Myers kernels hand over.

## Measured GPU facts (NVIDIA GeForce RTX 5090, shared host, November 2026)

`nvidia-smi --query-gpu=memory.free` reported 4,860-5,962 MiB over the session,
always above the 4,000 MiB floor, so the skip did not apply. The largest candidate
(16,384 vs 16,384 DP) needs two 128 KiB token buffers and one 128 KiB row buffer -
384 KiB, far inside the 2 GB cap.

`bench/gpu_probe.py` was run in a scratch `mojo + max` environment, because the
repository's own pixi environment does not carry `max`. The probe launches one
`_inc` kernel over 1,000,003 elements with a 256-thread grid (3,907 blocks, last
block partial) and reports `tail ok: True` for the last element.

| measurement | result |
| --- | --- |
| `DeviceContext()` / device present | `NVIDIA GeForce RTX 5090` |
| kernel launch + sync round trip | 6.8-12.7 us, best of 20 |
| host->device + device->host + sync, by size | see below |

Host<->device round trip, best of 20, steady-state values (see the note below):

| tokens each way | bytes each way | round trip |
| ---: | ---: | ---: |
| 16,384 | 128 KiB | 16 us |
| 65,536 | 512 KiB | 42 us |
| 262,144 | 2 MiB | 123 us |
| 1,048,576 | 8 MiB | 419 us |
| 4,194,304 | 32 MiB | 1155 us |
| 16,777,216 | 128 MiB | 4179 us |

That is about 8 GB/s both ways at 128 MiB, two orders below device bandwidth: the
copy cost is host-side `memcpy`, not the link.

**Measurement caveat, recorded rather than hidden.** The small-copy rows are bimodal
on this shared host. Twenty sequential cold invocations of the 128 KiB round trip
gave 8.5, 8.5, 8.3, 8.3, 8.2, 8.1, 10.0, 10.0, 10.4, 10.5, 10.7, 309, 348, 549, 608,
1302, 1386, 2236, 2270, 2269 us - six of twenty sat at hundreds of microseconds to
2.3 ms, with 4,860-5,962 MiB free GPU memory and a simple kernel that makes no
allocation calls; the bimodality persisted across fresh processes. The table shows
the fast mode. The slow mode only strengthens the conclusion below, so no decision
rests on choosing the smaller number.

## The comparison at real caller sizes

Each call would cross once - one copy in, one launch, one copy out - so it pays the
whole round trip. The CPU it competes with, measured in this checkout with
`pixi run python bench/bench.py` and `bench/gpu_sizes.py`:

| case | tokens | CPU today | GPU transfer floor | transfer as share of CPU |
| --- | ---: | ---: | ---: | ---: |
| eval ASCII 1,024 vs 1,024 (bit kernel) | 2,048 | 143-170 us | 16 KiB: ~16 us | ~9-11% |
| criterion 2,000, threshold 16 | 4,000 | 23-27 us | 32 KiB: ~30 us | ~110-130% |
| eval ASCII 64 vs 100,000 (bit kernel) | 100,064 | 484-488 us | 800 KiB: ~60 us | ~12%, but **below the gate** (1.12 ops/byte) |
| eval integer lists 16,384 vs 16,384 (DP fallback) | 32,768 | 508-520 ms | 256 KiB: ~17 us | ~0.03% |

Read row by row:

- **64 vs 100,000** is ruled out on intensity: 1.12 operations per byte, whatever
  the transfer costs.
- **criterion 2,000 / threshold 16 and below** is 23-27 us end to end and the round
  trip alone is ~30 us. Nothing to win.
- **1,024 vs 1,024** is the only interesting bit-kernel case: 143-170 us of CPU work
  against ~16 us of transfer. But the dependency that makes it cheap on the CPU also
  blocks it on the GPU: `vp`/`vn` carry from one text position to the next, and the
  text scan is `m` sequential steps. Splitting those over blocks needs one launch per
  position (forbidden by "one kernel launch per call") or a cross-block dependency
  the rules exclude. It also depends on a token->mask table built on the host.
- **DP fallback 16,384** is the only shape where the transfer is invisible. But it is
  the path real callers do not take - any `eval` of that size through the public API
  is routed to the bit kernel above 1,024 tokens - and Wagner-Fischer has no
  parallelism across rows, so one whole-input launch is a serial chain of band steps.

## Conclusion

| path | verdict |
| --- | --- |
| bit kernel, short pattern vs long text | **CPU only: memory-bound** (1.12 ops/byte, under the gate) |
| bit kernel, equal 1,024s | compute-bound, but the text scan is a serial dependency chain; no implementation of the required shape exists |
| criterion band | compute-bound, but the round trip costs more than the whole call at sizes callers pass |
| Wagner-Fischer fallback | compute-bound, but it is the fallback for sizes real callers never reach, and it has no row parallelism |

No GPU path ships. `bench/gpu_probe.py` and `bench/gpu_sizes.py` keep the GPU cost
and the CPU comparison reproducible on the next machine.
