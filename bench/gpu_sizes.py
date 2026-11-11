"""CPU cost at the sizes the package's callers pass, for judging a GPU path.

Timing only; call ``pixi run python bench/gpu_sizes.py``. These are the rows the
GPU pass in the README compares against the host<->device round trip measured on
this machine (4.7 ms for 16 MiB each way, 12 us per launch). The Wagner-Fischer
fallback is timed separately at a size where its work is not trimmed away.
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import editdistance  # noqa: E402
import numpy as np  # noqa: E402


def best_time(function, loops: int, repeat: int = 5) -> float:
    function()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(loops):
            function()
        best = min(best, (time.perf_counter() - start) / loops)
    return best


def main() -> None:
    print("| case | tokens | CPU time | loops |")
    print("| --- | ---: | ---: | ---: |")
    text_100k = "abcdefgh" * 12_500
    ascii_64 = text_100k[:64]
    for name, tokens, fn, loops in (
        (
            "eval ASCII 64 vs 100,000",
            100_064,
            lambda: editdistance.eval(ascii_64, text_100k),
            100,
        ),
        (
            "eval ASCII 1,024 vs 1,024",
            2_048,
            lambda: editdistance.eval(text_100k[:1024], text_100k[1024:2048]),
            100,
        ),
        (
            "criterion 2,000, threshold 16",
            4_000,
            lambda: editdistance.eval_criterion(
                text_100k[:2000], text_100k[2000:4000], 16
            ),
            10,
        ),
    ):
        print(f"| {name} | {tokens} | {best_time(fn, loops) * 1e6:.1f} us | {loops} |")

    # The DP fallback only runs above 1,024 tokens; random tokens with no
    # common prefix or suffix keep its full n*m work.
    rng = np.random.default_rng(11)
    for n in (4096, 16384):
        a = rng.integers(0, 8, n, dtype=np.int64)
        b = rng.integers(0, 8, n, dtype=np.int64)
        a[0], b[0], a[-1], b[-1] = 100, 101, 102, 103
        seconds = best_time(lambda: editdistance.eval(a, b), 1, repeat=3)
        print(f"| eval integer lists {n} vs {n} (no trim) | {2 * n} | {seconds * 1e3:.1f} ms | 1 |")


if __name__ == "__main__":
    main()
