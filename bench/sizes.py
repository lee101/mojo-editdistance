"""Serial-kernel cost by input size, for choosing the parallel threshold.

Timing only; call ``pixi run python bench/sizes.py``. Each row is the best of
five repetitions of a batch of calls to ``editdistance.eval`` on the sizes the
kernels switch between, so the crossover point for a size threshold is visible
without the upstream comparison in ``bench.py``.
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import editdistance  # noqa: E402


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
    print("| case | n | m | time |")
    print("| --- | ---: | ---: | ---: |")
    for kind, alphabet in (
        ("ascii", "abcdefgh"),
        ("unicode", "αβγδε日本語\U00010400"),
    ):
        for n in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            a = "".join(alphabet[i % len(alphabet)] for i in range(n))
            b = "".join(alphabet[(i + 1) % len(alphabet)] for i in range(n))
            seconds = best_time(lambda: editdistance.eval(a, b), 3 if n > 4096 else 20)
            print(f"| eval {kind} n x n | {n} | {n} | {seconds * 1e6:.1f} us |")
        for m in (1000, 10000, 100000, 1000000):
            a = "".join(alphabet[i % len(alphabet)] for i in range(64))
            b = "".join(alphabet[(i + 1) % len(alphabet)] for i in range(m))
            seconds = best_time(lambda: editdistance.eval(a, b), 5)
            print(f"| eval {kind} 64 x m | 64 | {m} | {seconds * 1e6:.1f} us |")


if __name__ == "__main__":
    main()
