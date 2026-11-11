"""Interleaved A/B comparison of two kernel builds.

Loads two shared libraries in one process and alternates timing batches, so a
noisy or busy machine cannot make one side look faster than the other. Results
are checked for equality before any timing happens.

Usage:
    pixi run python bench/ab.py OLD.so NEW.so [rounds]

The case list is exactly the one in ``bench/bench.py``, so the reported ratio
applies to the README table rows. Two sequential best-of-five runs of the same
pair agree only to a few percent on a shared machine; this agrees to about one.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from editdistance import _lib  # noqa: E402


def load(path: str):
    library = ctypes.CDLL(path)
    for name, (args, restype) in _lib._SIGNATURES.items():
        function = getattr(library, name)
        function.argtypes = args
        function.restype = restype
    return library


def load_cases():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
    spec = importlib.util.spec_from_file_location("med_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cases()


def main() -> None:
    old = load(sys.argv[1])
    new = load(sys.argv[2])
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    print("| case | old us | new us | ratio old/new |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream, _ in load_cases():
        def call(library):
            _lib._library = library
            return ours()

        if call(old) != call(new):
            raise SystemExit(f"{name}: results differ between builds")
        loops = 1
        start = time.perf_counter()
        while time.perf_counter() - start < 0.02:
            for _ in range(loops):
                call(old)
            loops *= 2
        old_best = new_best = float("inf")
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(loops):
                call(old)
            old_best = min(old_best, (time.perf_counter() - start) / loops)
            start = time.perf_counter()
            for _ in range(loops):
                call(new)
            new_best = min(new_best, (time.perf_counter() - start) / loops)
        print(
            f"| {name} | {old_best * 1e6:.1f} | {new_best * 1e6:.1f} | "
            f"{old_best / new_best:.3f} |"
        )


if __name__ == "__main__":
    main()
