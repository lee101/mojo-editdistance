"""Benchmark Mojo against editdistance 0.8.1 on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")
sys.path.insert(0, PYTHON_DIR)

import editdistance as mojo_editdistance  # noqa: E402

saved_path = sys.path[:]
saved_module = sys.modules.pop("editdistance")
sys.path = [
    entry
    for entry in sys.path
    if os.path.abspath(entry or os.curdir) != os.path.abspath(PYTHON_DIR)
]
try:
    import editdistance as upstream_editdistance  # noqa: E402
finally:
    sys.path = saved_path
    sys.modules["editdistance"] = saved_module


def best_time(function, loops: int, repeat: int = 5) -> float:
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(loops):
            function()
        best = min(best, (time.perf_counter() - start) / loops)
    return best


def mutate(text: str, every: int) -> str:
    chars = list(text)
    for index in range(every - 1, len(chars), every):
        chars[index] = "z" if chars[index] != "z" else "y"
    return "".join(chars)


def cases():
    rng = random.Random(123)
    ascii_64 = "".join(rng.choice("abcdefgh") for _ in range(64))
    ascii_1024 = "".join(rng.choice("abcdefgh") for _ in range(1024))
    text_100k = "".join(rng.choice("abcdefgh") for _ in range(100_000))
    alphabet = "αβγδε日本語" + "\U00010400"
    unicode_1k = "".join(rng.choice(alphabet) for _ in range(1000))
    changed_64 = mutate(ascii_64, 13)
    changed_1024 = mutate(ascii_1024, 29)
    changed_unicode = mutate(unicode_1k, 31)
    integers = [rng.randrange(100) for _ in range(1000)]
    criterion_a = "".join(rng.choice("abcd") for _ in range(2000))
    criterion_b = "z" * 12 + criterion_a[12:]
    return [
        (
            "eval ASCII 64 vs 64",
            lambda: mojo_editdistance.eval(ascii_64, changed_64),
            lambda: upstream_editdistance.eval(ascii_64, changed_64),
            2000,
        ),
        (
            "eval ASCII 1,024 vs 1,024",
            lambda: mojo_editdistance.eval(ascii_1024, changed_1024),
            lambda: upstream_editdistance.eval(ascii_1024, changed_1024),
            100,
        ),
        (
            "eval ASCII 64 vs 100,000",
            lambda: mojo_editdistance.eval(ascii_64, text_100k),
            lambda: upstream_editdistance.eval(ascii_64, text_100k),
            20,
        ),
        (
            "eval Unicode 1,000 vs 1,000",
            lambda: mojo_editdistance.eval(unicode_1k, changed_unicode),
            lambda: upstream_editdistance.eval(unicode_1k, changed_unicode),
            100,
        ),
        (
            "eval integer lists 1,000 vs 1,000",
            lambda: mojo_editdistance.eval(integers, integers[::-1]),
            lambda: upstream_editdistance.eval(integers, integers[::-1]),
            50,
        ),
        (
            "criterion 2,000 chars, threshold 2",
            lambda: mojo_editdistance.eval_criterion(
                criterion_a, criterion_b, 2
            ),
            lambda: upstream_editdistance.eval_criterion(
                criterion_a, criterion_b, 2
            ),
            10,
        ),
    ]


def machine() -> str:
    cpu = platform.processor()
    if not cpu or cpu in {"x86_64", "AMD64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                cpu = next(
                    line.split(":", 1)[1].strip()
                    for line in handle
                    if line.startswith("model name")
                )
        except (OSError, StopIteration):
            cpu = platform.machine()
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main() -> None:
    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-editdistance | editdistance 0.8.1 | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream, loops in cases():
        ours_result = ours()
        upstream_result = upstream()
        if ours_result != upstream_result:
            raise RuntimeError(
                f"benchmark parity failure for {name}: "
                f"{ours_result!r} != {upstream_result!r}"
            )
        mojo_seconds = best_time(ours, loops)
        upstream_seconds = best_time(upstream, loops)
        ratio = upstream_seconds / mojo_seconds
        label = "faster" if ratio >= 1 else "slower"
        relative = ratio if ratio >= 1 else 1 / ratio
        print(
            f"| {name} | {mojo_seconds * 1e6:.1f} us | "
            f"{upstream_seconds * 1e6:.1f} us | {relative:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
