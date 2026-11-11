"""Randomised differential check against editdistance 0.8.1.

Complements `tests/test_parity.py` for UNROLL work: it hits every kernel path
(short u8/u32 word kernels, the bit kernel on both sides of the 64- and
128-bit word boundaries, and the DP fallback above 1,024 tokens) on inputs the
unit tests do not enumerate.

Run with ``pixi run python bench/differential.py [rounds]``.
"""

from __future__ import annotations

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")
sys.path.insert(0, PYTHON_DIR)

import editdistance as ours  # noqa: E402

saved_module = sys.modules.pop("editdistance")
saved_path = sys.path[:]
sys.path = [
    entry
    for entry in sys.path
    if os.path.abspath(entry or os.curdir) != os.path.abspath(PYTHON_DIR)
]
try:
    import editdistance as upstream  # noqa: E402
finally:
    sys.path = saved_path
    sys.modules["editdistance"] = saved_module


def mutate(text, every, rng):
    chars = list(text)
    for index in range(0, len(chars), every):
        chars[index] = rng.choice("xyz")
    return "".join(chars)


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = random.Random(20261111)
    checked = 0
    for _ in range(rounds):
        size = rng.choice(
            [1, 2, 3, 15, 31, 62, 63, 64, 65, 66, 100, 127, 128, 129, 200, 640,
             1000, 1023, 1024, 1025, 1100]
        )
        other = max(0, size + rng.choice([-2, -1, 0, 0, 1, 7, 64, 65]))
        kind = rng.choice(["ascii", "unicode", "ints", "bytes"])
        if kind == "ascii":
            a = "".join(rng.choice("abcdefg") for _ in range(size))
            b = mutate(a, rng.randint(2, 20), rng) if rng.random() < 0.7 else "".join(
                rng.choice("abcdefg") for _ in range(other)
            )
        elif kind == "unicode":
            alphabet = "αβγδε日本語\U00010400"
            a = "".join(rng.choice(alphabet) for _ in range(size))
            b = mutate(a, rng.randint(2, 20), rng) if rng.random() < 0.7 else "".join(
                rng.choice(alphabet) for _ in range(other)
            )
        elif kind == "ints":
            a = [rng.randrange(50) for _ in range(size)]
            b = [rng.randrange(50) for _ in range(other)]
        else:
            a = bytes(rng.randrange(256) for _ in range(size))
            b = bytes(rng.randrange(256) for _ in range(other))

        checked += 1
        got = ours.eval(a, b)
        want = upstream.eval(a, b)
        if got != want:
            raise SystemExit(f"eval mismatch {kind} {size} {other}: {got} != {want}")

        expected = want
        for threshold in (1, 2, 3, 5, 17):
            checked += 1
            want_le = bool(expected <= threshold)
            if kind in ("ascii", "unicode") and min(len(a), len(b)) == 0:
                want_le = bool(expected <= threshold)
            got_le = ours.eval_criterion(a, b, threshold)
            if got_le != want_le:
                raise SystemExit(
                    f"criterion mismatch {kind} {size} {other} thr {threshold}: "
                    f"{got_le} != {want_le}"
                )
    print(f"differential: {checked} comparisons matched editdistance 0.8.1")


if __name__ == "__main__":
    main()
