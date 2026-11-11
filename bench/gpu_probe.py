"""GPU smoke test and host<->device cost probe; run with ``pixi run gpu-probe``.

This is not on any library path. It exists so the GPU gate in the README can be
re-measured on the next machine instead of trusted.

Requirements (verified on mojo ==1.2.0.dev2026092605 + max ==26.7.0.dev2026092605):

- The ``max`` package must be installed. The repository's pixi environment does
  not carry it, so this is meant to run from a scratch environment like::

      pixi init /tmp/gpu && cd /tmp/gpu
      pixi add 'mojo ==1.2.0.dev2026092605' 'max ==26.7.0.dev2026092605'
      pixi run python /path/to/bench/gpu_probe.py

- These API shapes are what the compiler accepts, and differ from the shorthand
  in MOJO_NOTES.md section 5 (``enqueue_create_buffer[Int64]`` is rejected):

      ctx.enqueue_create_buffer[DType.int64](n)   # dtype is a DType, not a type
      ctx.enqueue_copy(dst, src)                  # either direction
      ctx.enqueue_function[kernel](*args, grid_dim=..., block_dim=...)
      block_idx.x / thread_idx.x / block_dim.x    # per-axis .x; Int(...) is host-only

- Kernel arguments must be fixed-width (``Int64``, not ``Int``): ``Int`` does not
  conform to ``DevicePassable``. ``UInt64``/``UInt32`` are not passable either, so
  a ``uint64`` word buffer has no direct GPU representation on this toolchain yet.

Exits 2 with a clear message when no GPU is present, so it can be used in a gate.
Tests that need the GPU (`tests/test_gpu.py`) skip unless ``EDITDISTANCE_GPU_PROBE``
is set to a working checkout of this script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

MODEL = textwrap.dedent(
    """
    from max.gpu import block_idx, thread_idx, block_dim
    from max.gpu.host import DeviceContext
    from std.memory import alloc
    from std.time import perf_counter_ns

    # Passable args must be fixed-width; Pointer passes as a device buffer.
    def _inc(d: Pointer[Int64, AnyOrigin[mut=True]], n: Int64):
        var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        if g < Int(n):
            d.unsafe_store(g, d.unsafe_load(g) + 1)

    def main() raises:
        var ctx = DeviceContext()
        print("device:", ctx.name())

        # --- correctness: one launch, grid covers a size with a tail ---
        var n = 1000003
        var grid = (n + 255) // 256
        var db = ctx.enqueue_create_buffer[DType.int64](n)
        var host = alloc[Int64](n)
        for i in range(n):
            host.unsafe_store(i, Int64(i))
        ctx.enqueue_copy(db, host)
        ctx.enqueue_function[_inc](db, Int64(3 * n), grid_dim=grid, block_dim=256)
        ctx.synchronize()
        ctx.enqueue_copy(host, db)
        ctx.synchronize()
        # tail: the last element is processed once, and nothing past n-1 is written
        print("tail ok:", Int(host.unsafe_load(0)) == 1 and Int(host.unsafe_load(n - 1)) == 1000003)

        # --- launch round trip ---
        var big = ctx.enqueue_create_buffer[DType.int64](2097152)
        var best_launch = 1 << 62
        for _ in range(20):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[_inc](big, Int64(2097152), grid_dim=8192, block_dim=256)
            ctx.synchronize()
            var dt = perf_counter_ns() - t0
            if dt < best_launch:
                best_launch = dt
        print("launch+sync best us:", Float64(best_launch) / 1e3)

        # --- host<->device round trip by size ---
        for exp in range(10, 25, 2):
            var nbytes = 1 << exp
            var count = nbytes // 8
            var buf = ctx.enqueue_create_buffer[DType.int64](count)
            var h = alloc[Int64](count)
            for i in range(count):
                h.unsafe_store(i, Int64(0))
            var best = 1 << 62
            for _ in range(20):
                var t0 = perf_counter_ns()
                ctx.enqueue_copy(buf, h)
                ctx.enqueue_copy(h, buf)
                ctx.synchronize()
                var dt = perf_counter_ns() - t0
                if dt < best:
                    best = dt
            print("round trip bytes/direction:", nbytes, "best us:", Float64(best) / 1e3)
    """
)


def _environment(mojo: str) -> dict[str, str]:
    # MOJO_NOTES.md section 0: MODULAR_HOME must point at the env's share/max or
    # every import fails with "unable to locate module 'std'". It must belong to
    # the same environment as the `mojo` we run, not the caller's inherited one.
    env = dict(os.environ)
    resolved = shutil.which(mojo) if os.sep not in mojo else mojo
    if resolved:
        env["MODULAR_HOME"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(resolved))), "share", "max"
        )
    return env


def main() -> None:
    mojo = os.environ.get("MOJO", "mojo")
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "probe.mojo")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(MODEL)
        try:
            result = subprocess.run(
                [mojo, "run", source],
                capture_output=True,
                text=True,
                timeout=600,
                env=_environment(mojo),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"GPU probe unavailable: {error}", file=sys.stderr)
            raise SystemExit(2)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        print(
            "GPU probe failed (no device, or max not installed); "
            "see bench/gpu_probe.py docstring",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
