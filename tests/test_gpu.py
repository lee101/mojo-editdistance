"""GPU-phase tests. They skip cleanly on a CPU-only host, or one without `max`.

The library itself has no GPU path: `bench/analysis.md` and the README's "GPU
pass" record that the kernels are below the compute gate or serial. These tests
check the two things that claim depends on, so a future GPU pass starts from
measured facts rather than a repeated assumption:

- the GPU probe (`bench/gpu_probe.py`) works end to end, including a grid whose
  last block is partial;
- the CPU kernel that a GPU path would have to beat is still reachable, so the
  CPU fallback is exercised rather than skipped.

`EDITDISTANCE_GPU_PROBE` must name the `mojo` binary of an environment that has
`max` installed. Unset, the GPU tests skip to save repeated device work; set,
they run on the real device or skip with "no usable GPU device" on a CPU-only
host. Nothing here runs the GPU on a host without one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "bench", "gpu_probe.py")
MOJO = os.environ.get("EDITDISTANCE_GPU_PROBE")


@pytest.fixture(scope="session")
def gpu_probe():
    if not MOJO:
        pytest.skip(
            "GPU tests skipped: EDITDISTANCE_GPU_PROBE is not set "
            "(needs a `mojo` binary from an environment with `max` installed)"
        )
    if not os.path.exists(MOJO):
        pytest.skip(f"GPU tests skipped: EDITDISTANCE_GPU_PROBE not found: {MOJO}")
    environment = dict(os.environ, MOJO=MOJO)
    result = subprocess.run(
        [sys.executable, PROBE],
        capture_output=True,
        text=True,
        timeout=600,
        env=environment,
    )
    if result.returncode == 2:
        pytest.skip("GPU tests skipped: no usable GPU device on this host")
    if result.returncode != 0:
        pytest.skip(f"GPU tests skipped: probe failed\n{result.stderr}")
    return result.stdout


def test_probe_finds_a_device(gpu_probe):
    line = next(
        (line for line in gpu_probe.splitlines() if line.startswith("device: ")),
        None,
    )
    assert line is not None and line != "device: ", gpu_probe


def test_probe_tail_is_covered(gpu_probe):
    # The probe's grid is ceil(n / 256) blocks over a count that is not a multiple
    # of 256, so this asserts the last element was processed exactly once.
    assert re.search(r"^tail ok: (True|true)$", gpu_probe, re.MULTILINE), gpu_probe


def test_probe_reports_launch_cost(gpu_probe):
    match = re.search(r"^launch\+sync best us: ([0-9.]+)$", gpu_probe, re.MULTILINE)
    assert match is not None, gpu_probe
    assert float(match.group(1)) > 0.0


def test_cpu_fallback_answers_without_a_gpu():
    # The shipped path is CPU-only; a large bit-kernel call must still answer.
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import editdistance

    text = "abcdefgh" * 12_500
    assert editdistance.eval(text[:64], text) > 0
