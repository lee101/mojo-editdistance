"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "editdistance.mojo")
LIBRARY = os.path.join(ROOT, "dist", "libmojo-editdistance.so")
BUILD_SCRIPT = os.path.join(ROOT, "build", "build.sh")

I = ctypes.c_ssize_t
P = ctypes.c_void_p
_SIGNATURES = {
    "med_distance_u8_word": ([P, I, P, I], I),
    "med_distance_u32_word": ([P, I, P, I], I),
    "med_distance_bit": (
        [P, I, P, I, P, P, P, I, P, P, P, P, P],
        I,
    ),
    "med_distance_dp": ([P, I, P, I, P], I),
    "med_within": ([P, I, P, I, I, P], I),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIBRARY)
        and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE)
    ):
        return LIBRARY
    proc = subprocess.run(
        ["bash", BUILD_SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIBRARY):
        output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part)
        raise RuntimeError(output[-4000:] or "Mojo build produced no shared library")
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def address(array: np.ndarray, dtype: np.dtype | type) -> ctypes.c_void_p:
    """Return a typed contiguous buffer pointer while the caller retains *array*."""
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise TypeError(f"expected dtype {expected}, got {array.dtype}")
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be one-dimensional and C-contiguous")
    pointer = int(array.ctypes.data)
    if array.size and pointer == 0:
        raise ValueError("non-empty FFI buffer has a null pointer")
    return ctypes.c_void_p(pointer)
