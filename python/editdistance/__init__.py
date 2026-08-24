"""Levenshtein distance with Mojo kernels and editdistance-compatible names."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from ._lib import address, lib

__version__ = "0.1.0"
__all__ = ("eval", "distance", "eval_criterion", "distance_le_than")

_BIT_PATTERN_LIMIT = 1024


def _tokens(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    na = len(a)
    nb = len(b)

    if type(a) is str and type(b) is str:
        aa = np.frombuffer(
            a.encode("utf-32-le", errors="surrogatepass"), dtype="<u4", count=na
        ).astype(np.int64)
        bb = np.frombuffer(
            b.encode("utf-32-le", errors="surrogatepass"), dtype="<u4", count=nb
        ).astype(np.int64)
        return aa, bb

    bytes_types = (bytes, bytearray)
    if type(a) in bytes_types and type(b) in bytes_types:
        aa = np.frombuffer(a, dtype=np.uint8, count=na).astype(np.int64)
        bb = np.frombuffer(b, dtype=np.uint8, count=nb).astype(np.int64)
        return aa, bb

    aa = np.fromiter((hash(a[i]) for i in range(na)), dtype=np.int64, count=na)
    bb = np.fromiter((hash(b[i]) for i in range(nb)), dtype=np.int64, count=nb)
    return aa, bb


def _ordered(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (a, b) if len(a) <= len(b) else (b, a)


def _distance_tokens(a: np.ndarray, b: np.ndarray) -> int:
    pattern, text = _ordered(a, b)
    n = len(pattern)
    m = len(text)
    if n == 0:
        return m

    kernels = lib()
    if n <= _BIT_PATTERN_LIMIT:
        words = (n + 63) // 64
        capacity = 8
        while capacity < n * 2:
            capacity *= 2
        keys = np.empty(capacity, dtype=np.int64)
        used = np.empty(capacity, dtype=np.uint8)
        masks = np.empty(capacity * words, dtype=np.uint64)
        vectors = [np.empty(words, dtype=np.uint64) for _ in range(5)]
        return int(
            kernels.med_distance_bit(
                address(pattern, np.int64),
                n,
                address(text, np.int64),
                m,
                address(keys, np.int64),
                address(used, np.uint8),
                address(masks, np.uint64),
                capacity,
                *(address(vector, np.uint64) for vector in vectors),
            )
        )

    row = np.empty(n + 1, dtype=np.int64)
    return int(
        kernels.med_distance_dp(
            address(text, np.int64),
            m,
            address(pattern, np.int64),
            n,
            address(row, np.int64),
        )
    )


def eval(a: Any, b: Any) -> int:
    """Return the Levenshtein distance between two indexable sequences."""
    if (
        type(a) is str
        and type(b) is str
        and min(len(a), len(b)) <= 64
    ):
        if a.isascii() and b.isascii():
            return int(
                lib().med_distance_u8_word(
                    a.encode(), len(a), b.encode(), len(b)
                )
            )
        aa = a.encode("utf-32-le", errors="surrogatepass")
        bb = b.encode("utf-32-le", errors="surrogatepass")
        return int(lib().med_distance_u32_word(aa, len(a), bb, len(b)))

    if (
        type(a) is bytes
        and type(b) is bytes
        and min(len(a), len(b)) <= 64
    ):
        return int(lib().med_distance_u8_word(a, len(a), b, len(b)))

    aa, bb = _tokens(a, b)
    return _distance_tokens(aa, bb)


def distance(*args, **kwargs):
    """An alias to eval."""
    return eval(*args, **kwargs)


def _as_threshold(value: Any) -> int:
    try:
        threshold = operator.index(value)
    except TypeError:
        if isinstance(value, float):
            threshold = int(value)
        else:
            raise TypeError("an integer is required") from None
    if threshold < 0:
        raise OverflowError("can't convert negative value to unsigned int")
    if threshold > 0xFFFFFFFF:
        raise OverflowError("value too large to convert to unsigned int")
    return threshold


def eval_criterion(a: Any, b: Any, thr: int) -> bool:
    """Return whether the Levenshtein distance is at most ``thr``."""
    threshold = _as_threshold(thr)
    aa, bb = _tokens(a, b)
    n = len(aa)
    m = len(bb)
    if n == 0 or m == 0:
        return max(n, m) <= threshold
    if threshold == 0:
        return False
    if abs(n - m) > threshold:
        return False
    if threshold >= max(n, m):
        return True

    pattern, text = _ordered(aa, bb)
    row = np.empty(len(pattern) + 1, dtype=np.int64)
    return bool(
        lib().med_within(
            address(text, np.int64),
            len(text),
            address(pattern, np.int64),
            len(pattern),
            threshold,
            address(row, np.int64),
        )
    )


def distance_le_than(*args, **kwargs):
    """An alias to eval_criterion."""
    return eval_criterion(*args, **kwargs)
