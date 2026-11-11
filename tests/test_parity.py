from __future__ import annotations

import random

import numpy as np
import pytest


def test_ffi_buffer_contract():
    from editdistance._lib import address

    array = np.arange(8, dtype=np.int64)
    assert address(array, np.int64).value == array.ctypes.data
    with pytest.raises(TypeError):
        address(array, np.uint64)
    with pytest.raises(ValueError):
        address(array[::2], np.int64)
    with pytest.raises(ValueError):
        address(array.reshape(2, 4), np.int64)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("Saturday", "Sunday", 3),
        ("flaw", "lawn", 2),
        ("gumbo", "gambol", 2),
        ("same", "same", 0),
    ],
)
def test_published_and_standard_vectors(implementations, a, b, expected):
    ours, upstream = implementations
    assert ours.eval(a, b) == expected
    assert ours.eval(a, b) == upstream.eval(a, b)
    assert upstream.eval(a, b) == expected


def test_unicode_codepoints_match_upstream(implementations):
    ours, upstream = implementations
    cases = [
        ("café", "cafe"),
        ("\U00010400Mojo", "\U00010400mojo"),
        ("e\u0301", "é"),
        ("\ud800x", "\ud800y"),
        ("東京", "京都"),
    ]
    for a, b in cases:
        assert ours.eval(a, b) == upstream.eval(a, b)


def test_bytes_and_bytearray_match_upstream(implementations):
    ours, upstream = implementations
    cases = [
        (b"\x00\x01\xff", b"\x00\x02\xff"),
        (bytearray(range(64)), bytearray(range(1, 65))),
        (b"abc", bytearray(b"axc")),
    ]
    for a, b in cases:
        assert ours.eval(a, b) == upstream.eval(a, b)


def test_arbitrary_hashable_sequences_match_upstream(implementations):
    ours, upstream = implementations
    cases = [
        ([1, 2, 3], [1, 4, 3]),
        (("alpha", "beta"), ("alpha", "gamma", "beta")),
        ([1, True, None], [True, 1, None]),
        (np.array([1, 2, 3]), np.array([1, 3, 3])),
    ]
    for a, b in cases:
        assert ours.eval(a, b) == upstream.eval(a, b)


def test_hash_collisions_follow_upstream_token_semantics(implementations):
    ours, upstream = implementations

    class Collision:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return 7

        def __eq__(self, other):
            return isinstance(other, Collision) and self.value == other.value

    a = [Collision("a")]
    b = [Collision("b")]
    assert upstream.eval(a, b) == 0
    assert ours.eval(a, b) == upstream.eval(a, b)


def test_random_string_parity_across_word_boundaries(implementations):
    ours, upstream = implementations
    rng = random.Random(42)
    for n in (1, 2, 31, 63, 64, 65, 127, 128, 129, 639, 640, 641, 1024):
        a = "".join(rng.choice("abcde") for _ in range(n))
        b = "".join(rng.choice("abcde") for _ in range(n + 7))
        assert ours.eval(a, b) == upstream.eval(a, b)
        assert ours.eval(b, a) == upstream.eval(b, a)


def test_short_simd_vector_and_tail_parity(implementations):
    ours, upstream = implementations
    # 7, 8 and 9 straddle one eight-lane pattern chunk in the UTF-32 kernel;
    # 63, 64 and 65 straddle a 64-bit mask word in the bit kernel.
    for n in (1, 3, 4, 5, 7, 8, 9, 31, 63, 64, 65):
        a = ("abcd" * 16 + "abcd")[:n]
        b = ("abxd" * 17)[: n + 1]
        assert ours.eval(a, b) == upstream.eval(a, b)
        assert ours.eval(b, a) == upstream.eval(b, a)
        assert ours.eval(a.encode(), b.encode()) == upstream.eval(
            a.encode(), b.encode()
        )
        unicode_a = ("αβγδ" * 16 + "αβγδ")[:n]
        unicode_b = ("αβζδ" * 17)[: n + 1]
        assert ours.eval(unicode_a, unicode_b) == upstream.eval(
            unicode_a, unicode_b
        )
        assert ours.eval(unicode_b, unicode_a) == upstream.eval(
            unicode_b, unicode_a
        )


def test_simd_paths_cover_width_boundaries(implementations):
    ours, upstream = implementations
    # The UTF-32 word kernel compares eight pattern elements per chunk, so the
    # odd pattern lengths 7, 9, 63 and 65 all leave a scalar tail, and 63..65
    # also crosses a whole-word boundary in the pattern.
    for n in (0, 1, 7, 8, 9, 63, 64, 65, 129):
        for m in (0, 1, 7, 8, 9, 65, 100_003):
            a = "".join("αβγδεƒg"[i % 7] for i in range(n))
            b = "".join("αβγδεƒh"[i % 7] for i in range(m))
            assert ours.eval(a, b) == upstream.eval(a, b)
    for n in (0, 1, 7, 8, 9, 63, 64, 65, 129):
        for m in (0, 1, 8, 64, 65, 100_003):
            a = ("abcdefgh" * 17)[:n]
            b = ("abcdefgi" * 12500)[:m]
            assert ours.eval(a, b) == upstream.eval(a, b)
            assert ours.eval(a.encode(), b.encode()) == upstream.eval(
                a.encode(), b.encode()
            )


def test_long_odd_lengths_exercise_tails(implementations):
    ours, upstream = implementations
    for n, m in ((8, 100_007), (65, 100_007), (1_001, 1_003), (7, 999)):
        a = "".join("abcdefgh"[i % 8] for i in range(n))
        b = "".join("abcdefgi"[i % 8] for i in range(m))
        assert ours.eval(a, b) == upstream.eval(a, b)
        unicode_a = "".join("αβγδεƒgh"[i % 8] for i in range(n))
        unicode_b = "".join("αβγδεƒgi"[i % 8] for i in range(m))
        assert ours.eval(unicode_a, unicode_b) == upstream.eval(
            unicode_a, unicode_b
        )


def test_short_utf32_patterns_match_at_every_alignment(implementations):
    """The UTF-32 kernel picks its width from the buffer's real alignment."""
    ours, upstream = implementations
    for n in (0, 1, 7, 8, 9, 15, 16, 17, 63, 64):
        for pad in range(0, 3):
            a = "α" * pad + "".join("αβδε"[i % 4] for i in range(n))
            b = "".join("αβζε"[i % 4] for i in range(n + 3))
            assert ours.eval(a, b) == upstream.eval(a, b)
            assert ours.eval(b, a) == upstream.eval(b, a)


def test_bit_kernel_reuses_dirty_scratch_buffers(implementations):
    """The fused mask-table clear must not depend on a fresh allocation."""
    ours, upstream = implementations
    from editdistance._lib import address, lib

    kernels = lib()
    rng = np.random.default_rng(3)
    for n, m in ((1, 5), (8, 9), (63, 64), (64, 65), (65, 130), (129, 1001)):
        pattern = rng.integers(0, 50, n).astype(np.int64)
        text = rng.integers(0, 50, m).astype(np.int64)
        words = (n + 63) // 64
        capacity = 8
        while capacity < n * 2:
            capacity *= 2
        keys = np.full(capacity, -1, dtype=np.int64)
        used = np.full(capacity, 255, dtype=np.uint8)
        masks = np.full(capacity * words, 2**63, dtype=np.uint64)
        vectors = [np.full(words, 7, dtype=np.uint64) for _ in range(5)]
        result = kernels.med_distance_bit(
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
        assert result == upstream.eval(pattern.tolist(), text.tolist())


def test_short_pattern_long_text_uses_word_path(implementations):
    ours, upstream = implementations
    a = "abcd" * 16
    b = "z" * 257
    assert ours.eval(a, b) == upstream.eval(a, b)
    assert ours.eval(a.encode(), b.encode()) == upstream.eval(
        a.encode(), b.encode()
    )


def test_word_path_handles_large_text_and_simd_tail(implementations):
    ours, upstream = implementations
    a = "abcde" * 7
    b = ("abcdf" * 20_001)[:100_003]
    assert ours.eval(a, b) == upstream.eval(a, b)
    assert ours.eval(a.encode(), b.encode()) == upstream.eval(
        a.encode(), b.encode()
    )


def test_dp_fallback_matches_upstream(implementations):
    ours, upstream = implementations
    a = "ab" * 550
    b = "ba" * 550
    assert ours.eval(a, b) == upstream.eval(a, b) == 2 == 2


def test_criterion_matches_upstream_for_all_thresholds(implementations):
    ours, upstream = implementations
    cases = [
        ("", ""),
        ("", "abc"),
        ("abc", "abc"),
        ("abc", "abd"),
        ("kitten", "sitting"),
        ("a" * 200 + "x", "a" * 200 + "y"),
    ]
    for a, b in cases:
        for threshold in range(8):
            assert ours.eval_criterion(a, b, threshold) == upstream.eval_criterion(
                a, b, threshold
            )


def test_criterion_random_parity(implementations):
    ours, upstream = implementations
    rng = random.Random(7)
    for _ in range(100):
        n = rng.randrange(0, 80)
        m = rng.randrange(0, 80)
        a = "".join(rng.choice("abcd") for _ in range(n))
        b = "".join(rng.choice("abcd") for _ in range(m))
        threshold = rng.randrange(0, 12)
        assert ours.eval_criterion(a, b, threshold) == upstream.eval_criterion(
            a, b, threshold
        )


def test_aliases_match_upstream(implementations):
    ours, upstream = implementations
    assert ours.distance("banana", "bahama") == upstream.distance(
        "banana", "bahama"
    )
    assert ours.distance(a="abc", b="axc") == upstream.distance(a="abc", b="axc")
    assert ours.distance_le_than("abc", "axc", 1) == upstream.distance_le_than(
        "abc", "axc", 1
    )
    assert ours.distance_le_than(a="abc", b="axc", thr=1) == (
        upstream.distance_le_than(a="abc", b="axc", thr=1)
    )


@pytest.mark.parametrize("threshold", [-1, 2**32, "3", None])
def test_criterion_threshold_errors_match_upstream(
    implementations, threshold
):
    ours, upstream = implementations
    with pytest.raises(type(_raised_by(upstream, threshold))):
        ours.eval_criterion("a", "b", threshold)


def _raised_by(upstream, threshold):
    try:
        upstream.eval_criterion("a", "b", threshold)
    except Exception as error:
        return error
    raise AssertionError("upstream unexpectedly accepted invalid threshold")


def test_unhashable_elements_raise_like_upstream(implementations):
    ours, upstream = implementations
    with pytest.raises(TypeError):
        upstream.eval([[]], [[]])
    with pytest.raises(TypeError):
        ours.eval([[]], [[]])


def test_symmetry_and_distance_bounds(implementations):
    ours, _ = implementations
    rng = random.Random(99)
    for _ in range(50):
        a = "".join(rng.choice("abcdef") for _ in range(rng.randrange(100)))
        b = "".join(rng.choice("abcdef") for _ in range(rng.randrange(100)))
        result = ours.eval(a, b)
        assert result == ours.eval(b, a)
        assert abs(len(a) - len(b)) <= result <= max(len(a), len(b))
