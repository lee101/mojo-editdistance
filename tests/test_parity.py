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
    for n in (1, 3, 4, 5, 31, 63, 64):
        a = ("abcd" * 16)[:n]
        b = ("abxd" * 17)[: n + 1]
        assert ours.eval(a, b) == upstream.eval(a, b)
        assert ours.eval(a.encode(), b.encode()) == upstream.eval(
            a.encode(), b.encode()
        )
        unicode_a = ("αβγδ" * 16)[:n]
        unicode_b = ("αβζδ" * 17)[: n + 1]
        assert ours.eval(unicode_a, unicode_b) == upstream.eval(
            unicode_a, unicode_b
        )


def test_short_pattern_long_text_uses_multiword_path(implementations):
    ours, upstream = implementations
    a = "abcd" * 16
    b = "z" * 257
    assert ours.eval(a, b) == upstream.eval(a, b)


def test_dp_fallback_matches_upstream(implementations):
    ours, upstream = implementations
    a = "ab" * 550
    b = "ba" * 550
    assert ours.eval(a, b) == upstream.eval(a, b) == 2


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
