"""Levenshtein kernels and their C ABI.

The caller owns every buffer. Sequence elements are signed 64-bit tokens;
equal tokens represent equal elements.
"""

from std.sys import simd_width_of as simdwidthof
from std.memory import stack_allocation

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]


def _distance_word[dtype: DType](
    pattern: Pointer[Scalar[dtype], AnyOrigin[mut=True]],
    pattern_len: Int,
    text: Pointer[Scalar[dtype], AnyOrigin[mut=True]],
    text_len: Int,
) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var lane_bits = SIMD[DType.uint64, W]()
    for lane in range(W):
        lane_bits[lane] = UInt64(1) << UInt64(lane)
    var zeros = SIMD[DType.uint64, W]()
    var shift = SIMD[DType.uint64, W](UInt64(W))

    var vp = ~UInt64(0)
    var vn = UInt64(0)
    var top = UInt64(1) << UInt64(pattern_len - 1)
    var score = pattern_len
    for i in range(text_len):
        var x = UInt64(0)
        var weights = lane_bits
        var j = 0
        var target = SIMD[dtype, W](text.unsafe_load(i))
        while j + W <= pattern_len:
            var matches = pattern.unsafe_load[width=W](j).eq(target)
            x += matches.select(weights, zeros).reduce_add()
            weights = weights << shift
            j += W
        while j < pattern_len:
            if pattern.unsafe_load(j) == text.unsafe_load(i):
                x |= UInt64(1) << UInt64(j)
            j += 1

        var d0 = (((x & vp) + vp) ^ vp) | x | vn
        var hp = vn | ~(d0 | vp)
        var hn = d0 & vp
        var shifted_hp = (hp << 1) | UInt64(1)
        vp = (hn << 1) | ~(d0 | shifted_hp)
        vn = d0 & shifted_hp
        if (hp & top) != 0:
            score += 1
        elif (hn & top) != 0:
            score -= 1
    return score


def _distance_word_u8(
    pattern: U8Ptr, pattern_len: Int, text: U8Ptr, text_len: Int
) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var masks = stack_allocation[256, UInt64]()
    var zeros = SIMD[DType.uint64, W]()
    for i in range(0, 256, W):
        masks.unsafe_store(i, zeros)
    # A word kernel is only entered for pattern_len <= 64, so bit `i` fits.
    for i in range(pattern_len):
        var byte = Int(pattern.unsafe_load(i))
        masks.unsafe_store(byte, masks.unsafe_load(byte) | (UInt64(1) << UInt64(i)))

    var vp = ~UInt64(0)
    var vn = UInt64(0)
    var top = UInt64(1) << UInt64(pattern_len - 1)
    var score = pattern_len
    for i in range(text_len):
        var x = masks.unsafe_load(Int(text.unsafe_load(i)))
        var d0 = (((x & vp) + vp) ^ vp) | x | vn
        var hp = vn | ~(d0 | vp)
        var hn = d0 & vp
        var shifted_hp = (hp << 1) | UInt64(1)
        vp = (hn << 1) | ~(d0 | shifted_hp)
        vn = d0 & shifted_hp
        if (hp & top) != 0:
            score += 1
        elif (hn & top) != 0:
            score -= 1
    return score


def _slot(
    keys: I64Ptr, used: U8Ptr, capacity: Int, token: Int64, insert: Bool
) -> Int:
    var value = Int(token)
    var index = (value ^ (value >> 33) ^ (value >> 17)) & (capacity - 1)
    while used.unsafe_load(index) != 0:
        if keys.unsafe_load(index) == token:
            return index
        index = (index + 1) & (capacity - 1)
    if insert:
        used.unsafe_store(index, 1)
        keys.unsafe_store(index, token)
        return index
    return -1


def _distance_bit(
    pattern: I64Ptr,
    pattern_len: Int,
    text: I64Ptr,
    text_len: Int,
    keys: I64Ptr,
    used: U8Ptr,
    masks: U64Ptr,
    capacity: Int,
    vp: U64Ptr,
    vn: U64Ptr,
    d0: U64Ptr,
    hp: U64Ptr,
    hn: U64Ptr,
) -> Int:
    var words = (pattern_len + 63) >> 6
    for i in range(capacity):
        used.unsafe_store(i, 0)
    for i in range(capacity * words):
        masks.unsafe_store(i, 0)

    for i in range(pattern_len):
        var index = _slot(keys, used, capacity, pattern.unsafe_load(i), True)
        var word = i >> 6
        var bit = i & 63
        var mask_index = index * words + word
        var mask_value = masks.unsafe_load(mask_index) | (UInt64(1) << UInt64(bit))
        masks.unsafe_store(mask_index, mask_value)

    for word in range(words):
        vp.unsafe_store(word, ~UInt64(0))
        vn.unsafe_store(word, 0)

    var last = words - 1
    var last_bits = pattern_len - last * 64
    if last_bits < 64:
        vp.unsafe_store(last, (UInt64(1) << UInt64(last_bits)) - UInt64(1))
    var top = UInt64(1) << UInt64(last_bits - 1)
    var high_bit = UInt64(1) << 63
    var score = pattern_len

    for i in range(text_len):
        var index = _slot(keys, used, capacity, text.unsafe_load(i), False)
        for word in range(words):
            var x = UInt64(0)
            if index >= 0:
                x = masks.unsafe_load(index * words + word)
            if word > 0 and (hn.unsafe_load(word - 1) & high_bit) != 0:
                x |= UInt64(1)

            var vp_value = vp.unsafe_load(word)
            var vn_value = vn.unsafe_load(word)
            var d0_value = (((x & vp_value) + vp_value) ^ vp_value) | x | vn_value
            d0.unsafe_store(word, d0_value)
            hp.unsafe_store(word, vn_value | ~(d0_value | vp_value))
            hn.unsafe_store(word, d0_value & vp_value)

            var shifted_hp = hp.unsafe_load(word) << 1
            if word == 0 or (hp.unsafe_load(word - 1) & high_bit) != 0:
                shifted_hp |= UInt64(1)
            vp.unsafe_store(word, (hn.unsafe_load(word) << 1) | ~(d0_value | shifted_hp))
            if word > 0 and (hn.unsafe_load(word - 1) & high_bit) != 0:
                vp.unsafe_store(word, vp.unsafe_load(word) | UInt64(1))
            vn.unsafe_store(word, d0_value & shifted_hp)

        if (hp.unsafe_load(last) & top) != 0:
            score += 1
        elif (hn.unsafe_load(last) & top) != 0:
            score -= 1
    return score


def _distance_dp(a: I64Ptr, n: Int, b: I64Ptr, m: Int, row: I64Ptr) -> Int:
    var a_start = 0
    var b_start = 0
    var an = n
    var bm = m
    while an > 0 and bm > 0 and a[a_start] == b[b_start]:
        a_start += 1
        b_start += 1
        an -= 1
        bm -= 1
    while an > 0 and bm > 0 and a[a_start + an - 1] == b[b_start + bm - 1]:
        an -= 1
        bm -= 1
    if an == 0:
        return bm
    if bm == 0:
        return an

    for j in range(bm + 1):
        row.unsafe_store(j, Int64(j))
    for i in range(1, an + 1):
        var diagonal = Int(row.unsafe_load(0))
        row.unsafe_store(0, Int64(i))
        for j in range(1, bm + 1):
            var above = Int(row.unsafe_load(j))
            var deletion = above + 1
            var insertion = Int(row.unsafe_load(j - 1)) + 1
            var substitution = diagonal
            if a[a_start + i - 1] != b[b_start + j - 1]:
                substitution += 1
            var best = deletion if deletion < insertion else insertion
            if substitution < best:
                best = substitution
            row.unsafe_store(j, Int64(best))
            diagonal = above
    return Int(row.unsafe_load(bm))


def _within(
    a: I64Ptr, n: Int, b: I64Ptr, m: Int, threshold: Int, row: I64Ptr
) -> Int:
    var a_start = 0
    var b_start = 0
    var an = n
    var bm = m
    while an > 0 and bm > 0 and a[a_start] == b[b_start]:
        a_start += 1
        b_start += 1
        an -= 1
        bm -= 1
    while an > 0 and bm > 0 and a[a_start + an - 1] == b[b_start + bm - 1]:
        an -= 1
        bm -= 1
    if an == 0:
        return 1 if bm <= threshold else 0
    if bm == 0:
        return 1 if an <= threshold else 0
    if abs(an - bm) > threshold:
        return 0

    var infinity = threshold + 1
    for j in range(bm + 1):
        row.unsafe_store(j, Int64(j if j <= threshold else infinity))

    for i in range(1, an + 1):
        var start = 1 if i <= threshold + 1 else i - threshold
        var finish = bm if bm < i + threshold else i + threshold
        # `a[a_start + i - 1]` is fixed for the whole band scan.
        var deletion_token = a[a_start + i - 1]
        var diagonal = Int(row.unsafe_load(start - 1))
        if start == 1:
            row.unsafe_store(0, Int64(i if i <= threshold else infinity))
        else:
            row.unsafe_store(start - 1, Int64(infinity))
        var row_min = infinity
        for j in range(start, finish + 1):
            var above = Int(row.unsafe_load(j))
            var deletion = above + 1
            var insertion = Int(row.unsafe_load(j - 1)) + 1
            var substitution = diagonal
            if deletion_token != b[b_start + j - 1]:
                substitution += 1
            var best = deletion if deletion < insertion else insertion
            if substitution < best:
                best = substitution
            # A cell is exact unless its true distance exceeds `threshold`.
            # A capped input has true distance above the threshold too, so any
            # cell read from it is as well. Capping such a cell to `infinity`
            # and leaving it unclamped therefore agree on `<= threshold`, and
            # unclamped cells are never *below* `infinity`, so a stale value
            # can never look like a small distance.
            row.unsafe_store(j, Int64(best))
            if best < row_min:
                row_min = best
            diagonal = above
        if finish < bm:
            row.unsafe_store(finish + 1, Int64(infinity))
        if row_min > threshold:
            return 0
    return 1 if Int(row.unsafe_load(bm)) <= threshold else 0


@export("med_distance_u8_word")
def med_distance_u8_word(
    a_addr: Int, n: Int, b_addr: Int, m: Int
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    if n <= m:
        return _distance_word_u8(
            U8Ptr(unsafe_from_address=a_addr),
            n,
            U8Ptr(unsafe_from_address=b_addr),
            m,
        )
    return _distance_word_u8(
        U8Ptr(unsafe_from_address=b_addr),
        m,
        U8Ptr(unsafe_from_address=a_addr),
        n,
    )


@export("med_distance_u32_word")
def med_distance_u32_word(
    a_addr: Int, n: Int, b_addr: Int, m: Int
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    if n <= m:
        return _distance_word(
            U32Ptr(unsafe_from_address=a_addr),
            n,
            U32Ptr(unsafe_from_address=b_addr),
            m,
        )
    return _distance_word(
        U32Ptr(unsafe_from_address=b_addr),
        m,
        U32Ptr(unsafe_from_address=a_addr),
        n,
    )


@export("med_distance_bit")
def med_distance_bit(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    keys_addr: Int,
    used_addr: Int,
    masks_addr: Int,
    capacity: Int,
    vp_addr: Int,
    vn_addr: Int,
    d0_addr: Int,
    hp_addr: Int,
    hn_addr: Int,
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    return _distance_bit(
        I64Ptr(unsafe_from_address=a_addr),
        n,
        I64Ptr(unsafe_from_address=b_addr),
        m,
        I64Ptr(unsafe_from_address=keys_addr),
        U8Ptr(unsafe_from_address=used_addr),
        U64Ptr(unsafe_from_address=masks_addr),
        capacity,
        U64Ptr(unsafe_from_address=vp_addr),
        U64Ptr(unsafe_from_address=vn_addr),
        U64Ptr(unsafe_from_address=d0_addr),
        U64Ptr(unsafe_from_address=hp_addr),
        U64Ptr(unsafe_from_address=hn_addr),
    )


@export("med_distance_dp")
def med_distance_dp(
    a_addr: Int, n: Int, b_addr: Int, m: Int, row_addr: Int
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    return _distance_dp(
        I64Ptr(unsafe_from_address=a_addr),
        n,
        I64Ptr(unsafe_from_address=b_addr),
        m,
        I64Ptr(unsafe_from_address=row_addr),
    )


@export("med_within")
def med_within(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    threshold: Int,
    row_addr: Int,
) abi("C") -> Int:
    if n == 0:
        return 1 if m <= threshold else 0
    if m == 0:
        return 1 if n <= threshold else 0
    return _within(
        I64Ptr(unsafe_from_address=a_addr),
        n,
        I64Ptr(unsafe_from_address=b_addr),
        m,
        threshold,
        I64Ptr(unsafe_from_address=row_addr),
    )
