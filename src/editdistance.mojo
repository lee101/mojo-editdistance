"""Levenshtein kernels and their C ABI.

The caller owns every buffer. Sequence elements are signed 64-bit tokens;
equal tokens represent equal elements.
"""

from std.sys import simd_width_of as simdwidthof
from std.memory import stack_allocation

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def _distance_word[dtype: DType](
    pattern: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    pattern_len: Int,
    text: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
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
        var target = SIMD[dtype, W](text[i])
        while j + W <= pattern_len:
            var matches = pattern.load[width=W](j).eq(target)
            x += matches.select(weights, zeros).reduce_add()
            weights = weights << shift
            j += W
        while j < pattern_len:
            if pattern[j] == text[i]:
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
        masks.store(i, zeros)
    for i in range(pattern_len):
        masks[Int(pattern[i])] |= UInt64(1) << UInt64(i)

    var vp = ~UInt64(0)
    var vn = UInt64(0)
    var top = UInt64(1) << UInt64(pattern_len - 1)
    var score = pattern_len
    for i in range(text_len):
        var x = masks[Int(text[i])]
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
    while used[index] != 0:
        if keys[index] == token:
            return index
        index = (index + 1) & (capacity - 1)
    if insert:
        used[index] = 1
        keys[index] = token
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
        used[i] = 0
    for i in range(capacity * words):
        masks[i] = 0

    for i in range(pattern_len):
        var index = _slot(keys, used, capacity, pattern[i], True)
        var word = i >> 6
        var bit = i & 63
        masks[index * words + word] |= UInt64(1) << UInt64(bit)

    for word in range(words):
        vp[word] = ~UInt64(0)
        vn[word] = 0

    var last = words - 1
    var last_bits = pattern_len - last * 64
    if last_bits < 64:
        vp[last] = (UInt64(1) << UInt64(last_bits)) - UInt64(1)
    var top = UInt64(1) << UInt64(last_bits - 1)
    var high_bit = UInt64(1) << 63
    var score = pattern_len

    for i in range(text_len):
        var index = _slot(keys, used, capacity, text[i], False)
        for word in range(words):
            var x = UInt64(0)
            if index >= 0:
                x = masks[index * words + word]
            if word > 0 and (hn[word - 1] & high_bit) != 0:
                x |= UInt64(1)

            var vp_value = vp[word]
            var d0_value = (((x & vp_value) + vp_value) ^ vp_value) | x | vn[word]
            d0[word] = d0_value
            hp[word] = vn[word] | ~(d0_value | vp_value)
            hn[word] = d0_value & vp_value

            var shifted_hp = hp[word] << 1
            if word == 0 or (hp[word - 1] & high_bit) != 0:
                shifted_hp |= UInt64(1)
            vp[word] = (hn[word] << 1) | ~(d0_value | shifted_hp)
            if word > 0 and (hn[word - 1] & high_bit) != 0:
                vp[word] |= UInt64(1)
            vn[word] = d0_value & shifted_hp

        if (hp[last] & top) != 0:
            score += 1
        elif (hn[last] & top) != 0:
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
        row[j] = Int64(j)
    for i in range(1, an + 1):
        var diagonal = Int(row[0])
        row[0] = Int64(i)
        for j in range(1, bm + 1):
            var above = Int(row[j])
            var deletion = above + 1
            var insertion = Int(row[j - 1]) + 1
            var substitution = diagonal
            if a[a_start + i - 1] != b[b_start + j - 1]:
                substitution += 1
            var best = deletion if deletion < insertion else insertion
            if substitution < best:
                best = substitution
            row[j] = Int64(best)
            diagonal = above
    return Int(row[bm])


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
        row[j] = Int64(j if j <= threshold else infinity)

    for i in range(1, an + 1):
        var start = 1 if i <= threshold + 1 else i - threshold
        var finish = bm if bm < i + threshold else i + threshold
        var diagonal = Int(row[start - 1])
        if start == 1:
            row[0] = Int64(i if i <= threshold else infinity)
        else:
            row[start - 1] = Int64(infinity)
        var row_min = infinity
        for j in range(start, finish + 1):
            var above = Int(row[j])
            var deletion = above + 1
            var insertion = Int(row[j - 1]) + 1
            var substitution = diagonal
            if a[a_start + i - 1] != b[b_start + j - 1]:
                substitution += 1
            var best = deletion if deletion < insertion else insertion
            if substitution < best:
                best = substitution
            if best > infinity:
                best = infinity
            row[j] = Int64(best)
            if best < row_min:
                row_min = best
            diagonal = above
        if finish < bm:
            row[finish + 1] = Int64(infinity)
        if row_min > threshold:
            return 0
    return 1 if Int(row[bm]) <= threshold else 0


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
