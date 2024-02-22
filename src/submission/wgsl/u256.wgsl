// Big-endian 256-bit unsigned integer operations

struct u256 { components: array<u32, 8> }

struct u256s { u256s: array<u256> }

// 115792089237316195423570985008687907853269984665640564039457584007913129639935
const U256_MAX: u256 = u256(array<u32, 8>(4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295));
const U256_ONE: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 1));
const U256_TWO: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 2));
const U256_ZERO: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0));

// no overflow checking for u256
fn u256_add(a: ptr<function, u256>, b: ptr<function, u256>) -> u256 {
    var sum = U256_ZERO;
    var carry: u32 = 0u;
    var total: u32;
    var first: u32;

    first = (*a).components[7];
    total = first + (*b).components[7];
    sum.components[7] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[6];
    total = first + (*b).components[6] + carry;
    sum.components[6] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[5];
    total = first + (*b).components[5] + carry;
    sum.components[5] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[4];
    total = first + (*b).components[4] + carry;
    sum.components[4] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[3];
    total = first + (*b).components[3] + carry;
    sum.components[3] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[2];
    total = first + (*b).components[2] + carry;
    sum.components[2] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[1];
    total = first + (*b).components[1] + carry;
    sum.components[1] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[0];
    total = first + (*b).components[0] + carry;
    sum.components[0] = total;

    return sum;
}

// no underflow checking for u256
fn u256_sub(a: ptr<function, u256>, b: ptr<function, u256>) -> u256 {
    var sub = U256_ZERO;
    var carry: u32 = 0u;
    var total: u32;
    var first: u32;

    first = (*a).components[7];
    total = first - (*b).components[7] - carry;
    sub.components[7] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[6];
    total = first - (*b).components[6] - carry;
    sub.components[6] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[5];
    total = first - (*b).components[5] - carry;
    sub.components[5] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[4];
    total = first - (*b).components[4] - carry;
    sub.components[4] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[3];
    total = first - (*b).components[3] - carry;
    sub.components[3] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[2];
    total = first - (*b).components[2] - carry;
    sub.components[2] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[1];
    total = first - (*b).components[1] - carry;
    sub.components[1] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[0];
    total = first - (*b).components[0] - carry;
    sub.components[0] = total;

    return sub;
}

fn u256_rs1(a: ptr<function, u256>) -> u256 {
    var right_shifted = U256_ZERO;
    var carry: u32 = 0u;
    var orig: u32;

    orig = (*a).components[0];
    right_shifted.components[0] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[1];
    right_shifted.components[1] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[2];
    right_shifted.components[2] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[3];
    right_shifted.components[3] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[4];
    right_shifted.components[4] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[5];
    right_shifted.components[5] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[6];
    right_shifted.components[6] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = (*a).components[7];
    right_shifted.components[7] = (orig >> 1u) | carry;

    return right_shifted;
}

fn u256_double(a: ptr<function, u256>) -> u256 {
    var double: u256 = U256_ZERO;
    var carry: u32 = 0u;
    var total: u32;
    var single: u32;

    single = (*a).components[7];
    total = single << 1u;
    double.components[7] = total + carry;
    carry = u32(total < single);

    single = (*a).components[6];
    total = single << 1u;
    double.components[6] = total + carry;
    carry = u32(total < single);

    single = (*a).components[5];
    total = single << 1u;
    double.components[5] = total + carry;
    carry = u32(total < single);

    single = (*a).components[4];
    total = single << 1u;
    double.components[4] = total + carry;
    carry = u32(total < single);

    single = (*a).components[3];
    total = single << 1u;
    double.components[3] = total + carry;
    carry = u32(total < single);

    single = (*a).components[2];
    total = single << 1u;
    double.components[2] = total + carry;
    carry = u32(total < single);

    single = (*a).components[1];
    total = single << 1u;
    double.components[1] = total + carry;
    carry = u32(total < single);

    single = (*a).components[0];
    total = single << 1u;
    double.components[0] = total + carry;
    carry = u32(total < single);

    return double;
}

fn is_even(a: ptr<function, u256>) -> bool { return ((*a).components[7u] & 1u) == 0u; }

fn is_odd(a: ptr<function, u256>) -> bool { return ((*a).components[7u] & 1u) == 1u; }

// underflow allowed u256 subtraction
fn u256_subw(a: ptr<function, u256>, b: ptr<function, u256>) -> u256 {
    var sub: u256;
    if gte(a, b) {
        sub = u256_sub(a, b);
    } else {
        var b_minus_a: u256 = u256_sub(b, a);
        var one = U256_ONE;
        var b_minus_a_minus_one: u256 = u256_sub(&b_minus_a, &one);
        var max = U256_MAX;
        sub = u256_sub(&max, &b_minus_a_minus_one);
    }
    return sub;
}


fn equal(a: ptr<function, u256>, b: ptr<function, u256>) -> bool {
    var a_components = (*a).components;
    var b_components = (*b).components;
    for (var i = 0u; i < 8u; i++) {
        if a_components[i] != b_components[i] {
            return false;
        }
    }

    return true;
}

// returns whether a > b
fn gt(a: ptr<function, u256>, b: ptr<function, u256>) -> bool {
    var a_components = (*a).components;
    var b_components = (*b).components;
    for (var i = 0u; i < 8u; i++) {
        if a_components[i] < b_components[i] {
            return false;
        }
        if a_components[i] > b_components[i] {
            return true;
        }
    }
    // if a's components are never greater, than a is equal to b
    return false;
}

// returns whether a >= b
fn gte(a: ptr<function, u256>, b: ptr<function, u256>) -> bool {
    var a_components = (*a).components;
    var b_components = (*b).components;
    for (var i = 0u; i < 8u; i++) {
        if a_components[i] < b_components[i] {
            return false;
        }
        if a_components[i] > b_components[i] {
            return true;
        }
    }
    return true;
}

fn gte_field_order(a: ptr<function, u256>) -> bool {
    var a_components = (*a).components;
    if a_components[0] < ALEO_FIELD_ORDER.components[0] { return false; }
    if a_components[0] > ALEO_FIELD_ORDER.components[0] { return true; }
    if a_components[1] < ALEO_FIELD_ORDER.components[1] { return false; }
    if a_components[1] > ALEO_FIELD_ORDER.components[1] { return true; }
    if a_components[2] < ALEO_FIELD_ORDER.components[2] { return false; }
    if a_components[2] > ALEO_FIELD_ORDER.components[2] { return true; }
    if a_components[3] < ALEO_FIELD_ORDER.components[3] { return false; }
    if a_components[3] > ALEO_FIELD_ORDER.components[3] { return true; }
    if a_components[4] < ALEO_FIELD_ORDER.components[4] { return false; }
    if a_components[4] > ALEO_FIELD_ORDER.components[4] { return true; }
    if a_components[5] < ALEO_FIELD_ORDER.components[5] { return false; }
    if a_components[5] > ALEO_FIELD_ORDER.components[5] { return true; }
    if a_components[6] < ALEO_FIELD_ORDER.components[6] { return false; }
    if a_components[6] > ALEO_FIELD_ORDER.components[6] { return true; }
    if a_components[7] < ALEO_FIELD_ORDER.components[7] { return false; }
    if a_components[7] > ALEO_FIELD_ORDER.components[7] { return true; }
    return true;
}

fn sub_field_order_in_place(a: ptr<function, u256>) {
    var carry: u32 = 0u;
    var total: u32;
    var first: u32;

    first = (*a).components[7];
    total = first - ALEO_FIELD_ORDER.components[7] - carry;
    (*a).components[7] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[6];
    total = first - ALEO_FIELD_ORDER.components[6] - carry;
    (*a).components[6] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[5];
    total = first - ALEO_FIELD_ORDER.components[5] - carry;
    (*a).components[5] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[4];
    total = first - ALEO_FIELD_ORDER.components[4] - carry;
    (*a).components[4] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[3];
    total = first - ALEO_FIELD_ORDER.components[3] - carry;
    (*a).components[3] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[2];
    total = first - ALEO_FIELD_ORDER.components[2] - carry;
    (*a).components[2] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[1];
    total = first - ALEO_FIELD_ORDER.components[1] - carry;
    (*a).components[1] = total;
    carry = u32(total > first || (total + carry) > first);

    first = (*a).components[0];
    total = first - ALEO_FIELD_ORDER.components[0] - carry;
    (*a).components[0] = total;
}

fn u256_add_in_place(a: ptr<function, u256>, b: ptr<function, u256>) {
    var carry: u32 = 0u;
    var total: u32;
    var first: u32;

    first = (*a).components[7];
    total = first + (*b).components[7];
    (*a).components[7] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[6];
    total = first + (*b).components[6] + carry;
    (*a).components[6] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[5];
    total = first + (*b).components[5] + carry;
    (*a).components[5] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[4];
    total = first + (*b).components[4] + carry;
    (*a).components[4] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[3];
    total = first + (*b).components[3] + carry;
    (*a).components[3] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[2];
    total = first + (*b).components[2] + carry;
    (*a).components[2] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[1];
    total = first + (*b).components[1] + carry;
    (*a).components[1] = total;
    carry = u32(total < first || (total - carry) < first);

    first = (*a).components[0];
    total = first + (*b).components[0] + carry;
    (*a).components[0] = total;
}

fn u256_double_in_place(a: ptr<function, u256>) {
    var carry: u32 = 0u;
    var total: u32;
    var single: u32;

    single = (*a).components[7];
    total = single << 1u;
    (*a).components[7] = total + carry;
    carry = u32(total < single);

    single = (*a).components[6];
    total = single << 1u;
    (*a).components[6] = total + carry;
    carry = u32(total < single);

    single = (*a).components[5];
    total = single << 1u;
    (*a).components[5] = total + carry;
    carry = u32(total < single);

    single = (*a).components[4];
    total = single << 1u;
    (*a).components[4] = total + carry;
    carry = u32(total < single);

    single = (*a).components[3];
    total = single << 1u;
    (*a).components[3] = total + carry;
    carry = u32(total < single);

    single = (*a).components[2];
    total = single << 1u;
    (*a).components[2] = total + carry;
    carry = u32(total < single);

    single = (*a).components[1];
    total = single << 1u;
    (*a).components[1] = total + carry;
    carry = u32(total < single);

    single = (*a).components[0];
    total = single << 1u;
    (*a).components[0] = total + carry;
    carry = u32(total < single);
}

fn u256_mul(a: ptr<function, u256>, b: ptr<function, u256>, lo: ptr<function, u256>, hi: ptr<function, u256>) {
    var temp: array<u32, 32>;
    for (var i = 15; i >= 0; i--) {
        var a_digit = (*a).components[i >> 1u];
        a_digit = select(a_digit & 0xffffu, a_digit >> 16u, (i & 1) == 0);
        for (var j = 15; j >= 0; j--) {
            var b_digit = (*b).components[j >> 1u];
            b_digit = select(b_digit & 0xffffu, b_digit >> 16u, (j & 1) == 0);
            let prod = a_digit * b_digit;
            temp[i + j + 1] += prod & 0xffffu;
            temp[i + j] += prod >> 16u;
        }
    }
    var carry = 0u;
    for (var i = 31; i >= 0; i--) {
        temp[i] += carry;
        carry = temp[i] >> 16u;
        temp[i] &= 0xffffu;
    }
    for (var i = 15; i >= 8; i--) {
        (*lo).components[i - 8] = temp[(i << 1u) + 1] | (temp[i << 1u] << 16u);
    }
    for (var i = 7; i >= 0; i--) {
        (*hi).components[i] = temp[(i << 1u) + 1] | (temp[i << 1u] << 16u);
    }
}

fn u256_mul_lo(a: ptr<function, u256>, b: ptr<function, u256>, lo: ptr<function, u256>) {
    var temp: array<u32, 16>;
    for (var i = 15; i >= 0; i--) {
        var a_digit = (*a).components[i >> 1u];
        a_digit = select(a_digit & 0xffffu, a_digit >> 16u, (i & 1) == 0);
        for (var j = 15; j >= 15 - i; j--) {
            var b_digit = (*b).components[j >> 1u];
            b_digit = select(b_digit & 0xffffu, b_digit >> 16u, (j & 1) == 0);
            let prod = a_digit * b_digit;
            temp[i + j - 15] += prod & 0xffffu;
            if i + j >= 16 { temp[i + j - 16] += prod >> 16u; }
        }
    }
    var carry = 0u;
    for (var i = 15; i >= 0; i--) {
        temp[i] += carry;
        carry = temp[i] >> 16u;
        temp[i] &= 0xffffu;
    }
    for (var i = 7; i >= 0; i--) {
        (*lo).components[i] = temp[(i << 1u) + 1] | (temp[i << 1u] << 16u);
    }
}

fn u512_add_inplace(
    a_lo: ptr<function, u256>, a_hi: ptr<function, u256>,
    b_lo: ptr<function, u256>, b_hi: ptr<function, u256>
) {
    var carry = 0u;
    for (var i = 7; i >= 0; i--) {
        let a_component = (*a_lo).components[i];
        var sum = a_component + (*b_lo).components[i] + carry;
        (*a_lo).components[i] = sum;
        carry = u32(sum < a_component || (sum - carry) < a_component);
    }
    for (var i = 7; i >= 0; i--) {
        let a_component = (*a_hi).components[i];
        var sum = a_component + (*b_hi).components[i] + carry;
        (*a_hi).components[i] = sum;
        carry = u32(sum < a_component || (sum - carry) < a_component);
    }
}