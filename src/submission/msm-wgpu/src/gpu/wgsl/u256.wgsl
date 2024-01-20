// Big-endian 256-bit unsigned integer operations

struct u256 { components: array<u32, 8> }

struct u256s { u256s: array<u256> }

// 115792089237316195423570985008687907853269984665640564039457584007913129639935
const U256_MAX: u256 = u256(array<u32, 8>(4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295));
const U256_ONE: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 1));
const U256_TWO: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 2));
const U256_ZERO: u256 = u256(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0));

// no overflow checking for u256
fn u256_add(a: u256, b: u256) -> u256 {
    var sum = U256_ZERO;
    var carry: u32 = 0u;
    var total: u32;
    var first: u32;

    first = a.components[7];
    total = first + b.components[7] + carry;
    sum.components[7] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[6];
    total = first + b.components[6] + carry;
    sum.components[6] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[5];
    total = first + b.components[5] + carry;
    sum.components[5] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[4];
    total = first + b.components[4] + carry;
    sum.components[4] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[3];
    total = first + b.components[3] + carry;
    sum.components[3] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[2];
    total = first + b.components[2] + carry;
    sum.components[2] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[1];
    total = first + b.components[1] + carry;
    sum.components[1] = total;
    carry = u32(total < first || (total - carry) < first);

    first = a.components[0];
    total = first + b.components[0] + carry;
    sum.components[0] = total;

    return sum;
}

// no underflow checking for u256
fn u256_sub(a: u256, b: u256) -> u256 {
    var sub = U256_ZERO;
    var carry: u32 = 0u;
    var total: u32;
    var first: u32;

    first = a.components[7];
    total = first - b.components[7] - carry;
    sub.components[7] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[6];
    total = first - b.components[6] - carry;
    sub.components[6] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[5];
    total = first - b.components[5] - carry;
    sub.components[5] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[4];
    total = first - b.components[4] - carry;
    sub.components[4] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[3];
    total = first - b.components[3] - carry;
    sub.components[3] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[2];
    total = first - b.components[2] - carry;
    sub.components[2] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[1];
    total = first - b.components[1] - carry;
    sub.components[1] = total;
    carry = u32(total > first || (total + carry) > first);

    first = a.components[0];
    total = first - b.components[0] - carry;
    sub.components[0] = total;

    return sub;
}


fn u256_rs1(a: u256) -> u256 {
    var right_shifted = U256_ZERO;
    var carry: u32 = 0u;
    var orig: u32;

    orig = a.components[0];
    right_shifted.components[0] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[1];
    right_shifted.components[1] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[2];
    right_shifted.components[2] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[3];
    right_shifted.components[3] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[4];
    right_shifted.components[4] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[5];
    right_shifted.components[5] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[6];
    right_shifted.components[6] = (orig >> 1u) | carry;
    carry = orig << 31u;

    orig = a.components[7];
    right_shifted.components[7] = (orig >> 1u) | carry;

    return right_shifted;
}

fn u256_double(a: u256) -> u256 {
    var double: u256 = U256_ZERO;
    var carry: u32 = 0u;
    var total: u32;
    var single: u32;

    single = a.components[7];
    total = single << 1u;
    double.components[7] = total + carry;
    carry = u32(total < single);

    single = a.components[6];
    total = single << 1u;
    double.components[6] = total + carry;
    carry = u32(total < single);

    single = a.components[5];
    total = single << 1u;
    double.components[5] = total + carry;
    carry = u32(total < single);

    single = a.components[4];
    total = single << 1u;
    double.components[4] = total + carry;
    carry = u32(total < single);

    single = a.components[3];
    total = single << 1u;
    double.components[3] = total + carry;
    carry = u32(total < single);

    single = a.components[2];
    total = single << 1u;
    double.components[2] = total + carry;
    carry = u32(total < single);

    single = a.components[1];
    total = single << 1u;
    double.components[1] = total + carry;
    carry = u32(total < single);

    single = a.components[0];
    total = single << 1u;
    double.components[0] = total + carry;
    carry = u32(total < single);

    return double;
}

fn is_even(a: u256) -> bool { return (a.components[7u] & 1u) == 0u; }

fn is_odd(a: u256) -> bool { return (a.components[7u] & 1u) == 1u; }

// underflow allowed u256 subtraction
fn u256_subw(a: u256, b: u256) -> u256 {
    var sub: u256;
    if gte(a, b) {
        sub = u256_sub(a, b);
    } else {
        var b_minus_a: u256 = u256_sub(b, a);
        var b_minus_a_minus_one: u256 = u256_sub(b_minus_a, U256_ONE);
        sub = u256_sub(U256_MAX, b_minus_a_minus_one);
    }

    return sub;
}


fn equal(a: u256, b: u256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if a.components[i] != b.components[i] {
            return false;
        }
    }

    return true;
}

// returns whether a > b
fn gt(a: u256, b: u256) -> bool {
    for (var i = 0u; i < 8u; i++) {
        if a.components[i] < b.components[i] {
            return false;
        }

        if a.components[i] > b.components[i] {
            return true;
        }
    }
    // if a's components are never greater, than a is equal to b
    return false;
}

// returns whether a >= b
fn gte(a: u256, b: u256) -> bool {
    var agtb = gt(a, b);
    var aeqb = equal(a, b);
    return agtb || aeqb;
}
