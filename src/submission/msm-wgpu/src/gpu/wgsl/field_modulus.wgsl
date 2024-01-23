// Finite field arithmetic

alias Field = u256;

struct Fields { fields: array<Field> }

// 8444461749428370424248824938781546531375899335154063827935233455917409239041
const ALEO_FIELD_ORDER: Field = Field(
    array<u32, 8>(313222494, 2586617174, 1622428958, 1547153409, 1504343806, 3489660929, 168919040, 1)
);

// 8444461749428370424248824938781546531375899335154063827935233455917409239042
const ALEO_FIELD_ORDER_PLUS_ONE: Field = Field(
    array<u32, 8>(313222494, 2586617174, 1622428958, 1547153409, 1504343806, 3489660929, 168919040, 2)
);

// 8444461749428370424248824938781546531375899335154063827935233455917409239040
const ALEO_FIELD_ORDER_MINUS_ONE: Field = Field(
    array<u32, 8>(313222494, 2586617174, 1622428958, 1547153409, 1504343806, 3489660929, 168919040, 0)
);

fn field_reduce(a: u256) -> Field {
    var reduction: Field = a;
    var a_gte_ALEO = gte(a, ALEO_FIELD_ORDER);

    while a_gte_ALEO {
        reduction = u256_sub(reduction, ALEO_FIELD_ORDER);
        a_gte_ALEO = gte(reduction, ALEO_FIELD_ORDER);
    }

    return reduction;
}

fn field_reduce_single(a: u256) -> Field {
    var reduction: Field = a;
    if gte(a, ALEO_FIELD_ORDER) {
        reduction = u256_sub(reduction, ALEO_FIELD_ORDER);
    }
    return reduction;
}

fn field_add(a: Field, b: Field) -> Field {
    var sum = u256_add(a, b);
    var result = field_reduce_single(sum);
    return result;
}

fn field_sub(a: Field, b: Field) -> Field {
    var sub: Field;
    if gte(a, b) {
        sub = u256_sub(a, b);
    } else {
        var b_minus_a: Field = u256_sub(b, a);
        sub = u256_sub(ALEO_FIELD_ORDER, b_minus_a);
    }

    return sub;
}

fn field_double(a: Field) -> Field {
    var double = u256_double(a);
    var result = field_reduce_single(double);
    return result;
}

fn field_multiply(a: Field, b: Field) -> Field {
    // return a;
    var accumulator: Field = U256_ZERO;
    var newA: Field = a;
    var b_components = b.components;

    for (var i = 7i; i >= 0i; i--) {
        var temp = b_components[i];
        for (var j = 0u; j < 32u; j++) {
            if (temp & 1u) == 1u { accumulator = field_add(accumulator, newA); }
            newA = field_double(newA);
            temp >>= 1u;
        }
    }

    return accumulator;
}

fn field_pow(base: Field, exponent: Field) -> Field {
    if equal(exponent, U256_ZERO) {
        return U256_ONE;
    }

    if equal(exponent, U256_ONE) {
        return base;
    }

    var exp = exponent;
    var bse = base;
    var result: u256 = U256_ONE;
    while gt(exp, U256_ZERO) {
        if is_odd(exp) { result = field_multiply(result, bse); }
        exp = u256_rs1(exp);
        bse = field_multiply(bse, bse);
    }

    return result;
}

// assume that the input is NOT 0, as there's no inverse for 0
// this function implements the Guajardo Kumar Paar Pelzl (GKPP) algorithm,
// Algorithm 16 (BEA for inversion in Fp)
fn field_inverse(num: Field) -> Field {
    var u: Field = num;
    var v: u256 = ALEO_FIELD_ORDER;
    var b: Field = U256_ONE;
    var c: Field = U256_ZERO;

    while !equal(u, U256_ONE) && !equal(v, U256_ONE) {
        while is_even(u) {
            // divide by 2
            u = u256_rs1(u);

            if is_even(b) {
                // divide by 2
                b = u256_rs1(b);
            } else {
                b = u256_add(b, ALEO_FIELD_ORDER);
                b = u256_rs1(b);
            }
        }

        while is_even(v) {
            // divide by 2
            v = u256_rs1(v);
            if is_even(c) {
                c = u256_rs1(c);
            } else {
                c = u256_add(c, ALEO_FIELD_ORDER);
                c = u256_rs1(c);
            }
        }

        if gte(u, v) {
            u = u256_sub(u, v);
            b = field_sub(b, c);
        } else {
            v = u256_sub(v, u);
            c = field_sub(c, b);
        }
    }

    if equal(u, U256_ONE) {
        return field_reduce(b);
    } else {
        return field_reduce(c);
    }
}