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

fn field_reduce_single(a: ptr<function, u256>) {
    if gte_field_order(a) {
        sub_field_order_in_place(a);
    }
}

fn field_add(a: ptr<function, Field>, b: ptr<function, Field>) -> Field {
    var sum = u256_add(a, b);
    field_reduce_single(&sum);
    return sum;
}

fn field_sub(a: ptr<function, Field>, b: ptr<function, Field>) -> Field {
    var sub: Field;
    if gte(a, b) {
        sub = u256_sub(a, b);
    } else {
        var b_minus_a: Field = u256_sub(b, a);
        var order = ALEO_FIELD_ORDER;
        sub = u256_sub(&order, &b_minus_a);
    }
    return sub;
}

fn field_double(a: ptr<function, Field>) -> Field {
    var double = u256_double(a);
    field_reduce_single(&double);
    return double;
}

fn field_double_in_place(a: ptr<function, Field>) {
    u256_double_in_place(a);
    field_reduce_single(a);
}

fn field_multiply(a: ptr<function, Field>, b: ptr<function, Field>) -> Field {
    // return a;
    var accumulator: Field = U256_ZERO;
    var newA: Field = *a;
    var b_components = (*b).components;

    for (var i = 7i; i >= 0i; i--) {
        var temp = b_components[i];
        for (var j = 0u; j < 32u; j++) {
            if (temp & 1u) == 1u {
                u256_add_in_place(&accumulator, &newA);
                field_reduce_single(&accumulator);
            }
            u256_double_in_place(&newA);
            field_reduce_single(&newA);
            temp >>= 1u;
        }
    }

    return accumulator;
}

fn field_multiply_by_u32(a: ptr<function, Field>, c: u32) -> Field {
    var accumulator: Field = U256_ZERO;
    var newA: Field = *a;
    var temp = c;
    while temp > 0u {
        if (temp & 1u) == 1u {
            u256_add_in_place(&accumulator, &newA);
            field_reduce_single(&accumulator);
        }
        u256_double_in_place(&newA);
        field_reduce_single(&newA);
        temp >>= 1u;
    }
    return accumulator;
}