// Twisted Edwards Curve Arithmetic

// 3021
const EDWARDS_D: Field = Field(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 3021));

const EDWARDS_D_PLUS_ONE: Field = Field(array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 3022));

struct AffinePoint {
    x: Field,
    y: Field
}

struct Point {
    x: Field,
    y: Field,
    t: Field,
    z: Field
}

struct MulPointIntermediate {
    result: Point,
    temp: Point,
    scalar: Field
}

const ZERO_POINT = Point(U256_ZERO, U256_ONE, U256_ZERO, U256_ONE);
const ZERO_AFFINE = AffinePoint(U256_ZERO, U256_ONE);

fn mul_by_a(f: ptr<function, Field>) -> Field {
    // mul by a is just negation of f
    var order = ALEO_FIELD_ORDER;
    return u256_sub(&order, f);
}

// follows aleo's projective addition algorithm
fn add_points(p1: ptr<function, Point>, p2: ptr<function, Point>) -> Point {
    var p1_x = (*p1).x;
    var p1_y = (*p1).y;
    var p1_t = (*p1).t;
    var p1_z = (*p1).z;
    var p2_x = (*p2).x;
    var p2_y = (*p2).y;
    var p2_t = (*p2).t;
    var p2_z = (*p2).z;

    var a = field_multiply(&p1_x, &p2_x);
    var b = field_multiply(&p1_y, &p2_y);
    var t_prod = field_multiply(&p1_t, &p2_t);
    var c = field_multiply_by_u32(&t_prod, 3021u);
    var d = field_multiply(&p1_z, &p2_z);
    var p1_added = field_add(&p1_x, &p1_y);
    var p2_added = field_add(&p2_x, &p2_y);
    var e = field_multiply(&p1_added, &p2_added);
    var h = field_add(&b, &a);
    e = field_sub(&e, &h);
    var f = field_sub(&d, &c);
    var g = field_add(&d, &c);
    var added_x = field_multiply(&e, &f);
    var added_y = field_multiply(&g, &h);
    var added_t = field_multiply(&e, &h);
    var added_z = field_multiply(&f, &g);
    return Point(added_x, added_y, added_t, added_z);
}

fn add_points_in_place(p1: ptr<function, Point>, p2: ptr<function, Point>) {
    var p1_x = (*p1).x;
    var p1_y = (*p1).y;
    var p1_t = (*p1).t;
    var p1_z = (*p1).z;
    var p2_x = (*p2).x;
    var p2_y = (*p2).y;
    var p2_t = (*p2).t;
    var p2_z = (*p2).z;

    var a = field_multiply(&p1_x, &p2_x);
    var b = field_multiply(&p1_y, &p2_y);
    var t_prod = field_multiply(&p1_t, &p2_t);
    var c = field_multiply_by_u32(&t_prod, 3021u);
    var d = field_multiply(&p1_z, &p2_z);
    var p1_added = field_add(&p1_x, &p1_y);
    var p2_added = field_add(&p2_x, &p2_y);
    var e = field_multiply(&p1_added, &p2_added);
    var h = field_add(&b, &a);
    e = field_sub(&e, &h);
    var f = field_sub(&d, &c);
    var g = field_add(&d, &c);
    (*p1).x = field_multiply(&e, &f);
    (*p1).y = field_multiply(&g, &h);
    (*p1).t = field_multiply(&e, &h);
    (*p1).z = field_multiply(&f, &g);
}

fn double_point_in_place(p: ptr<function, Point>) {
    var p_x = (*p).x;
    var p_y = (*p).y;
    var p_z = (*p).z;
    var p_t = (*p).t;

    var a = field_multiply(&p_x, &p_x);
    var b = field_multiply(&p_y, &p_y);
    var c = field_multiply(&p_z, &p_z);
    field_double_in_place(&c);
    var d = mul_by_a(&a);
    var h = field_sub(&d, &b); // -a - b = -(a + b)
    var e = field_add(&p_x, &p_y);
    e = field_multiply(&e, &e);
    e = field_add(&e, &h);
    var g = field_add(&d, &b);
    var f = field_sub(&g, &c);
    (*p).x = field_multiply(&e, &f);
    (*p).y = field_multiply(&g, &h);
    (*p).t = field_multiply(&e, &h);
    (*p).z = field_multiply(&f, &g);
}

// The functions below have not been touched during refactoring so they are not guaranteed to work.

// fn mul_point(p: Point, scalar: Field) -> Point {
//     var result: Point = Point(U256_ZERO, U256_ONE, U256_ZERO, U256_ONE);
//     var temp = p;
//     var scalar_iter = scalar;
//     while !equal(scalar_iter, U256_ZERO) {
//         if is_odd(scalar_iter) {
//             result = add_points(result, temp);
//             // result = temp;
//         }

//         temp = double_point(temp);
//         scalar_iter = u256_rs1(scalar_iter);
//     }

//     return result;
// }

// fn mul_point_64_bits_start(p: Point, scalar: Field) -> MulPointIntermediate {
//     var result: Point = Point(U256_ZERO, U256_ONE, U256_ZERO, U256_ONE);
//     var temp = p;
//     var scalar_iter = scalar;
//     for (var i = 0u; i < 64u; i = i + 1u) {
//         if equal(scalar_iter, U256_ZERO) {
//             break;
//         }

//         if is_odd(scalar_iter) {
//             result = add_points(result, temp);
//         }

//         temp = double_point(temp);

//         scalar_iter = u256_rs1(scalar_iter);
//     }

//     return MulPointIntermediate(result, temp, scalar_iter);
// }

// fn mul_point_64_bits(p: Point, scalar: Field, t: Point) -> MulPointIntermediate {
//     if equal(scalar, U256_ZERO) {
//         return MulPointIntermediate(p, t, scalar);
//     }

//     var result: Point = p;
//     var temp = t;
//     var scalar_iter = scalar;
//     for (var i = 0u; i < 64u; i = i + 1u) {
//         if equal(scalar_iter, U256_ZERO) { break; }

//         if is_odd(scalar_iter) {
//             result = add_points(result, temp);
//         }

//         temp = double_point(temp);

//         scalar_iter = u256_rs1(scalar_iter);
//     }

//     return MulPointIntermediate(result, temp, scalar_iter);
// }

// fn mul_point_test(p: Point, scalar: Field) -> Point {
//     var result: Point = Point(U256_ZERO, U256_ONE, U256_ONE, U256_ZERO);
//     var temp = p;
//     var scalar_iter = scalar;
//     while !equal(scalar_iter, U256_ZERO) {
//         if (scalar_iter.components[7u] & 1u) == 1u {
//             var added = add_points(result, temp);
//             result = added;
//         }

//         temp = double_point(temp);

//         var right_shifted = u256_rs1(scalar_iter);
//         scalar_iter = right_shifted;
//     }

//     return result;
// }

// fn mul_point_32_bit_scalar(p: Point, scalar: u32) -> Point {
//     var result: Point = Point(U256_ZERO, U256_ONE, U256_ZERO, U256_ONE);
//     var temp = p;

//     var scalar_iter = scalar;
//     while !(scalar_iter == 0u) {
//         if (scalar_iter & 1u) == 1u {
//             result = add_points(result, temp);
//         }
//         temp = double_point(temp);
//         scalar_iter = scalar_iter >> 1u;
//     }
//     return result;
// }