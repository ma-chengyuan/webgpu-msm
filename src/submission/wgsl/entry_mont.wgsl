// Converts a 256-bit integer to its Montgomery form.

@group(0) @binding(0)
var<storage, read> input: array<Field>;
@group(0) @binding(1)
var<storage, read_write> output: array<Field>;

@compute @workgroup_size(64)
fn to_mont(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var r_squared = R_SQUARED;
    var x = input[global_id.x];
    var t_lo: u256;
    var t_hi: u256;
    u256_mul(&x, &r_squared, &t_lo, &t_hi);
    field_redc(&t_lo, &t_hi, &x);
    output[global_id.x] = x;
}

@compute @workgroup_size(64)
fn from_mont(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var zero = U256_ZERO;
    var x = input[global_id.x];
    var result: Field;
    field_redc(&x, &zero, &result);
    output[global_id.x] = result;
}