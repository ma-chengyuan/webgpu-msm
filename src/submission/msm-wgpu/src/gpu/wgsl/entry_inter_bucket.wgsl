// Reduction kernel for the Pippenger algorithm

// Reduction: 
// Given: 
//   A1 = sum_{i=0}^{n/2-1} i*points[i],
//   A2 = (n/2) * sum_{i=0}^{n/2-1} points[i],
//   B1 = sum_{i=0}^{n/2-1} i*points[i+n/2],
//   B2 = (n/2) * sum_{i=0}^{n/2-1} points[i+n/2],
// Observe that:
//   sum_{i=0}^{n-1} i*points[i] = A1 + B1 + B2
//   n * sum_{i=0}^{n-1} points[i] = 2 * (A2 + B2)
// This kernel applies this reduction 6 times, to reduce 64 points to 1 for
// every workgroup.

@group(0) @binding(0)
var<storage, read> input: array<Point>; 
@group(0) @binding(1)
var<storage, read_write> output: array<Point>; 
@group(0) @binding(2)
var<storage, read> input_length: u32;
@group(0) @binding(3)
var<storage, read_write> output_length: u32;

@compute @workgroup_size(64)
fn main_1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= input_length / 2u { return; }
    if global_id.x == 0u { output_length = input_length / 2u; }

    var i1 = global_id.x << 1u;
    var i2 = i1 + 1u;

    var p1 = input[i1];
    var p2 = input[i2];
    output[i1] = p2;
    var n_sum = p1;
    add_points_in_place(&n_sum, &p2);
    double_point_in_place(&n_sum);
    output[i2] = n_sum;
}

@compute @workgroup_size(64)
fn main_2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= input_length / 2u { return; }
    if global_id.x == 0u { output_length = input_length / 2u; }

    var i1 = global_id.x << 1u;
    var i2 = i1 + 1u;

    var p_a1 = input[i1 << 1u];
    var p_a2 = input[(i1 << 1u) + 1u];
    var p_b1 = input[i2 << 1u];
    var p_b2 = input[(i2 << 1u) + 1u];
    var n_sum = p_a2;
    add_points_in_place(&n_sum, &p_b2);
    double_point_in_place(&n_sum);
    output[i2] = n_sum;
    var i_sum = p_a1;
    add_points_in_place(&i_sum, &p_b1);
    add_points_in_place(&i_sum, &p_b2);
    output[i1] = i_sum;
}