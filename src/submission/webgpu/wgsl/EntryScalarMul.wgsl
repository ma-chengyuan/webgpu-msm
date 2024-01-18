@group(0) @binding(0)
var<storage, read> input1: array<Point>;
@group(0) @binding(1)
var<storage, read> input2: array<u32>;
@group(0) @binding(2)
var<storage, read_write> output: array<Point>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var extended_point = input1[global_id.x];
    var scalar = input2[global_id.x];

    var result = mul_point_32_bit_scalar(extended_point, scalar);

    output[global_id.x] = result;
}