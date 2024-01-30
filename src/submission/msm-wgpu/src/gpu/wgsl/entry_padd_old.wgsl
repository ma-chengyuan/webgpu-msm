// Takes in two points in Montgomery form and returns the sum of the two points

@group(0) @binding(1)
var<storage, read_write> output: array<Point>;
@group(0) @binding(0)
var<storage, read> input: array<Point>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var point_1 = input[2u * global_id.x];
    var point_2 = input[2u * global_id.x + 1u];
    output[global_id.x] = add_points(&point_1, &point_2);
}