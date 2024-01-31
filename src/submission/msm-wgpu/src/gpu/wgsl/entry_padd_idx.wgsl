// A flexible elliptic curve point addition (PADD) kernel.

struct padd_index {
    // Index into buffer `inputs`
    in_idx_1: u32,
    // Index into `inputs`, or `buckets` if MSB is set, or nothing if 0xffffffffu.
    in_idx_2: u32,
    // Index into `outputs`, or `buckets` if MSB is set.
    out_idx: u32
}

@group(0) @binding(0)
var<storage, read> indices: array<padd_index>;
@group(0) @binding(1)
var<storage, read> inputs: array<Point>;
@group(0) @binding(2)
var<storage, read_write> outputs: array<Point>;
@group(0) @binding(3)
var<storage, read_write> buckets: array<Point>;
@group(0) @binding(4)
var<storage, read> indices_length: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= indices_length { return; }

    var indices = indices[global_id.x];
    var point_1 = inputs[indices.in_idx_1];
    var output: Point;

    if indices.in_idx_2 != 0xffffffffu {
        var point_2: Point;
        if (indices.in_idx_2 & 0x80000000u) == 0u {
            point_2 = inputs[indices.in_idx_2];
        } else {
            point_2 = buckets[indices.in_idx_2 & 0x7fffffffu];
        }
        output = add_points(&point_1, &point_2);
    } else {
        output = point_1;
    }
    if (indices.out_idx & 0x80000000u) == 0u {
        outputs[indices.out_idx] = output;
    } else {
        buckets[indices.out_idx & 0x7fffffffu] = output;
    }
}