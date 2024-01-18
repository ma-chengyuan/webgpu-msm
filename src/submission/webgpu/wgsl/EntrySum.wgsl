@group(0) @binding(0)
var<storage, read> input1: array<Point>;
@group(0) @binding(1)
var<storage, read_write> output: array<Point>;

var<workgroup> stage_1: array<Point, 32>;
var<workgroup> stage_2: array<Point, 16>;
var<workgroup> stage_3: array<Point, 8>;
var<workgroup> stage_4: array<Point, 4>;
var<workgroup> stage_5: array<Point, 2>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    if (local_id.x & 1u) == 0u {
        stage_1[local_id.x >> 1u] = add_points(
            input1[global_id.x],
            input1[global_id.x + 1u]
        );
    }
    workgroupBarrier();
    if (local_id.x & 3u) == 0u {
        stage_2[local_id.x >> 2u] = add_points(
            stage_1[(local_id.x >> 1u)],
            stage_1[(local_id.x >> 1u) + 1u]
        );
    }
    workgroupBarrier();
    if (local_id.x & 7u) == 0u {
        stage_3[local_id.x >> 3u] = add_points(
            stage_2[(local_id.x >> 2u)],
            stage_2[(local_id.x >> 2u) + 1u]
        );
    }
    workgroupBarrier();
    if (local_id.x & 15u) == 0u {
        stage_4[local_id.x >> 4u] = add_points(
            stage_3[(local_id.x >> 3u)],
            stage_3[(local_id.x >> 3u) + 1u]
        );
    }
    workgroupBarrier();
    if (local_id.x & 31u) == 0u {
        stage_5[local_id.x >> 5u] = add_points(
            stage_4[(local_id.x >> 4u)],
            stage_4[(local_id.x >> 4u) + 1u]
        );
    }
    workgroupBarrier();
    if local_id.x == 0u {
        let result = add_points(stage_5[0u], stage_5[1u]);
        output[global_id.x] = result;
    }
}