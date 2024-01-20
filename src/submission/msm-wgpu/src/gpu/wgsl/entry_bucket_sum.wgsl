// Reduction kernel for the Pippenger algorithm

// Reduction: 
// Given: 
//   A1 = (n/2) * sum_{i=0}^{n/2-1} points[i],
//   A2 = sum_{i=0}^{n/2-1} i*points[i],
//   B1 = (n/2) * sum_{i=0}^{n/2-1} points[i+n/2],
//   B2 = sum_{i=0}^{n/2-1} i*points[i+n/2],
// Observe that:
//   n * sum_{i=0}^{n-1} points[i] = 2 * (A1 + B1)
//   sum_{i=0}^{n-1} i*points[i] = A1 + B2 + B1
// This kernel applies this reduction 6 times, to reduce 64 points to 1 for
// every workgroup.

@group(0) @binding(0)
var<storage, read> i_sum_n: array<Point>; // n * sum(point[i])
@group(0) @binding(1)
var<storage, read> i_sum_i: array<Point>; // sum(i * point[i])
@group(0) @binding(2)
var<storage, read_write> o_sum_n: array<Point>; // n * sum(point[i])
@group(0) @binding(3)
var<storage, read_write> o_sum_i: array<Point>; // sum(i * point[i])

// Scratch space
// Note: this is the maximum amount of workgroup address space storage we can
// use per WebGPU spec.
var<workgroup> t_sum_n: array<array<Point, 32>, 2>;
var<workgroup> t_sum_i: array<array<Point, 32>, 2>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    var cur_idx = 0u;
    if (local_id.x & 1u) == 0u {
        t_sum_n[cur_idx][local_id.x >> 1u] = double_point(add_points(
            i_sum_n[global_id.x],
            i_sum_n[global_id.x + 1u]
        ));
        t_sum_i[cur_idx][local_id.x >> 1u] = add_points(
            i_sum_i[global_id.x],
            add_points(i_sum_i[global_id.x + 1u], i_sum_n[global_id.x + 1u])
        );
    }
    workgroupBarrier();
    for (var i = 2u; i <= 6u; i++) {
        let prev_idx = cur_idx;
        cur_idx = 1 - cur_idx;
        let mask = (1u << i) - 1u;
        if (local_id.x & mask) == 0u {
            let idx = local_id.x >> i;
            let a = idx << 1u;
            let b = a + 1u;
            t_sum_n[cur_idx][idx] = double_point(add_points(
                t_sum_n[prev_idx][a],
                t_sum_n[prev_idx][b]
            ));
            t_sum_i[cur_idx][idx] = add_points(
                t_sum_i[prev_idx][a],
                add_points(t_sum_i[prev_idx][b], t_sum_n[prev_idx][b])
            );
        }
        workgroupBarrier();
    }
    if local_id.x == 0u {
        o_sum_n[group_id.x] = t_sum_n[cur_idx][0];
        o_sum_i[group_id.x] = t_sum_i[cur_idx][0];
    }
}