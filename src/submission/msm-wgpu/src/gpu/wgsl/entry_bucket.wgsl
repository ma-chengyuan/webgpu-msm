// The bucketing / EC point histogram kernel.

// offsets[0] = first bucket to consider in the current dispatch
// offsets[1] = total number of buckets considered in the current dispatch
// offsets[2 + i]: the starting offset of the i'th bucket in the points array
@group(0) @binding(0)
var<storage, read> offsets: array<u32>;
@group(0) @binding(1)
var<storage, read> points: array<Point>;
@group(0) @binding(2)
var<storage, read> buckets: array<u32>;
@group(0) @binding(3)
var<storage, read_write> sums: array<Point>; // The sums array, one sum per bucket.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bucket = buckets[global_id.x];
    if bucket == 0u { return; }
    let offset = offsets[global_id.x];
    let next_offset = offsets[global_id.x + 1];

    var point = sums[bucket];
    for (var i = offset; i < next_offset; i++) {
        var cur_point = points[i];
        point = add_points(&point, &cur_point);
    }
    sums[bucket] = point;
}