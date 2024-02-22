// Converts a 256-bit integer to its Montgomery form.
// Use with arith.wgsl.

@group(0) @binding(0)
var<storage, read> input: array<u32>;
@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

const WG_SIZE: u32 = 32;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = group_id.x * (WG_SIZE / 16u) + tid / 16u;
    let t = tid & 15u;
    var x: u32;
    if t < 8u {
        x = input[gid * 8u + t];
    } else {
        var r_squared = R_SQUARED;
        x = r_squared[t - 8u];
    }
    var m = mul_256(x, tid);
    m = redc_256(m, tid);
    if t >= 8u {
        output[gid * 8u + t - 8u] = m;
    }
}
