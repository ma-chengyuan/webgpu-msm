// Takes in two points in Montgomery form and returns the sum of the two points

@group(0) @binding(0)
var<storage, read> input: array<u32>;
@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

const WG_SIZE: u32 = 64;
// const EDWARDS_D: u32 = 3021u;
// EDWARDS_D in Montgomery form
const EDWARDS_D_MONT: array<u32, 8> = array<u32, 8>(4294925872, 3494379519, 4294924242, 4037611558, 219161986, 151076694, 2814269184, 167584812);

var<workgroup> scratch: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    padd_32(local_id.x, 2u * group_id.x + (local_id.x >> 5u));
}

fn padd_32(tid: u32, gid: u32) {
    let p1_x_base = (2 * gid) * 32u;
    let p1_y_base = p1_x_base + 8u;
    let p1_t_base = p1_x_base + 16u;
    let p1_z_base = p1_x_base + 24u;

    let p2_x_base = p1_x_base + 32u;
    let p2_y_base = p2_x_base + 8u;
    let p2_t_base = p2_x_base + 16u;
    let p2_z_base = p2_x_base + 24u;

    let out_x_base = gid * 32u;
    let out_y_base = out_x_base + 8u;
    let out_t_base = out_x_base + 16u;
    let out_z_base = out_x_base + 24u;

    let t = tid & 7u;
    let t_ = tid & 31u;
    let l = (tid >> 3u) & 3u;
    var n_ = N;
    var n = n_[t + 8u];

    var a: u32;
    switch l {
        case 0u: { a = input[p1_x_base + t]; }
        case 1u: { a = input[p2_x_base + t]; }
        case 2u: { a = input[p1_y_base + t]; }
        case 3u: { a = input[p2_y_base + t]; }
        default: {}
    }
    // Threads 0-15 compute p1.x * p2.x, threads 16-31 compute p1.y * p2.y
    a = mul_256(a, tid);
    a = redc_256(a, tid);
    // a: threads 8-15 holds the result of p1.x * p2.x, threads 24-31 hold the result of p1.y * p2.y

    var b: u32;
    switch l {
        case 0u: { b = input[p1_z_base + t]; }
        case 1u: { b = input[p2_z_base + t]; }
        case 2u: { b = input[p1_t_base + t]; }
        case 3u: { b = input[p2_t_base + t]; }
        default: {}
    }
    // Threads 0-15 compute p1.z * p2.z, threads 16-31 compute p1.t * p2.t
    b = mul_256(b, tid);
    b = redc_256(b, tid);
    // b: threads 8-15 holds the result of p1.z * p2.z, threads 24-31 hold the result of p1.t * p2.t

    var c: u32;
    var d: u32;
    switch l {
        case 0u: { c = input[p1_x_base + t]; d = input[p1_y_base + t]; }
        case 1u: { c = input[p2_x_base + t]; d = input[p2_y_base + t]; }
        default: { c = 0u; d = 0u; }
    }
    c = add_256(c, d, tid);
    c = cas_256(c, n, tid);
    // c: threads 0-7 hold the result of p1.x + p1.y, threads 8-15 hold the result of p2.x + p2.y
    // threads 16-31 hold 0

    var e: u32 = select(select(b, select(0u, 1u, t_ == 16u), t_ < 24u), c, t_ < 16u);
    switch l {
        // Threads 0-15 set e = c (i.e., p1.x + p1.y for 0-7, p2.x + p2.y for 8-15)
        case 0u, 1u: { e = c; }    
        // Threads 16-23 set e = EDWARDS_D
        case 2u: { var edwards_d = EDWARDS_D_MONT; e = edwards_d[t]; }
        // Threads 24-31 set e = b (i.e., p1.t * p2.t for 24-31)
        case 3u: { e = b; }
        default: {}
    }
    e = mul_256(e, tid);
    e = redc_256(e, tid);
    // e: threads 8-15 hold the result of (p1.x + p1.y) * (p2.x + p2.y); threads 24-31 hold the result of EDWARDS_D * (p1.t * p2.t),

    if (tid & 8u) == 8u { // Applies to threads 8-15 and 24-31
        scratch[tid - 8u] = a;
        scratch[tid] = e;
        // Layout: 
        // - scratch[0:7]: p1.x * p2.x
        // - scratch[8:15]: (p1.x + p1.y) * (p2.x + p2.y)
        // - scratch[16:32]: p1.y * p2.y
        // - scratch[24:31]: EDWARDS_D * (p1.t * p2.t)
    }
    workgroupBarrier();
    var f: u32;
    var g: u32;
    switch l {
        // f = p1.x * p2.x; g = p1.y * p2.y
        case 0u: { f = scratch[tid]; g = scratch[tid + 16u]; }
        // f = p1.z * p2.z; g = EDWARDS_D * (p1.t * p2.t)
        case 1u: { f = b; g = scratch[tid + 16u]; }
        default: { f = 0u; g = 0u; }
    }
    f = add_256(f, g, tid);
    f = cas_256(f, n, tid);

    // f: threads 0-7 hold the result of (p1.x * p2.x) + (p1.y * p2.y), 
    // threads 8-15 hold the result of (p1.z * p2.z) + (EDWARDS_D * (p1.t * p2.t)),
    // threads 16-31 hold 0
    var h: u32;
    var i: u32;
    switch l {
        // h = (p1.x + p1.y) * (p2.x + p2.y); i = (p1.x * p2.x) + (p1.y * p2.y)
        case 0u: { h = scratch[tid + 8u]; i = f; }
        // h = p1.z * p2.z; i = EDWARDS_D * (p1.t * p2.t)
        case 1u: { h = b; i = scratch[tid + 16u]; }
        default: { h = 0u; i = 0u; }
    }
    // Start performing subtractions
    var sign = cmp_256(h, i, tid);
    // Swap if h < i to avoid underflow
    if sign < 0 { var tmp = h; h = i; i = tmp; }
    h = sub_256(h, i, tid);
    // Correct sign 
    if sign < 0 { h = sub_256(n, h, tid); }
    // h: threads 0-7 hold the result of (p1.x + p1.y) * (p2.x + p2.y) - ((p1.x * p2.x) + (p1.y * p2.y)), 
    // threads 8-15 hold the result of (p1.z * p2.z) - (EDWARDS_D * (p1.t * p2.t)),
    // threads 16-31 hold 0

    if t_ < 16u {
        scratch[tid] = f;
        scratch[tid + 16u] = h;
        f = h;
        // Layout: 
        // scratch[0:7]: (p1.x + p1.y) * (p2.x + p2.y)
        // scratch[8:15]: (p1.z * p2.z) + (EDWARDS_D * (p1.t * p2.t))
        // scratch[16:23]: (p1.x + p1.y) * (p2.x + p2.y) - ((p1.x * p2.x) + (p1.y * p2.y))
        // scratch[24:31]: (p1.z * p2.z) - (EDWARDS_D * (p1.t * p2.t))
    }
    workgroupBarrier();
    if t_ >= 16u { f = scratch[tid - 16u]; }
    // f: threads 0-7 hold the result of (p1.x + p1.y) * (p2.x + p2.y) - ((p1.x * p2.x) + (p1.y * p2.y)),
    // threads 8-15 hold the result of (p1.z * p2.z) - (EDWARDS_D * (p1.t * p2.t)),
    // threads 16-23 hold the result of (p1.x * p2.x) + (p1.y * p2.y),
    // threads 24-31 hold the result of (p1.z * p2.z) + (EDWARDS_D * (p1.t * p2.t)),

    h = mul_256(f, tid);
    h = redc_256(h, tid);
    if (tid & 8u) == 8u {
        let base = select(out_y_base, out_x_base, t_ < 16u);
        output[base + t] = h;
    }
    switch l {
        // f = (p1.x + p1.y) * (p2.x + p2.y)
        case 1u: { f = scratch[tid - 8u]; }
        // f = (p1.z * p2.z) - (EDWARDS_D * (p1.t * p2.t))
        case 2u: { f = scratch[tid + 8u]; }
        default: {}
    }
    h = mul_256(f, tid);
    h = redc_256(h, tid);
    if (tid & 8u) == 8u {
        let base = select(out_z_base, out_t_base, t_ < 16u);
        output[base + t] = h;
    }
}
