@group(0) @binding(0)
var<storage, read> input1: array<u32>;
@group(0) @binding(1)
var<storage, read> input2: array<u32>;
@group(0) @binding(2)
var<storage, read_write> output: array<u32>;

const WG_SIZE: u32 = 32;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    if tid == 0u {
        c1 = R_SQUARED;
    }
    let gid = group_id.x * (WG_SIZE / 16u) + tid / 16u;
    let t = tid & 15u;
    var x: u32;
    if t < 8u {
        x = input1[gid * 8u + t];
    } else {
        var r_squared = R_SQUARED;
        x = r_squared[t - 8u];
    }
    var m = mul_256(x, tid);
    m = redc_256(m, tid);
    if t >= 8u {
        output[gid * 8u + t - 8u] = m;
    }
    // output[gid] = add_256(input1[gid], input2[gid], tid);
    // output[gid] = sub_256(input1[gid], input2[gid], tid);
    // output[gid] = cas_256(input1[gid], input2[gid], tid);
    // output[gid] = bitcast<u32>(cmp_256(input1[gid], input2[gid], tid));
    // output[gid] = mul_256(input1[gid], tid);
    // output[gid] = redc_256(input1[gid], tid);
    // output[gid] = field_mul(input1[gid], tid);
}

var<workgroup> carry: array<i32, WG_SIZE>;

// 8 threads cooperate to add 2 256-bit numbers
fn add_256(a: u32, b: u32, tid: u32) -> u32 {
    var sum = a + b;
    let t = tid & 7u;
    // c: 0 for no carry, 1 for generate carry, -1 for propagate carry
    var c: i32 = select(select(0, -1, sum == 0xffffffff), 1, sum < a);
    // Perform reduction below: carry[t] = max(carry[t], carry[k]) & carry[t] for all k < t
    carry[tid] = c;
    workgroupBarrier();
    if t >= 1u { c = max(c, carry[tid - 1]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 2u { c = max(c, carry[tid - 2]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 4u { c = max(c, carry[tid - 4]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 1u { sum += u32(max(carry[tid - 1], 0)); }
    return sum;
}

// Similar to add_256, but for 512-bit numbers.
// To be used in Montgomery multiplication.
fn add_512(a: u32, b: u32, tid: u32) -> u32 {
    var sum = a + b;
    let t = tid & 15u;
    // c: 0 for no carry, 1 for generate carry, -1 for propagate carry
    var c: i32 = select(select(0, -1, sum == 0xffffffff), 1, sum < a);
    // Perform reduction below: carry[t] = max(carry[t], carry[k]) & carry[t] for all k < t
    carry[tid] = c;
    workgroupBarrier();
    if t >= 1u { c = max(c, carry[tid - 1]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 2u { c = max(c, carry[tid - 2]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 4u { c = max(c, carry[tid - 4]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 8u { c = max(c, carry[tid - 8]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 1u { sum += u32(max(carry[tid - 1], 0)); }
    return sum;
}

// 8 threads cooperate to subtract 2 256-bit numbers
fn sub_256(a: u32, b: u32, tid: u32) -> u32 {
    var sub = a - b;
    let t = tid & 7u;
    // c: 0 for no borrow, 1 for generate borrow, -1 for propagate borrow
    var c = select(select(0, -1, sub == 0x00000000), 1, sub > a);
    carry[tid] = c;
    workgroupBarrier();
    if t >= 1u { c = max(c, carry[tid - 1]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 2u { c = max(c, carry[tid - 2]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 4u { c = max(c, carry[tid - 4]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 1u { sub -= u32(max(carry[tid - 1], 0)); }
    return sub;
}

// compare-and-subtract: evaluated a >= b ? a - b : a
fn cas_256(a: u32, b: u32, tid: u32) -> u32 {
    var sub = a - b;
    let t = tid & 7u;
    // c: 0 for no borrow, 1 for generate borrow, -1 for propagate borrow
    var c = select(select(0, -1, sub == 0x00000000), 1, sub > a);
    carry[tid] = c;
    workgroupBarrier();
    if t >= 1u { c = max(c, carry[tid - 1]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 2u { c = max(c, carry[tid - 2]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 4u { c = max(c, carry[tid - 4]) & c; carry[tid] = c; }
    workgroupBarrier();
    if t >= 1u { sub -= u32(max(carry[tid - 1], 0)); }
    let top_borrow = carry[(tid & 0xF8u) + 7u];
    // Need to borrow from more significant limbs: a < b, so return a
    return select(sub, a, top_borrow == 1);
}

// 8 threads cooperate to compare 2 256-bit numbers
fn cmp_256(a: u32, b: u32, tid: u32) -> i32 {
    let t = tid & 7u;
    // sign of a - b
    var s: i32 = select(select(0, -1, a < b), 1, a > b);
    carry[tid] = s;
    workgroupBarrier();
    if t <= 6u { var s_ = carry[tid + 1u]; s = select(s_, s, s_ == 0); carry[tid] = s; }
    workgroupBarrier();
    if t <= 5u { var s_ = carry[tid + 2u]; s = select(s_, s, s_ == 0); carry[tid] = s; }
    workgroupBarrier();
    if t <= 3u { var s_ = carry[tid + 4u]; s = select(s_, s, s_ == 0); carry[tid] = s; }
    workgroupBarrier();
    return carry[tid & 0xF8u];
}

// TODO: merge and recycle arrays
var<workgroup> mul_limbs: array<u32, WG_SIZE>;
const WG_SIZE_DOUBLED: u32 = WG_SIZE * 2u;
// Accumulator for multiplication result
var<workgroup> mul_acc: array<u32, WG_SIZE_DOUBLED>;
// Carry propagation for multiplication result
var<workgroup> mul_g: array<u32, WG_SIZE_DOUBLED>;
var<workgroup> mul_p: array<u32, WG_SIZE_DOUBLED>;

// 16 threads cooperate to multiply 2 256-bit numbers
// x: threads 1-8 hold the first 8 limbs of the first number, threads 9-16 hold
// the first 8 limbs of the second number
fn mul_256(x: u32, tid: u32) -> u32 {
    let t = tid & 15u; // Thread id with 16 threads
    let t_ = tid & 0xF0u;
    let base = 2u * t_; // Base address of the accumulator

    // There are cases where we want to operate on 32 stuffs at once, but we only have 16 threads
    // Assign two indices to each thread
    let i1 = base + t;
    let i2 = i1 + 16u;

    mul_limbs[tid] = x; // Share limbs
    mul_acc[i1] = 0u; // Clear accumulators
    mul_acc[i2] = 0u;

    workgroupBarrier();
    // Each thread pick up a 16-bit limb from the first number
    let tmp = mul_limbs[t_ + (t >> 1u)];
    let a = select(tmp & 0xFFFF, tmp >> 16u, (t & 1u) == 1u);

    for (var i = 0u; i < 16u; i++) {
        let tmp = mul_limbs[t_ + (i >> 1u) + 8u];
        let b = select(tmp & 0xFFFF, tmp >> 16u, (i & 1u) == 1u);
        let prod = a * b;
        mul_acc[i1 + i] += prod & 0xFFFF;
        workgroupBarrier();
        mul_acc[i1 + i + 1u] += prod >> 16u;
        workgroupBarrier();
    }
    // Time for some carry propagation
    mul_g[i1] = mul_acc[i1] >> 16u;
    mul_p[i1] = 0xFFFF - (mul_acc[i1] & 0xFFFF);
    mul_g[i2] = mul_acc[i2] >> 16u;
    mul_p[i2] = 0xFFFF - (mul_acc[i2] & 0xFFFF);
    workgroupBarrier();

    for (var s = 1u; s <= 16u; s *= 2u) {
        var p1 = mul_p[i2];
        var g2 = mul_g[i2 - s];
        var p2 = mul_p[i2 - s];
        workgroupBarrier();
        mul_g[i2] += u32(g2 > p1);
        mul_p[i2] = select(0xFFFFu, p2, g2 == p1);
        if t >= s {
            p1 = mul_p[i1];
            g2 = mul_g[i1 - s];
            p2 = mul_p[i1 - s];
        }
        workgroupBarrier();
        if t >= s {
            mul_g[i1] += u32(g2 > p1);
            mul_p[i1] = select(0xFFFFu, p2, g2 == p1);
        }
    }

    var lo = mul_acc[base + 2 * t];
    if t >= 1u { lo += mul_g[base + 2 * t - 1u]; }
    let hi = mul_acc[base + 2 * t + 1u] + mul_g[base + 2 * t];
    return (lo & 0xFFFF) | (hi << 16u);
}

// Modulo
const N: array<u32, 16> = array<u32, 16>(0, 0, 0, 0, 0, 0, 0, 0, 1, 168919040, 3489660929, 1504343806, 1547153409, 1622428958, 2586617174, 313222494);
// For use in Montgomery multiplication
const N_PRIME: array<u32, 16> = array<u32, 16>(0, 0, 0, 0, 0, 0, 0, 0, 4294967295, 168919039, 2415919105, 1159862220, 1200660480, 613901763, 1756534102, 1771229434);
// Montgomery R
const R: array<u32, 8> = array<u32, 8>(4294967283, 2099019775, 1879048178, 1918366991, 1361842158, 383260021, 733715101, 223074866);
// Montgomery R^2 mod N, used to convert from normal to Montgomery form
const R_SQUARED: array<u32, 8> = array<u32, 8>(3093398907, 634746810, 2288015647, 3425445813, 3856434579, 2815164559, 4025600313, 18864871);

// Montgomery reduction 
fn redc_256(x: u32, tid: u32) -> u32 {
    let t = tid & 15u; // Thread id with 16 threads
    var n_prime = N_PRIME;
    var n = N;
    var m = mul_256(select(x, n_prime[t], t >= 8u), tid); // (x mod R)N′
    m = mul_256(select(m, n[t], t >= 8u), tid); // (m mod R)N
    m = add_512(m, x, tid); // (m + x)
    if t >= 8u { m = cas_256(m, n[t], tid); } else { m = 0u; }
    return m;
}

fn field_mul(x: u32, tid: u32) -> u32 {
    var m = mul_256(x, tid);
    return redc_256(m, tid);
}