//! Handles serialization of EC points to and from bytes.

use ark_ed_on_bls12_377::{EdwardsProjective, Fq};
use ark_ff::{BigInt, PrimeField};

// 4 components per point, 8 u32s per component
pub const N_U32S_PER_POINT: usize = 4 * 8;
pub const N_BYTES_PER_POINT: usize = N_U32S_PER_POINT * 4;

pub fn read_fq(buf: &[u32]) -> Fq {
    debug_assert_eq!(buf.len(), 8);
    let bigint = BigInt([
        ((buf[6] as u64) << 32) + buf[7] as u64,
        ((buf[4] as u64) << 32) + buf[5] as u64,
        ((buf[2] as u64) << 32) + buf[3] as u64,
        ((buf[0] as u64) << 32) + buf[1] as u64,
    ]);
    Fq::from_bigint(bigint).unwrap()
}

pub fn write_fq(buf: &mut [u32], fq: &Fq) {
    debug_assert_eq!(buf.len(), 8);
    let bigint = fq.into_bigint();
    buf[7] = (bigint.0[0] & 0xffffffff) as u32;
    buf[6] = (bigint.0[0] >> 32) as u32;
    buf[5] = (bigint.0[1] & 0xffffffff) as u32;
    buf[4] = (bigint.0[1] >> 32) as u32;
    buf[3] = (bigint.0[2] & 0xffffffff) as u32;
    buf[2] = (bigint.0[2] >> 32) as u32;
    buf[1] = (bigint.0[3] & 0xffffffff) as u32;
    buf[0] = (bigint.0[3] >> 32) as u32;
}

pub fn read_points(points_flat: &[u32]) -> Vec<EdwardsProjective> {
    let n_points = points_flat.len() / N_U32S_PER_POINT;
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = read_fq(&points_flat[32 * i..32 * i + 8]);
        let y = read_fq(&points_flat[32 * i + 8..32 * i + 16]);
        let t = read_fq(&points_flat[32 * i + 16..32 * i + 24]);
        let z = read_fq(&points_flat[32 * i + 24..32 * i + 32]);
        let point = EdwardsProjective::new_unchecked(x, y, t, z);
        points.push(point);
    }
    points
}

pub fn write_points(points: &[EdwardsProjective]) -> Vec<u32> {
    let mut points_flat = vec![0u32; N_U32S_PER_POINT * points.len()];
    for (i, point) in points.iter().enumerate() {
        write_fq(&mut points_flat[32 * i..32 * i + 8], &point.x);
        write_fq(&mut points_flat[32 * i + 8..32 * i + 16], &point.y);
        write_fq(&mut points_flat[32 * i + 16..32 * i + 24], &point.t);
        write_fq(&mut points_flat[32 * i + 24..32 * i + 32], &point.z);
    }
    points_flat
}
