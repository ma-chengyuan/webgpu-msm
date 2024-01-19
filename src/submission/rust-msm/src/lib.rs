mod utils;

use ark_ec::{CurveGroup, Group};
use ark_ed_on_bls12_377::{EdwardsProjective, Fq};
use ark_ff::{BigInt, PrimeField, Zero};
use wasm_bindgen::prelude::*;

use crate::utils::set_panic_hook;

fn read_fq(buf: &[u32]) -> Fq {
    debug_assert_eq!(buf.len(), 8);
    let bigint = BigInt([
        ((buf[6] as u64) << 32) + buf[7] as u64,
        ((buf[4] as u64) << 32) + buf[5] as u64,
        ((buf[2] as u64) << 32) + buf[3] as u64,
        ((buf[0] as u64) << 32) + buf[1] as u64,
    ]);
    Fq::from_bigint(bigint).unwrap()
}

fn write_fq(buf: &mut [u32], fq: &Fq) {
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

#[allow(clippy::all)]
#[wasm_bindgen]
pub fn compute_msm(points_flat: &[u32], scalars_flat: &[u32]) -> Vec<u32> {
    set_panic_hook();

    // 8 u32s for a coordinate component; 4 components per point
    debug_assert_eq!(points_flat.len() % 32, 0);
    let n_points = points_flat.len() / 32;
    debug_assert_eq!(scalars_flat.len() % 8, 0);
    debug_assert_eq!(scalars_flat.len() / 8, n_points);

    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = read_fq(&points_flat[32 * i..32 * i + 8]);
        let y = read_fq(&points_flat[32 * i + 8..32 * i + 16]);
        let t = read_fq(&points_flat[32 * i + 16..32 * i + 24]);
        let z = read_fq(&points_flat[32 * i + 24..32 * i + 32]);
        let point = EdwardsProjective::new_unchecked(x, y, t, z);
        points.push(point);
    }

    let mut scalars = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let scalar = read_fq(&scalars_flat[8 * i..8 * i + 8]);
        scalars.push(scalar);
    }

    let result = compute_msm_impl(&points, &scalars);
    let result_affine = result.into_affine();
    let mut result_buf = vec![0u32; 16];
    write_fq(&mut result_buf[0..8], &result_affine.x);
    write_fq(&mut result_buf[8..16], &result_affine.y);
    result_buf
}

#[allow(clippy::all)]
fn compute_msm_impl(points: &[EdwardsProjective], scalars: &[Fq]) -> EdwardsProjective {
    let mut acc = EdwardsProjective::zero();
    for (point, scalar) in points.iter().zip(scalars.iter()) {
        acc += point.mul_bigint(scalar.into_bigint());
    }
    acc
}
