mod split;
mod utils;

use std::convert::TryInto;

use ark_ec::{CurveGroup, Group};
use ark_ed_on_bls12_377::{EdwardsProjective, Fq};
use ark_ff::{BigInt, PrimeField, Zero};
use wasm_bindgen::prelude::*;

use crate::split::{Split16, SplitterConstants};
use crate::utils::set_panic_hook;

type Splitter = Split16;

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

#[wasm_bindgen]
pub fn compute_msm(points_flat: &[u32], scalars_flat: &[u32]) -> Vec<u32> {
    set_panic_hook();

    use web_sys::console;
    console::log_1(&"Computing MSM".into());

    // Sanity check
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

    console::time_with_label("split");
    let mut splitted: Vec<Vec<u32>> = (0..Splitter::N_WINDOWS)
        .map(|_| Vec::with_capacity(n_points))
        .collect();
    for i in 0..n_points {
        let slice: &[u32; 8] = unsafe {
            (&scalars_flat[8 * i..8 * i + 8])
                .try_into()
                .unwrap_unchecked()
        };
        let splitted_scalars = Splitter::split(slice);
        for (j, splitted_scalar) in splitted_scalars.iter().enumerate() {
            splitted[j].push(*splitted_scalar);
        }
    }
    console::time_end_with_label("split");

    const N_BUCKETS: usize = 1 << Splitter::WINDOW_SIZE;

    console::time_with_label("bucket");
    let mut buckets = vec![vec![Option::<EdwardsProjective>::None; N_BUCKETS]; Splitter::N_WINDOWS];
    for (i, splitted_scalars) in splitted.iter().enumerate() {
        for (split_scalar, point) in splitted_scalars.iter().zip(points.iter()) {
            let bucket = (*split_scalar) as usize;
            if bucket == 0 {
                continue;
            }
            assert!(bucket < N_BUCKETS);
            buckets[i][bucket] = Some(match buckets[i][bucket] {
                None => *point,
                Some(existing) => existing + point,
            })
        }
    }
    console::time_end_with_label("bucket");
    let bucket_sums = buckets.into_iter().map(|bucket| {
        let mut sum = EdwardsProjective::zero();
        let mut carry = EdwardsProjective::zero();
        for i in (1..N_BUCKETS).rev() {
            if let Some(point) = bucket[i] {
                carry += point;
            }
            sum += carry;
        }
        sum
    });
    let mut sum = EdwardsProjective::zero();
    for bucket_sum in bucket_sums {
        for _ in 0..Splitter::WINDOW_SIZE {
            sum.double_in_place();
        }
        sum += bucket_sum;
    }

    let result = sum;
    let result_affine = result.into_affine();
    let mut result_buf = vec![0u32; 16];
    write_fq(&mut result_buf[0..8], &result_affine.x);
    write_fq(&mut result_buf[8..16], &result_affine.y);
    result_buf
}
