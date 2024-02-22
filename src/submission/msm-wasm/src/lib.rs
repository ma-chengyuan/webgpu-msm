#![allow(dead_code)]

mod bytes;
mod split;
mod utils;

use std::convert::TryInto;

use ark_ec::{CurveGroup, Group};
use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::Zero;
use rayon::iter::ParallelIterator;

use crate::bytes::{read_points, write_fq};
#[allow(unused_imports)]
use crate::split::{Split12, Split16, Split20, SplitterConstants};
use wasm_bindgen::prelude::*;

static INIT: std::sync::Once = std::sync::Once::new();

fn bucket_cpu(
    scalars: &[u32],
    points: &[EdwardsProjective],
    n_buckets: usize,
) -> Vec<EdwardsProjective> {
    let mut bucket = vec![EdwardsProjective::zero(); n_buckets];
    for (scalar, point) in scalars.iter().zip(points.iter()) {
        let bucket_id = (*scalar) as usize;
        if bucket_id == 0 {
            continue;
        }
        assert!(bucket_id < n_buckets);
        let existing = &bucket[bucket_id];
        bucket[bucket_id] = if existing.is_zero() {
            *point
        } else {
            existing + point
        };
    }
    bucket
}

fn bucket_sum_cpu(bucket: Vec<EdwardsProjective>) -> EdwardsProjective {
    let mut sum = EdwardsProjective::zero();
    let mut carry = EdwardsProjective::zero();
    for i in (1..bucket.len()).rev() {
        if !bucket[i].is_zero() {
            carry += bucket[i];
        }
        sum += carry;
    }
    sum
}

#[wasm_bindgen]
pub fn split_16(scalars_flat: &[u32]) -> Vec<u32> {
    type Splitter = Split16;

    INIT.call_once(|| {
        console_log::init_with_level(log::Level::Info).unwrap();
    });
    crate::utils::set_panic_hook();

    debug_assert_eq!(scalars_flat.len() % 8, 0);
    let n_points = scalars_flat.len() / 8;
    let mut result = vec![0u32; n_points * Splitter::N_WINDOWS];
    for i in 0..n_points {
        let slice: &[u32; 8] = unsafe {
            (&scalars_flat[8 * i..8 * i + 8])
                .try_into()
                .unwrap_unchecked()
        };
        let splitted_scalars = Splitter::split(slice);
        for (j, splitted_scalar) in splitted_scalars.iter().enumerate() {
            result[j * n_points + i] = *splitted_scalar;
        }
    }
    result
}

use rayon::prelude::ParallelSlice;

#[wasm_bindgen]
pub fn inter_bucket_reduce_16(raw_buckets: &[u32]) -> Vec<u32> {
    type Splitter = Split16;
    let chunk_size = raw_buckets.len() / Splitter::N_WINDOWS;
    let bucket_sums = raw_buckets
        .par_chunks(chunk_size)
        .map(|chunk| bucket_sum_cpu(read_points(chunk)))
        .collect::<Vec<_>>();
    let mut sum = EdwardsProjective::zero();
    for bucket_sum in bucket_sums {
        for _ in 0..Splitter::WINDOW_SIZE {
            sum.double_in_place();
        }
        sum += bucket_sum;
    }
    let result_affine = sum.into_affine();
    let mut result_buf = vec![0u32; 16];
    write_fq(&mut result_buf[0..8], &result_affine.x);
    write_fq(&mut result_buf[8..16], &result_affine.y);
    result_buf
}

#[wasm_bindgen]
pub fn inter_bucket_reduce_last(raw_buckets: &[u32]) -> Vec<u32> {
    type Splitter = Split16;
    let bucket_sums = read_points(raw_buckets);
    let mut sum = EdwardsProjective::zero();
    for bucket_sum in bucket_sums {
        for _ in 0..Splitter::WINDOW_SIZE {
            sum.double_in_place();
        }
        sum += bucket_sum;
    }
    let result_affine = sum.into_affine();
    let mut result_buf = vec![0u32; 16];
    write_fq(&mut result_buf[0..8], &result_affine.x);
    write_fq(&mut result_buf[8..16], &result_affine.y);
    result_buf
}

// WASM bindings

pub use wasm_bindgen_rayon::init_thread_pool;
