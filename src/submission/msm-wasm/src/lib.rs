#![allow(dead_code)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod bytes;
mod split;
mod utils;

use std::convert::TryInto;

use ark_ec::{CurveGroup, Group};
use ark_ed_on_bls12_377::{EdwardsAffine, EdwardsProjective};
use ark_ff::Zero;
use paste::paste;
use rayon::iter::ParallelIterator;

use crate::bytes::{read_fq, read_points, write_fq};
#[allow(unused_imports)]
use crate::split::*;
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

pub fn split<Split>(scalars_flat: &[u32]) -> Vec<u32>
where
    Split: SplitImpl,
    Split::Output: Into<u32> + Copy,
    [(); Split::N_WINDOWS]: Sized,
{
    INIT.call_once(|| {
        console_log::init_with_level(log::Level::Info).unwrap();
    });
    crate::utils::set_panic_hook();

    debug_assert_eq!(scalars_flat.len() % 8, 0);
    let n_points = scalars_flat.len() / 8;
    let mut result = vec![0u32; n_points * Split::N_WINDOWS];
    for i in 0..n_points {
        let slice: &[u32; 8] = unsafe {
            (&scalars_flat[8 * i..8 * i + 8])
                .try_into()
                .unwrap_unchecked()
        };
        let splitted_scalars = Split::split(slice);
        for (j, splitted_scalar) in splitted_scalars.iter().enumerate() {
            result[j * n_points + i] = (*splitted_scalar).into();
        }
    }
    result
}

use rayon::prelude::ParallelSlice;

fn reduce_last<Split>(bucket_sums: Vec<EdwardsProjective>) -> Vec<u32>
where
    Split: SplitImpl,
{
    let mut sum = EdwardsProjective::zero();
    for bucket_sum in bucket_sums {
        for _ in 0..Split::WINDOW_SIZE {
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

fn msm_end_to_end<Split>(scalars_flat: &[u32], points_flat: &[u32]) -> Vec<u32>
where
    Split: SplitImpl,
    Split::Output: Into<u32> + Copy,
    [(); Split::N_WINDOWS]: Sized,
{
    let split = split::<Split>(scalars_flat);
    let chunk_size = split.len() / Split::N_WINDOWS;
    let points = read_points(points_flat);
    let n_buckets = 1 << Split::WINDOW_SIZE;
    let bucket_sums = split
        .par_chunks(chunk_size)
        .map(|chunk| bucket_sum_cpu(bucket_cpu(chunk, &points, n_buckets)))
        .collect::<Vec<_>>();
    reduce_last::<Split>(bucket_sums)
}

fn inter_bucket_reduce<Split>(raw_buckets: &[u32]) -> Vec<u32>
where
    Split: SplitImpl,
{
    let chunk_size = raw_buckets.len() / Split::N_WINDOWS;
    let bucket_sums = raw_buckets
        .par_chunks(chunk_size)
        .map(|chunk| bucket_sum_cpu(read_points(chunk)))
        .collect::<Vec<_>>();
    reduce_last::<Split>(bucket_sums)
}

fn inter_bucket_reduce_last<Split>(raw_buckets: &[u32]) -> Vec<u32>
where
    Split: SplitImpl,
{
    reduce_last::<Split>(read_points(raw_buckets))
}

macro_rules! define_msm_functions {
    ($($w:expr),*) => {
        $(paste! {
            #[wasm_bindgen]
            pub fn [<split_ $w>](scalars_flat: &[u32]) -> Vec<u32> {
                split::<[<Split $w>]>(scalars_flat)
            }

            #[wasm_bindgen]
            pub fn [<msm_end_to_end_ $w>](scalars_flat: &[u32], points_flat: &[u32]) -> Vec<u32> {
                msm_end_to_end::<[<Split $w>]>(scalars_flat, points_flat)
            }

            #[wasm_bindgen]
            pub fn [<inter_bucket_reduce_ $w>](raw_buckets: &[u32]) -> Vec<u32> {
                inter_bucket_reduce::<[<Split $w>]>(raw_buckets)
            }

            #[wasm_bindgen]
            pub fn [<inter_bucket_reduce_last_ $w>](raw_buckets: &[u32]) -> Vec<u32> {
                inter_bucket_reduce_last::<[<Split $w>]>(raw_buckets)
            }
        })*

        paste!{
            #[wasm_bindgen]
            pub fn split_dynamic(window_size: u32, scalars_flat: &[u32]) -> Vec<u32> {
                match window_size {
                    $( $w => split::<[<Split $w>]>(scalars_flat), )*
                    _ => panic!("Unsupported window size: {}", window_size),
                }
            }

            #[wasm_bindgen]
            pub fn msm_end_to_end_dynamic(window_size: u32, scalars_flat: &[u32], points_flat: &[u32]) -> Vec<u32> {
                match window_size {
                    $( $w => msm_end_to_end::<[<Split $w>]>(scalars_flat, points_flat), )*
                    _ => panic!("Unsupported window size: {}", window_size),
                }
            }

            #[wasm_bindgen]
            pub fn inter_bucket_reduce_dynamic(window_size: u32, raw_buckets: &[u32]) -> Vec<u32> {
                match window_size {
                    $( $w => inter_bucket_reduce::<[<Split $w>]>(raw_buckets), )*
                    _ => panic!("Unsupported window size: {}", window_size),
                }
            }

            #[wasm_bindgen]
            pub fn inter_bucket_reduce_last_dynamic(window_size: u32, raw_buckets: &[u32]) -> Vec<u32> {
                match window_size {
                    $( $w => inter_bucket_reduce_last::<[<Split $w>]>(raw_buckets), )*
                    _ => panic!("Unsupported window size: {}", window_size),
                }
            }
        }
    };
}

define_msm_functions!(8, 9, 10, 11, 12, 13, 14, 15, 16, 20);

#[wasm_bindgen]
pub fn point_add_affine(a: &[u32], b: &[u32]) -> Vec<u32> {
    assert_eq!(a.len(), 16);
    assert_eq!(b.len(), 16);
    let a = EdwardsAffine::new_unchecked(read_fq(&a[0..8]), read_fq(&a[8..16]));
    let b = EdwardsAffine::new_unchecked(read_fq(&b[0..8]), read_fq(&b[8..16]));
    let c = (a + b).into_affine();
    let mut result_buf = vec![0u32; 16];
    write_fq(&mut result_buf[0..8], &c.x);
    write_fq(&mut result_buf[8..16], &c.y);
    result_buf
}

// WASM bindings

pub use wasm_bindgen_rayon::init_thread_pool;
