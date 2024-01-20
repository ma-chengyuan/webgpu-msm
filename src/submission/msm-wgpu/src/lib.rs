mod bytes;
mod gpu;
mod split;
mod utils;

use std::convert::TryInto;

use ark_ec::{CurveGroup, Group};
use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::Zero;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::console;
use web_sys::js_sys::Uint32Array;

use crate::bytes::{read_points, write_fq, N_U32S_PER_POINT};
use crate::split::{Split16, SplitterConstants};
use crate::utils::set_panic_hook;

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BucketImplementation {
    Cpu,
    Gpu,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BucketSumImplementation {
    Cpu,
    Gpu,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]

struct Options {
    bucket_impl: BucketImplementation,
    bucket_sum_impl: BucketSumImplementation,
}

#[wasm_bindgen]
#[allow(clippy::never_loop)]
pub async fn compute_msm(
    points_flat: &[u32],
    scalars_flat: &[u32],
    options: JsValue,
) -> Uint32Array {
    type Splitter = Split16;

    set_panic_hook();
    console_log::init_with_level(log::Level::Info).unwrap();

    let options: Options = serde_wasm_bindgen::from_value(options).expect("invalid options");

    const N_BUCKETS: usize = 1 << Splitter::WINDOW_SIZE;
    let gpu_context = gpu::GpuDeviceQueue::new().await;

    // Sanity check
    debug_assert_eq!(points_flat.len() % N_U32S_PER_POINT, 0);
    let n_points = points_flat.len() / N_U32S_PER_POINT;
    debug_assert_eq!(scalars_flat.len() % 8, 0);
    debug_assert_eq!(scalars_flat.len() / 8, n_points);

    let points = read_points(points_flat);

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

    let mut buckets = Vec::with_capacity(Splitter::N_WINDOWS);
    match options.bucket_impl {
        BucketImplementation::Cpu => {
            console::time_with_label("bucketing (cpu)");
            for splitted_scalars in splitted.iter() {
                let bucket = bucket_cpu(splitted_scalars, &points, N_BUCKETS);
                buckets.push(bucket);
            }
            console::time_end_with_label("bucketing (cpu)");
        }
        BucketImplementation::Gpu => {
            console::time_with_label("bucketing (gpu)");
            let bucketer = gpu::bucket::GpuBucketer::new(&gpu_context);
            for splitted_scalars in splitted.iter() {
                let bucket = bucketer.bucket(splitted_scalars, &points, N_BUCKETS).await;
                buckets.push(bucket);
            }
            console::time_end_with_label("bucketing (gpu)");
        }
    }

    let mut bucket_sums;
    match options.bucket_sum_impl {
        BucketSumImplementation::Cpu => {
            console::time_with_label("summing (cpu)");
            bucket_sums = buckets.into_iter().map(bucket_sum_cpu).collect::<Vec<_>>();
            console::time_end_with_label("summing (cpu)");
        }
        BucketSumImplementation::Gpu => {
            console::time_with_label("summing (gpu)");
            let bucket_summer = gpu::bucket_sum::GpuBucketSummer::new(&gpu_context);
            bucket_sums = Vec::with_capacity(Splitter::N_WINDOWS);
            for bucket in buckets.into_iter() {
                bucket_sums.push(bucket_summer.bucket_sum(&bucket).await);
            }
            console::time_end_with_label("summing (gpu)");
        }
    }

    let mut sum = EdwardsProjective::zero();
    for bucket_sum in bucket_sums {
        for _ in 0..Splitter::WINDOW_SIZE {
            sum.double_in_place();
        }
        sum += bucket_sum;
    }

    let result = sum;
    let result_affine = result.into_affine();
    let mut result_buf = [0u32; 16];
    write_fq(&mut result_buf[0..8], &result_affine.x);
    write_fq(&mut result_buf[8..16], &result_affine.y);
    Uint32Array::from(&result_buf[..])
}

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
