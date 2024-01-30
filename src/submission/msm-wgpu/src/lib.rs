mod bytes;
pub mod gpu;
mod split;
mod utils;

use std::convert::TryInto;

use ark_ec::{CurveGroup, Group};
use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::Zero;
use bytemuck::{Pod, Zeroable};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::bytes::{read_points, write_fq, N_U32S_PER_POINT};
#[allow(unused_imports)]
use crate::split::{Split12, Split16, Split20, SplitterConstants};
use crate::utils::{time_begin, time_end};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketImplementation {
    Cpu,
    Gpu,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketSumImplementation {
    Cpu,
    Gpu,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Options {
    pub bucket_impl: BucketImplementation,
    pub bucket_sum_impl: BucketSumImplementation,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct PodPoint {
    pub x: [u32; 8],
    pub y: [u32; 8],
    pub t: [u32; 8],
    pub z: [u32; 8],
}

pub async fn compute_msm(points_flat: &[u32], scalars_flat: &[u32], options: Options) -> [u32; 16] {
    time_end("ffi");
    type Splitter = Split16;
    const N_BUCKETS: usize = 1 << Splitter::WINDOW_SIZE;
    let gpu_context = gpu::GpuDeviceQueue::new().await;

    // Sanity check
    debug_assert_eq!(points_flat.len() % N_U32S_PER_POINT, 0);
    let n_points = points_flat.len() / N_U32S_PER_POINT;
    debug_assert_eq!(scalars_flat.len() % 8, 0);
    debug_assert_eq!(scalars_flat.len() / 8, n_points);

    // let points = read_points(points_flat);

    time_begin("split");
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
    time_end("split");

    let mut buckets = Vec::with_capacity(Splitter::N_WINDOWS);
    match options.bucket_impl {
        BucketImplementation::Cpu => {
            time_begin("bucketing (cpu)");
            let points = read_points(points_flat);
            buckets = splitted
                .par_iter()
                .map(|splitted_scalars| bucket_cpu(splitted_scalars, &points, N_BUCKETS))
                .collect::<Vec<_>>();
            time_end("bucketing (cpu)");
        }
        BucketImplementation::Gpu => {
            time_begin("bucketing (gpu)");
            // let bucketer = gpu::bucket::GpuBucketer::new(&gpu_context, N_BUCKETS);
            let test_reducer = gpu::bucket::GpuIntraBucketReducer::new(&gpu_context, N_BUCKETS);
            let mut raw_buckets = Vec::with_capacity(Splitter::N_WINDOWS);
            for splitted_scalars in splitted.iter() {
                raw_buckets.push(
                    test_reducer
                        .reduce(splitted_scalars, bytemuck::cast_slice(points_flat))
                        .await,
                );
            }
            buckets = raw_buckets
                .par_iter()
                .map(|raw_buckets| read_points(raw_buckets))
                .collect::<Vec<_>>();
            // let mut preprocessed = Some(bucketer.preprocess(&splitted[0], &points));
            // for i in 0..Splitter::N_WINDOWS {
            //     let mut next_preprocessed = None;
            //     let interlude = {
            //         let next_preprocessed = &mut next_preprocessed;
            //         let points = &points;
            //         let bucketer = &bucketer;
            //         let splitted = &splitted;
            //         move || {
            //             if i + 1 < Splitter::N_WINDOWS {
            //                 *next_preprocessed =
            //                     Some(bucketer.preprocess(&splitted[i + 1], points));
            //             }
            //         }
            //     };
            //     let bucket = bucketer
            //         .bucket(preprocessed.take().unwrap(), interlude)
            //         .await;
            //     preprocessed = next_preprocessed;
            //     buckets.push(bucket);
            // }
            // A theoretically more parallelizable version of the above:
            // However this could lead to obscure errors down the line, so don't?
            // buckets.extend(
            //     futures::future::join_all(splitted.iter().map(|scalars| {
            //         let bucketer = gpu::bucket::GpuBucketer::new(&gpu_context);
            //         let points = &points;
            //         async move { bucketer.bucket(scalars, points, N_BUCKETS).await }
            //     }))
            //     .await,
            // );
            time_end("bucketing (gpu)");
        }
    }

    let mut bucket_sums;
    match options.bucket_sum_impl {
        BucketSumImplementation::Cpu => {
            time_begin("summing (cpu)");
            bucket_sums = buckets
                .into_par_iter()
                .map(bucket_sum_cpu)
                .collect::<Vec<_>>();
            time_end("summing (cpu)");
        }
        BucketSumImplementation::Gpu => {
            time_begin("summing (gpu)");
            bucket_sums = Vec::with_capacity(Splitter::N_WINDOWS);
            // let bucket_summer = gpu::bucket_sum::GpuBucketSummer::new(&gpu_context);
            // for bucket in buckets.into_iter() {
            //     bucket_sums.push(bucket_summer.bucket_sum(&bucket).await);
            // }
            bucket_sums.extend(
                futures::future::join_all(buckets.into_iter().map(|bucket| {
                    let bucket_summer = gpu::bucket_sum::GpuBucketSummer::new(&gpu_context);
                    async move { bucket_summer.bucket_sum(&bucket).await }
                }))
                .await,
            );
            time_end("summing (gpu)");
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
    result_buf
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

#[cfg(target_arch = "wasm32")]
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn inter_bucket_reduce_16(raw_buckets: Vec<wasm_bindgen::JsValue>) -> Vec<u32> {
    type Splitter = Split16;

    let buckets = raw_buckets
        .into_iter()
        .map(|raw_bucket| {
            let bucket: js_sys::Uint32Array = raw_bucket.into();
            read_points(&bucket.to_vec())
        })
        .collect::<Vec<_>>();
    let bucket_sums = buckets
        .into_par_iter()
        .map(bucket_sum_cpu)
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

// WASM bindings

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
static INIT: std::sync::Once = std::sync::Once::new();

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn compute_msm_js(
    points_flat: &[u32],
    scalars_flat: &[u32],
    options: JsValue,
) -> js_sys::Uint32Array {
    INIT.call_once(|| {
        console_log::init_with_level(log::Level::Info).unwrap();
    });
    crate::utils::set_panic_hook();
    let options: Options = serde_wasm_bindgen::from_value(options).expect("invalid options");
    let result = compute_msm(points_flat, scalars_flat, options).await;
    js_sys::Uint32Array::from(&result[..])
}
