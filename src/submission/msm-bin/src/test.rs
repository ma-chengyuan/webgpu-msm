#![allow(dead_code)]

use std::str::FromStr;

use anyhow::{bail, Result};
use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::PrimeField;
use itertools::Itertools;
use rand::{Rng, SeedableRng};

use crate::{gpu::*, load_arc_points};
use num_bigint::{BigUint, RandomBits};

const SHADER_CODE: &str = include_str!("../../msm-wgpu/src/gpu/wgsl/test.wgsl");

fn write_u256(vec: &mut Vec<u32>, bignum: &BigUint) {
    vec.extend(bignum.iter_u32_digits().take(8).pad_using(8, |_| 0u32));
}

fn write_u256s(bignums: &[BigUint]) -> Vec<u32> {
    let mut vec = Vec::with_capacity(bignums.len() * 8);
    for bignum in bignums {
        write_u256(&mut vec, bignum);
    }
    vec
}

fn write_u512(vec: &mut Vec<u32>, bignum: &BigUint) {
    vec.extend(bignum.iter_u32_digits().take(16).pad_using(16, |_| 0u32));
}

fn write_u512s(bignums: &[BigUint]) -> Vec<u32> {
    let mut vec = Vec::with_capacity(bignums.len() * 16);
    for bignum in bignums {
        write_u512(&mut vec, bignum);
    }
    vec
}

fn print_u32s(u32s: &[u32]) {
    println!("{}", u32s.iter().map(|&x| format!("{:08x}", x)).join(" "));
}

async fn test_add() -> Result<()> {
    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    log::info!("Acquired device");
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );
    log::info!("Created pipeline");

    let n_numbers = (32768 * 4) as usize;
    let n_bytes = (n_numbers * 32) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 4) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, n_bytes);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand::thread_rng();

    let mut input_1 = vec![BigUint::default(); n_numbers];
    let mut input_2 = vec![BigUint::default(); n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            input_1[i] = rng.sample(RandomBits::new(256));
            input_2[i] = rng.sample(RandomBits::new(256));
            expected_output[i] = &input_1[i] + &input_2[i];
        }

        let input_1_le = write_u256s(&input_1);
        let input_2_le = write_u256s(&input_2);
        let expected_output_le = write_u256s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_1_le));
        queue.write_buffer(&input_2_buffer, 0, bytemuck::cast_slice(&input_2_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 8..(i + 1) * 8];
            let actual = &actual_output[i * 8..(i + 1) * 8];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_1_le[i * 8..(i + 1) * 8]);
                println!("Input 2:");
                print_u32s(&input_2_le[i * 8..(i + 1) * 8]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

async fn test_sub() -> Result<()> {
    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    log::info!("Acquired device");
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );
    log::info!("Created pipeline");

    let n_numbers = (32768 * 4) as usize;
    let n_bytes = (n_numbers * 32) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 4) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, n_bytes);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand::thread_rng();

    let mut input_1 = vec![BigUint::default(); n_numbers];
    let mut input_2 = vec![BigUint::default(); n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            let (a, b) = gen_cmp_u256(&mut rng);
            input_1[i] = a;
            input_2[i] = b;
            // input_1[i] = rng.sample(RandomBits::new(256));
            // input_2[i] = rng.sample(RandomBits::new(256));
            if input_1[i] < input_2[i] {
                std::mem::swap(&mut input_1[i], &mut input_2[i]);
            }
            expected_output[i] = &input_1[i] - &input_2[i];
        }

        let input_1_le = write_u256s(&input_1);
        let input_2_le = write_u256s(&input_2);
        let expected_output_le = write_u256s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_1_le));
        queue.write_buffer(&input_2_buffer, 0, bytemuck::cast_slice(&input_2_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 8..(i + 1) * 8];
            let actual = &actual_output[i * 8..(i + 1) * 8];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_1_le[i * 8..(i + 1) * 8]);
                println!("Input 2:");
                print_u32s(&input_2_le[i * 8..(i + 1) * 8]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

fn gen_cmp_u256(rng: &mut impl Rng) -> (BigUint, BigUint) {
    let dist = rand::distributions::Uniform::new_inclusive(-1, 1);
    let mut a_raw = [0u32; 8];
    let mut b_raw = [0u32; 8];
    for i in 0..8 {
        let mut x = rng.gen::<u32>();
        let mut y = rng.gen::<u32>();
        if x < y {
            std::mem::swap(&mut x, &mut y);
        }
        match rng.sample(dist) {
            1 => {
                // a > b
                a_raw[i] = x;
                b_raw[i] = y;
            }
            -1 => {
                // a < b
                a_raw[i] = y;
                b_raw[i] = x;
            }
            0 => {
                // a == b
                a_raw[i] = x;
                b_raw[i] = x;
            }
            _ => unreachable!(),
        }
    }
    (BigUint::from_slice(&a_raw), BigUint::from_slice(&b_raw))
}

async fn test_cmp() -> Result<()> {
    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    log::info!("Acquired device");
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );
    log::info!("Created pipeline");

    let n_numbers = (32768 * 4) as usize;
    let n_bytes = (n_numbers * 32) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 4) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, n_bytes);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand::thread_rng();

    let mut input_1 = vec![BigUint::default(); n_numbers];
    let mut input_2 = vec![BigUint::default(); n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            let (a, b) = gen_cmp_u256(&mut rng);
            input_1[i] = a;
            input_2[i] = b;
            expected_output[i] = match &input_1[i].cmp(&input_2[i]) {
                std::cmp::Ordering::Greater => BigUint::from_slice(&[1u32; 8]),
                std::cmp::Ordering::Less => BigUint::from_slice(&[0xffffffffu32; 8]),
                std::cmp::Ordering::Equal => BigUint::from_slice(&[0u32; 8]),
            };
        }

        let input_1_le = write_u256s(&input_1);
        let input_2_le = write_u256s(&input_2);
        let expected_output_le = write_u256s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_1_le));
        queue.write_buffer(&input_2_buffer, 0, bytemuck::cast_slice(&input_2_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 8..(i + 1) * 8];
            let actual = &actual_output[i * 8..(i + 1) * 8];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_1_le[i * 8..(i + 1) * 8]);
                println!("Input 2:");
                print_u32s(&input_2_le[i * 8..(i + 1) * 8]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

async fn test_mul() -> Result<()> {
    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );

    let n_numbers = (32768 * 2) as usize;
    let n_bytes = (n_numbers * 64) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 2) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, 1);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(998244353);

    let mut input = vec![BigUint::default(); 2 * n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            let a = rng.sample(RandomBits::new(256));
            let b = rng.sample(RandomBits::new(256));
            expected_output[i] = &a * &b;
            input[2 * i] = a;
            input[2 * i + 1] = b;
        }

        let input_le = write_u256s(&input);
        let expected_output_le = write_u512s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 16..(i + 1) * 16];
            let actual = &actual_output[i * 16..(i + 1) * 16];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_le[i * 16..i * 16 + 8]);
                println!("Input 2:");
                print_u32s(&input_le[i * 16 + 8..(i + 1) * 16]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

async fn test_cas() -> Result<()> {
    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );

    let n_numbers = (32768 * 4) as usize;
    let n_bytes = (n_numbers * 32) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 4) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, n_bytes);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand::thread_rng();

    let mut input_1 = vec![BigUint::default(); n_numbers];
    let mut input_2 = vec![BigUint::default(); n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            let (a, b) = gen_cmp_u256(&mut rng);
            expected_output[i] = if a >= b { &a - &b } else { a.clone() };
            input_1[i] = a;
            input_2[i] = b;
        }

        let input_1_le = write_u256s(&input_1);
        let input_2_le = write_u256s(&input_2);
        let expected_output_le = write_u256s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_1_le));
        queue.write_buffer(&input_2_buffer, 0, bytemuck::cast_slice(&input_2_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 8..(i + 1) * 8];
            let actual = &actual_output[i * 8..(i + 1) * 8];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_1_le[i * 8..(i + 1) * 8]);
                println!("Input 2:");
                print_u32s(&input_2_le[i * 8..(i + 1) * 8]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

async fn test_redc() -> Result<()> {
    let n = BigUint::from_str(
        "8444461749428370424248824938781546531375899335154063827935233455917409239041",
    )?;
    let n_ = BigUint::from_str(
        "47752251086953357377073236701509605140872345086634869599321669320666611974143",
    )?;
    let r = BigUint::from(2u32).pow(256);
    let r_inv = BigUint::from_str(
        "3482466379256973933331601287759811764685972354380176549708408303012390300674",
    )?;

    // let ref_redc = |t: &BigUint| {
    //     // (t * &r_inv) % &n
    //     let m = ((t % &r) * &n_) % &r;
    //     let t = (t + &m * &n) / &r;
    //     if t >= n {
    //         t - &n
    //     } else {
    //         t
    //     }
    // };

    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );

    let n_numbers = (32768 * 2) as usize;
    let n_bytes = (n_numbers * 64) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 2) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, 1);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(998244353);

    let mut input = vec![BigUint::default(); n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    let upper_bound = &r * &n;
    loop {
        for i in 0..n_numbers {
            let t = rng.sample::<BigUint, _>(RandomBits::new(512)) % &upper_bound;
            // let a = ref_redc(&t);
            let b = (&t * &r_inv) % &n;
            expected_output[i] = b << 256;
            input[i] = t;
        }

        let input_le = write_u512s(&input);
        let expected_output_le = write_u512s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 16..(i + 1) * 16];
            let actual = &actual_output[i * 16..(i + 1) * 16];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_le[i * 16..(i + 1) * 16]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

async fn test_fmul() -> Result<()> {
    let n = BigUint::from_str(
        "8444461749428370424248824938781546531375899335154063827935233455917409239041",
    )?;
    let r = BigUint::from_str(
        "6014086494747379908336260804527802945383293308637734276299549080986809532403",
    )?;

    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );

    let n_numbers = (32768 * 2) as usize;
    let n_bytes = (n_numbers * 64) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 2) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, 1);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(998244353);

    let mut input = vec![BigUint::default(); 2 * n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            let a = rng.sample::<BigUint, _>(RandomBits::new(256)) % &n;
            let b = rng.sample::<BigUint, _>(RandomBits::new(256)) % &n;
            expected_output[i] = (&a * &b * &r) % &n;
            expected_output[i] <<= 256;
            input[2 * i] = (&a * &r) % &n;
            input[2 * i + 1] = (&b * &r) % &n;
        }

        let input_le = write_u256s(&input);
        let expected_output_le = write_u512s(&expected_output);

        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 16..(i + 1) * 16];
            let actual = &actual_output[i * 16..(i + 1) * 16];
            if expected != actual {
                println!("Input 1:");
                print_u32s(&input_le[i * 16..i * 16 + 8]);
                println!("Input 2:");
                print_u32s(&input_le[i * 16 + 8..(i + 1) * 16]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

async fn test_to_mont() -> Result<()> {
    let n = BigUint::from_str(
        "8444461749428370424248824938781546531375899335154063827935233455917409239041",
    )?;
    let r = BigUint::from_str(
        "6014086494747379908336260804527802945383293308637734276299549080986809532403",
    )?;

    let GpuDeviceQueue { device, queue } = GpuDeviceQueue::new().await;
    let pipeline = create_pipeline(
        &device,
        SHADER_CODE,
        &[
            BufferBinding::ReadOnly,
            BufferBinding::ReadOnly,
            BufferBinding::ReadWrite,
        ],
    );

    let n_numbers = (32768 * 2) as usize;
    let n_bytes = (n_numbers * 32) as u64;
    let n_u32s = n_bytes / 4;
    let n_workgroups = (n_numbers / 2) as u32;

    let input_1_buffer = create_buffer(&device, n_bytes);
    let input_2_buffer = create_buffer(&device, 1);
    let output_buffer = create_buffer(&device, n_bytes);
    let output_staging_buffer = create_staging_buffer(&device, n_bytes);
    let bind_group = create_bind_group(
        &device,
        &pipeline,
        &[&input_1_buffer, &input_2_buffer, &output_buffer],
    );

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(998244353);

    let mut input = vec![BigUint::default(); n_numbers];
    let mut expected_output = vec![BigUint::default(); n_numbers];

    let mut iter = 0;
    let mut total = std::time::Duration::default();

    loop {
        for i in 0..n_numbers {
            let a = rng.sample::<BigUint, _>(RandomBits::new(256)) % &n;
            expected_output[i] = (&a * &r) % &n;
            input[i] = a;
        }

        let input_le = write_u256s(&input);
        let expected_output_le = write_u256s(&expected_output);
        let now = std::time::Instant::now();
        queue.write_buffer(&input_1_buffer, 0, bytemuck::cast_slice(&input_le));
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        dispatch(&mut command_encoder, &pipeline, &bind_group, n_workgroups);
        command_encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &output_staging_buffer,
            0,
            n_bytes,
        );
        queue.submit(Some(command_encoder.finish()));
        let mut actual_output = Vec::with_capacity(n_u32s as usize);
        map_buffers! {
            device,
            (output_staging_buffer, [..]) => |view: &[u32]| { actual_output.extend_from_slice(view); }
        }
        total += now.elapsed();
        for i in 0..n_numbers {
            let expected = &expected_output_le[i * 8..(i + 1) * 8];
            let actual = &actual_output[i * 8..(i + 1) * 8];
            if expected != actual {
                println!("Input:");
                print_u32s(&input_le[i * 8..(i + 1) * 8]);
                println!("Expected:");
                print_u32s(expected);
                println!("Actual:");
                print_u32s(actual);
                bail!("Mismatch at index {}", i);
            }
        }
        iter += 1;
        if iter % 100 == 0 {
            println!("{} iterations in {:?}", iter, total);
        }
    }
}

#[allow(clippy::never_loop)]
pub(crate) async fn main_test() -> Result<()> {
    // test_cas().await?;
    let points = load_arc_points(20).await?;
    log::info!("Loaded {} points", points.len());
    loop {
        msm_wgpu::gpu::mont::test_ser(&points);
    }
    let dq = msm_wgpu::gpu::GpuDeviceQueue::new().await;
    let mont = msm_wgpu::gpu::mont::GpuMontgomeryConverter::new(&dq);
    let padd = msm_wgpu::gpu::mont::GpuPadd::new(&dq);
    let padd_orig = msm_wgpu::gpu::mont::GpuPaddOriginal::new(&dq);

    loop {
        // let now = std::time::Instant::now();
        // let ref2 = padd_orig.add(&points).await;
        // log::info!(
        //     "Got {} points added in original form in {:?}",
        //     ref2.len(),
        //     now.elapsed()
        // );

        let now = std::time::Instant::now();
        let output = mont.to_mont(&points).await;
        log::info!(
            "Got {} points converted to Montgomery form in {:?}",
            output.len(),
            now.elapsed()
        );

        let now = std::time::Instant::now();
        let sum = padd.add(&output).await;
        log::info!(
            "Got {} points added in Montgomery form in {:?}",
            sum.len(),
            now.elapsed()
        );

        let now = std::time::Instant::now();
        let back = mont.from_mont(&sum).await;
        log::info!(
            "Got {} points converted from Montgomery form in {:?}",
            back.len(),
            now.elapsed()
        );

        for (i, (operands, result)) in points.chunks(2).zip(back.iter()).enumerate() {
            let p1 = &operands[0];
            let p2 = &operands[1];

            // use ark_ed_on_bls12_377::Fq;
            // use ark_ff::BigInt;
            // let edwards_d = Fq::from_bigint(BigInt!("3021")).unwrap();
            // let expected2 = {
            //     let a = p1.x * p2.x;
            //     let b = p1.y * p2.y;
            //     let c = edwards_d * (p1.t * p2.t);
            //     let d = p1.z * p2.z;
            //     let e = (p1.x + p1.y) * (p2.x + p2.y) - a - b;
            //     let h = a + b;
            //     let f = d - c;
            //     let g = d + c;
            //     EdwardsProjective::new_unchecked(e * f, g * h, e * h, f * g)
            // };

            let expected = p1 + p2;

            if expected != *result {
                log::info!("Mismatch at index {}", i);
                log::info!("p1 = {}", p1);
                log::info!("p2 = {}", p2);
                log::info!("result.x = {}", result.x);
                log::info!("ref.x    = {}", expected.x);
                log::info!("result.y = {}", result.y);
                log::info!("ref.y    = {}", expected.y);
                log::info!("result.t = {}", result.t);
                log::info!("ref.t    = {}", expected.t);
                log::info!("result.z = {}", result.z);
                log::info!("ref.z    = {}", expected.z);
                bail!("Mismatch");
            }
        }
    }
}
