mod gpu;
mod test;
use std::{str::FromStr, vec};

use anyhow::{bail, Result};
use ark_ed_on_bls12_377::EdwardsProjective;
use itertools::Itertools;
use msm_wgpu::{compute_msm, Options};
use num_bigint::BigUint;
use serde::Deserialize;
use tokio::{
    fs::File,
    io::{AsyncBufReadExt, BufReader},
};

#[derive(Deserialize)]
struct InputPoint {
    x: String,
    y: String,
    t: String,
    z: String,
}

fn extend_bignum(vec: &mut Vec<u32>, bignum: &BigUint) {
    vec.extend(bignum.iter_u32_digits().pad_using(8, |_| 0u32).rev());
}

fn get_ref_answer(power: usize) -> (BigUint, BigUint) {
    let (x, y) = match power {
        16 => (
            "4490298471131273381350715833932091894064554978284853693957586604825823442429",
            "207233051598812890797414182362695316831408959017076683749810755208551572458",
        ),
        17 => (
            "405755281347735151880827575059343698498813029460786026451708154294960743560",
            "7112985356832152643523650125935205310677117771129806490701829425450717492869",
        ),
        18 => (
            "4020134989704514076121556080357844499902614818105934254331815581426895427831",
            "2694327822589008080344499645494473764166611881342421427746308662023437975766",
        ),
        19 => (
            "3856727778963570638772781884183843350150969534777451295534564482755471873113",
            "1398750101296346671684024297455637342909036274728274942667983346895370713922",
        ),
        20 => (
            "5201851187583570844529445080011852189038251929148722905178398320328749074909",
            "3586360219804356686204324370397321114669962278596135149389460948678051407803",
        ),
        _ => unreachable!(),
    };
    (BigUint::from_str(x).unwrap(), BigUint::from_str(y).unwrap())
}

async fn load_arc_points(power: u32) -> Result<Vec<EdwardsProjective>> {
    use ark_ed_on_bls12_377::Fq;
    use ark_ff::PrimeField;

    let points_file = File::open(format!(
        "../../../public/test-data/points/{}-power-points.txt",
        power
    ))
    .await?;
    let reader = BufReader::new(points_file);
    let mut lines = reader.lines();
    let mut points = vec![];
    while let Some(next_line) = lines.next_line().await? {
        let point: InputPoint = serde_json::from_str(&next_line)?;
        let make_fq = |s: &str| {
            let big_uint = BigUint::from_str(s).unwrap();
            Fq::from_le_bytes_mod_order(&big_uint.to_bytes_le())
        };
        points.push(EdwardsProjective::new_unchecked(
            make_fq(&point.x),
            make_fq(&point.y),
            make_fq(&point.t),
            make_fq(&point.z),
        ));
    }
    Ok(points)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    // return test::main_test().await;

    let power = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16);
    if !(16..=20).contains(&power) {
        bail!("invalid power: {} (must be from 16..20)", power);
    }
    let points_flat = {
        let points_file = File::open(format!(
            "../../../public/test-data/points/{}-power-points.txt",
            power
        ))
        .await?;
        let reader = BufReader::new(points_file);
        let mut lines = reader.lines();
        let mut points_flat = vec![];
        log::info!("reading points...");
        while let Some(next_line) = lines.next_line().await? {
            let point: InputPoint = serde_json::from_str(&next_line)?;
            let x = BigUint::from_str(&point.x)?;
            let y = BigUint::from_str(&point.y)?;
            let t = BigUint::from_str(&point.t)?;
            let z = BigUint::from_str(&point.z)?;
            extend_bignum(&mut points_flat, &x);
            extend_bignum(&mut points_flat, &y);
            extend_bignum(&mut points_flat, &t);
            extend_bignum(&mut points_flat, &z);
        }
        log::info!("done reading points.");
        points_flat
    };
    let scalars_flat = {
        let scalars_file = File::open(format!(
            "../../../public/test-data/scalars/{}-power-scalars.txt",
            power
        ))
        .await?;
        let reader = BufReader::new(scalars_file);
        let mut lines = reader.lines();
        let mut scalars_flat = vec![];
        log::info!("reading scalars...");
        while let Some(next_line) = lines.next_line().await? {
            let scalar = BigUint::from_str(next_line.trim())?;
            extend_bignum(&mut scalars_flat, &scalar);
        }
        log::info!("done reading scalars.");
        scalars_flat
    };
    let options = Options {
        bucket_impl: msm_wgpu::BucketImplementation::Gpu,
        bucket_sum_impl: msm_wgpu::BucketSumImplementation::Cpu,
    };
    log::info!("scalars_flat.len() = {}", scalars_flat.len());
    log::info!("points_flat.len() = {}", points_flat.len());
    let output = compute_msm(&points_flat, &scalars_flat, options).await;
    let output_x = BigUint::new(output[..8].iter().rev().copied().collect());
    let output_y = BigUint::new(output[8..16].iter().rev().copied().collect());
    let (ref_x, ref_y) = get_ref_answer(power);
    log::info!("output_x = {}", output_x);
    log::info!("   ref_x = {}", ref_x);
    log::info!("output_y = {}", output_y);
    log::info!("   ref_y = {}", ref_y);
    if output_x != ref_x || output_y != ref_y {
        bail!("wrong answer");
    }
    Ok(())
}
