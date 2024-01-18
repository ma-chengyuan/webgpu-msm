import CurveWGSL from "../wgsl/Curve.wgsl";
import FieldModulusWGSL from "../wgsl/FieldModulus.wgsl";
import U256WGSL from "../wgsl/U256.wgsl";
import EntryScalarMulWGSL from "../wgsl/EntryScalarMul.wgsl";
import EntrySumWGSL from "../wgsl/EntrySum.wgsl";

import { entry, getDevice } from "./entryCreator";
import { ExtPointType } from "@noble/curves/abstract/edwards";
import { FieldMath } from "../../utils/FieldMath";
import { bigIntsToU32Array, gpuU32Inputs, u32ArrayToBigInts } from "../utils";
import { EXT_POINT_SIZE, FIELD_SIZE } from "../params";
import { prune } from "../prune";

/// Pippinger Algorithm Summary:
///
/// Great explanation of algorithm can be found here:
/// https://www.youtube.com/watch?v=Bl5mQA7UL2I
///
/// 1) Break down each 256bit scalar k into 256/c, c bit scalars
///    ** Note: c = 16 seems to be optimal per source mentioned above
///
/// 2) Set up 256/c different MSMs T_1, T_2, ..., T_c where
///    T_1 = a_1_1(P_1) + a_2_1(P_2) + ... + a_n_1(P_n)
///    T_2 = a_1_2(P_1) + a_2_2(P_2) + ... + a_n_2(P_n)
///     .
///     .
///     .
///    T_c = a_1_c(P_1) + a_2_c(P_2) + ... + a_n_c(P_n)
///
/// 3) Use Bucket Method to efficiently compute each MSM
///    * Create 2^c - 1 buckets where each bucket represents a c-bit scalar
///    * In each bucket, keep a running sum of all the points that are mutliplied
///      by the corresponding scalar
///    * T_i = 1(SUM(Points)) + 2(SUM(Points)) + ... + (2^c - 1)(SUM(Points))
///
/// 4) Once the result of each T_i is calculated, can compute the original
///    MSM (T) with the following formula:
///    T <- T_1
///    for j = 2,...,256/c:
///        T <- (2^c) * T
///        T <- T + T_j

// Breaks up an array into separate arrays of size chunkSize
function chunkArray<T>(inputArray: T[], chunkSize = 20000): T[][] {
  let index = 0;
  const arrayLength = inputArray.length;
  const tempArray = [];

  while (index < arrayLength) {
    tempArray.push(inputArray.slice(index, index + chunkSize));
    index += chunkSize;
  }

  return tempArray;
}

export const pippinger_msm = async (
  points: ExtPointType[],
  scalars: number[],
  fieldMath: FieldMath
) => {
  const C = 16;

  console.time("Preprocessing");
  ///
  /// DICTIONARY SETUP
  ///
  // Need to setup our 256/C MSMs (T_1, T_2, ..., T_n). We'll do this
  // by via the bucket method for each MSM
  const numMsms = 256 / C;
  const msms: Map<number, ExtPointType>[] = [];
  for (let i = 0; i < numMsms; i++) {
    msms.push(new Map<number, ExtPointType>());
  }

  ///
  /// BUCKET METHOD
  ///
  let scalarIndex = 0;
  let pointsIndex = 0;
  while (pointsIndex < points.length) {
    const scalar = scalars[scalarIndex];
    const pointToAdd = points[pointsIndex];

    const msmIndex = scalarIndex % msms.length;

    const currentPoint = msms[msmIndex].get(scalar);
    if (currentPoint === undefined) {
      msms[msmIndex].set(scalar, pointToAdd);
    } else {
      msms[msmIndex].set(scalar, currentPoint.add(pointToAdd));
    }

    scalarIndex += 1;
    if (scalarIndex % msms.length == 0) {
      pointsIndex += 1;
    }
  }

  ///
  /// GPU INPUT SETUP & COMPUTATION
  ///
  const pointsConcatenated: bigint[] = [];
  const scalarsConcatenated: number[] = [];
  for (let i = 0; i < msms.length; i++) {
    Array.from(msms[i].values()).map((x) => {
      const expandedPoint = [x.ex, x.ey, x.et, x.ez];
      pointsConcatenated.push(...expandedPoint);
    });
    scalarsConcatenated.push(...Array.from(msms[i].keys()));
  }

  // Need to consider GPU buffer and memory limits so need to chunk
  // the concatenated inputs into reasonable sizes. The ratio of points
  // to scalars is 4:1 since we expanded the point object into its
  // x, y, t, z coordinates.
  const chunkSize = 11_000;
  const chunkedPoints = chunkArray(pointsConcatenated, 4 * chunkSize);
  const chunkedScalars = chunkArray(scalarsConcatenated, chunkSize);
  console.timeEnd("Preprocessing");

  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const device = (await getDevice())!;

  console.time("GPU Point Mul");
  // const gpuResultsAsBigInts = [];
  const allResultsRaw = new Uint32Array(pointsConcatenated.length * 8);
  let offset = 0;

  for (let i = 0; i < chunkedPoints.length; i++) {
    const bufferResult = await point_mul(
      {
        u32Inputs: bigIntsToU32Array(chunkedPoints[i]),
        individualInputSize: EXT_POINT_SIZE,
      },
      {
        u32Inputs: Uint32Array.from(chunkedScalars[i]),
        individualInputSize: FIELD_SIZE,
      },
      device
    );

    console.assert(bufferResult.length === chunkedPoints[i].length * 8);
    allResultsRaw.set(bufferResult || new Uint32Array(0), offset * 8);
    offset += chunkedPoints[i].length;
  }
  console.timeEnd("GPU Point Mul");

  ///
  /// SUMMATION OF SCALAR MULTIPLICATIONS FOR EACH MSM
  ///
  const msmResults = [];
  const bucketing = msms.map((msm) => msm.size);
  let prevBucketSum = 0;
  for (const bucket of bucketing) {
    // console.time("Sum clever");
    const currentSum = await point_sum_wgpu(
      allResultsRaw.slice(
        prevBucketSum * EXT_POINT_SIZE,
        (prevBucketSum + bucket) * EXT_POINT_SIZE
      ),
      device,
      fieldMath
    );
    msmResults.push(currentSum);
    prevBucketSum += bucket;
  }

  ///
  /// SOLVE FOR ORIGINAL MSM
  ///
  let originalMsmResult = msmResults[0];
  for (let i = 1; i < msmResults.length; i++) {
    originalMsmResult = originalMsmResult.multiplyUnsafe(
      BigInt(Math.pow(2, C))
    );
    originalMsmResult = originalMsmResult.add(msmResults[i]);
  }

  device.destroy();
  ///
  /// CONVERT TO AFFINE POINT FOR FINAL RESULT
  ///
  const affineResult = originalMsmResult.toAffine();
  return { x: affineResult.x, y: affineResult.y };
};

const point_mul = async (
  input1: gpuU32Inputs,
  input2: gpuU32Inputs,
  device: GPUDevice
) => {
  const shaderCode = prune([U256WGSL, FieldModulusWGSL, CurveWGSL].join(""), [
    "mul_point_32_bit_scalar",
  ]);

  return await entry(
    [input1, input2],
    shaderCode + EntryScalarMulWGSL,
    EXT_POINT_SIZE,
    device
  );
};

const point_sum_impl = async (input1: gpuU32Inputs, device: GPUDevice) => {
  const shaderCode = prune([U256WGSL, FieldModulusWGSL, CurveWGSL].join(""), [
    "add_points",
  ]);

  return await entry(
    [input1],
    shaderCode + EntrySumWGSL,
    EXT_POINT_SIZE,
    device
  );
};

const point_sum_cpu = (input: Uint32Array, fieldMath: FieldMath) => {
  const resultBigInts = u32ArrayToBigInts(input);
  let sum = fieldMath.createPoint(
    resultBigInts[0],
    resultBigInts[1],
    resultBigInts[2],
    resultBigInts[3]
  );
  const nPoints = input.length / EXT_POINT_SIZE;
  for (let i = 1; i < nPoints; i++) {
    const point = fieldMath.createPoint(
      resultBigInts[i * 4],
      resultBigInts[i * 4 + 1],
      resultBigInts[i * 4 + 2],
      resultBigInts[i * 4 + 3]
    );
    sum = sum.add(point);
  }
  return sum;
};

const point_sum_wgpu = async (
  input1: Uint32Array,
  device: GPUDevice,
  fieldMath: FieldMath
): Promise<ExtPointType> => {
  const BASECASE_SIZE = 1024;
  if (input1.length <= BASECASE_SIZE * EXT_POINT_SIZE) {
    return point_sum_cpu(input1, fieldMath);
  }

  let input = input1;
  const WORKGROUP_SIZE = 64;
  const BATCH_SIZE = EXT_POINT_SIZE * WORKGROUP_SIZE;
  if (input1.length % BATCH_SIZE !== 0) {
    input = new Uint32Array(
      input1.length + BATCH_SIZE - (input1.length % BATCH_SIZE)
    );
    input.set(input1);
    for (let i = input1.length; i < input.length; i += EXT_POINT_SIZE) {
      // Pad with identity points
      input.set([0, 0, 0, 0, 0, 0, 0, 0], i);
      input.set([0, 0, 0, 0, 0, 0, 0, 1], i + 8);
      input.set([0, 0, 0, 0, 0, 0, 0, 0], i + 16);
      input.set([0, 0, 0, 0, 0, 0, 0, 1], i + 24);
    }
  }
  const nResultPoints = input.length / BATCH_SIZE;
  const resultRaw = await point_sum_impl(
    {
      u32Inputs: input,
      individualInputSize: EXT_POINT_SIZE,
    },
    device
  );
  const resultBuf = new Uint32Array(nResultPoints * EXT_POINT_SIZE);
  for (let i = 0; i < nResultPoints; i++) {
    resultBuf.set(
      resultRaw.slice(i * BATCH_SIZE, i * BATCH_SIZE + EXT_POINT_SIZE),
      i * EXT_POINT_SIZE
    );
  }
  return await point_sum_wgpu(resultBuf, device, fieldMath);
};
