import type { BigIntPoint, U32ArrayPoint } from "../reference/types";

import { nBytesPerPoint, nUint32PerPoint, nUint32PerScalar } from "./consts";
import { setWindowSize, gpuIntraBucketReduction } from "./gpu";

import init, {
  split_dynamic,
  inter_bucket_reduce_dynamic,
  msm_end_to_end_dynamic,
  msm_end_to_end_dynamic_with_idle,
  point_add_affine,
  initThreadPool,
} from "./msm-wasm/pkg/msm_wasm.js";

let initialized = false;
let gpuWorker: Worker | undefined = undefined;

function getBestWindowSize(n: number): number {
  const logN = Math.log2(n);
  return logN == 20 ? 13 : 12;
}

export const compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<{ x: bigint; y: bigint }> => {
  const windowSizeStr = new URLSearchParams(location.search).get("windowSize");
  const windowSize = windowSizeStr
    ? parseInt(windowSizeStr)
    : getBestWindowSize(baseAffinePoints.length);
  setWindowSize(windowSize);

  const sabPoints = new SharedArrayBuffer(
    baseAffinePoints.length * nBytesPerPoint
  );
  const pointBuffer = new Uint32Array(sabPoints);
  const scalarBuffer = new Uint32Array(scalars.length * nUint32PerScalar);

  // Convert Points to Uint32Arrays
  console.time("convert points");
  const hasBigInt =
    (baseAffinePoints.length > 0 &&
      typeof baseAffinePoints[0].x === "bigint") ||
    (scalars.length > 0 && typeof scalars[0] === "bigint");
  if (hasBigInt) {
    // At some point the cost of inter-worker communication will outweigh the
    // benefits of parallelism. 8 seems to be a good number for now.
    const concurrency = Math.min(8, navigator.hardwareConcurrency);
    const chunkSize = Math.ceil(baseAffinePoints.length / concurrency);
    const promises = [];
    for (let i = 0; i < concurrency; i++) {
      const pointsChunk = baseAffinePoints.slice(
        i * chunkSize,
        (i + 1) * chunkSize
      );
      const scalarsChunk = scalars.slice(i * chunkSize, (i + 1) * chunkSize);
      const worker = new Worker("./convert_worker.js");
      let resolvePromise: (_: void) => void;
      promises.push(new Promise<void>((resolve) => (resolvePromise = resolve)));
      worker.onmessage = (e) => {
        pointBuffer.set(e.data.pointBuffer, i * chunkSize * 32);
        scalarBuffer.set(e.data.scalarBuffer, i * chunkSize * 8);
        console.timeStamp(`end worker ${i}`);
        resolvePromise();
      };
      console.timeStamp(`start worker ${i}`);
      worker.postMessage({
        points: pointsChunk,
        scalars: scalarsChunk,
      });
    }
    await Promise.all(promises);
  } else {
    for (let i = 0; i < baseAffinePoints.length; i++) {
      const p = baseAffinePoints[i] as U32ArrayPoint;
      pointBuffer.set(p.x, i * 32);
      pointBuffer.set(p.y, i * 32 + 8);
      pointBuffer.set(p.t, i * 32 + 16);
      pointBuffer.set(p.z, i * 32 + 24);
    }
    for (let i = 0; i < scalars.length; i++) {
      scalarBuffer.set(scalars[i] as Uint32Array, i * 8);
    }
  }
  console.timeEnd("convert points");

  if (!initialized) {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
    initialized = true;
  }

  let result: Uint32Array;
  const cpuWorkRatio: number = parseFloat(
    new URLSearchParams(location.search).get("cpuWorkRatio") || "0.0"
  );
  const cpuShare = Math.floor(cpuWorkRatio * baseAffinePoints.length);
  if (cpuShare === 0.0) {
    // GPU only
    console.time("scalar split (rust)");
    const splitScalars = split_dynamic(windowSize, scalarBuffer);
    console.timeEnd("scalar split (rust)");
    console.time("intra bucket reduction (gpu)");
    const reduced = await gpuIntraBucketReduction(pointBuffer, splitScalars);
    console.timeEnd("intra bucket reduction (gpu)");
    console.time("inter bucket reduction (rust)");
    result = inter_bucket_reduce_dynamic(windowSize, reduced);
    console.timeEnd("inter bucket reduction (rust)");
  } else if (cpuShare >= baseAffinePoints.length) {
    // CPU only
    console.time("end to end rust msm (cpu)");
    result = msm_end_to_end_dynamic(windowSize, scalarBuffer, pointBuffer);
    console.timeEnd("end to end rust msm (cpu)");
  } else {
    // GPU-CPU co-computation
    console.time("scalar split (rust)");
    const splitScalars = split_dynamic(
      windowSize,
      scalarBuffer.subarray(cpuShare * nUint32PerScalar)
    );
    console.timeEnd("scalar split (rust)");
    console.time("intra bucket reduction (gpu)");
    const reducedPromise = new Promise<Uint32Array>((resolve) => {
      if (!gpuWorker) gpuWorker = new Worker("./gpu_worker.js");
      gpuWorker.onmessage = (e) => {
        console.timeEnd("intra bucket reduction (gpu)");
        resolve(e.data as Uint32Array);
      };
      // No need to transfer the pointBuffer, as it's already in the sab.
      gpuWorker.postMessage(
        {
          windowSize,
          pointBuffer: pointBuffer.subarray(cpuShare * nUint32PerPoint),
          splitScalars,
        },
        [splitScalars.buffer]
      );
    });
    console.time("end to end rust msm (cpu)");
    const resultCpu = msm_end_to_end_dynamic_with_idle(
      windowSize,
      scalarBuffer.subarray(0, cpuShare * nUint32PerScalar),
      pointBuffer.subarray(0, cpuShare * nUint32PerPoint),
      0 // Math.floor(navigator.hardwareConcurrency / 2)
    );
    console.timeEnd("end to end rust msm (cpu)");
    const reduced = await reducedPromise;
    console.time("inter bucket reduction (rust)");
    const resultGpu = inter_bucket_reduce_dynamic(windowSize, reduced);
    result = point_add_affine(resultCpu, resultGpu);
    console.timeEnd("inter bucket reduction (rust)");
  }
  const resultBigInts = u32ArrayToBigInts(result);
  return { x: resultBigInts[0], y: resultBigInts[1] };
};

const u32ArrayToBigInts = (u32Array: Uint32Array): bigint[] => {
  const bigInts = [];
  const chunkSize = 8;
  const bitsPerElement = 32;

  for (let i = 0; i < u32Array.length; i += chunkSize) {
    let bigInt = BigInt(0);
    for (let j = 0; j < chunkSize; j++) {
      if (i + j >= u32Array.length) break; // Avoid out-of-bounds access
      const u32 = BigInt(u32Array[i + j]);
      bigInt |= u32 << (BigInt(chunkSize - 1 - j) * BigInt(bitsPerElement));
    }
    bigInts.push(bigInt);
  }

  return bigInts;
};
