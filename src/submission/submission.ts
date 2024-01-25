import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { bigIntToU32Array, u32ArrayToBigInts, probeGPUMemory } from "./utils";

import init, {
  compute_msm_js,
  initThreadPool,
} from "./msm-wgpu/pkg/msm_wgpu.js";

let initialized = false;

type MSMOptions = {
  bucketImpl: "gpu" | "cpu";
  bucketSumImpl: "gpu" | "cpu";
};

export const compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<{ x: bigint; y: bigint }> => {
  console.time("convert");
  const pointBuffer = new Uint32Array(baseAffinePoints.length * 32);
  for (let i = 0; i < baseAffinePoints.length; i++) {
    const p = baseAffinePoints[i];
    // prettier-ignore
    pointBuffer.set(typeof p.x === "bigint" ? bigIntToU32Array(p.x) : p.x, i * 32);
    // prettier-ignore
    pointBuffer.set(typeof p.y === "bigint" ? bigIntToU32Array(p.y) : p.y, i * 32 + 8);
    // prettier-ignore
    pointBuffer.set(typeof p.t === "bigint" ? bigIntToU32Array(p.t) : p.t, i * 32 + 16);
    // prettier-ignore
    pointBuffer.set(typeof p.z === "bigint" ? bigIntToU32Array(p.z) : p.z, i * 32 + 24);
  }
  const scalarBuffer = new Uint32Array(scalars.length * 8);
  for (let i = 0; i < scalars.length; i++) {
    const s = scalars[i];
    scalarBuffer.set(typeof s === "bigint" ? bigIntToU32Array(s) : s, i * 8);
  }
  console.timeEnd("convert");

  // const worker = new Worker(new URL("worker.js", import.meta.url));

  if (!initialized) {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
    initialized = true;
  }

  const options: MSMOptions = {
    bucketImpl: "gpu",
    bucketSumImpl: "cpu",
  };

  const result = await compute_msm_js(pointBuffer, scalarBuffer, options);
  const resultBigInts = u32ArrayToBigInts(result);
  return { x: resultBigInts[0], y: resultBigInts[1] };
};
