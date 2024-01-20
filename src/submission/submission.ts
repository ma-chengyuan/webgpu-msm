import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { bigIntToU32Array, u32ArrayToBigInts } from "./utils";

type MSMOptions = {
  bucketImpl: "gpu" | "cpu";
  bucketSumImpl: "gpu" | "cpu";
};

export const compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<{ x: bigint; y: bigint }> => {
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

  const worker = new Worker(new URL("worker.js", import.meta.url));
  const options: MSMOptions = {
    bucketImpl: "gpu",
    bucketSumImpl: "cpu",
  };
  worker.postMessage(
    {
      points: pointBuffer,
      scalars: scalarBuffer,
      options,
    },
    [pointBuffer.buffer, scalarBuffer.buffer]
  );
  return new Promise((resolve) => {
    worker.onmessage = (event) => {
      const result = u32ArrayToBigInts(event.data.result);
      resolve({ x: result[0], y: result[1] });
    };
  });
};
