import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { FieldMath } from "./utils/FieldMath";
import { pippinger_msm } from "./webgpu/entries/pippengerMSMEntry";
import {
  bigIntToU32Array,
  bigIntsToU16Array,
  u32ArrayToBigInts,
} from "./webgpu/utils";

import { compute_msm as rust_compute_msm } from "./msm-wgpu/pkg/msm_wgpu";

/* eslint-disable @typescript-eslint/no-unused-vars */
export const compute_msm1 = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<{ x: bigint; y: bigint }> => {
  const fieldMath = new FieldMath();
  const pointsAsU32s = (baseAffinePoints as BigIntPoint[]).map((point) =>
    fieldMath.createPoint(point.x, point.y, point.t, point.z)
  );
  const scalarsAsU16s = Array.from(bigIntsToU16Array(scalars as bigint[]));
  return await pippinger_msm(pointsAsU32s, scalarsAsU16s, fieldMath);
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
  const resultBuffer = rust_compute_msm(pointBuffer, scalarBuffer);
  const result = u32ArrayToBigInts(resultBuffer);
  return { x: result[0], y: result[1] };
};
