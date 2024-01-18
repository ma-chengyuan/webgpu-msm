import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { FieldMath } from "./utils/FieldMath";
import { pippinger_msm } from "./webgpu/entries/pippengerMSMEntry";
import { bigIntsToU16Array } from "./webgpu/utils";

/* eslint-disable @typescript-eslint/no-unused-vars */
export const compute_msm = async (
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
