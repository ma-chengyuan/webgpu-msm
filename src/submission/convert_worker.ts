import type { BigIntPoint, U32ArrayPoint } from "../reference/types";

type Data = {
  points: BigIntPoint[] | U32ArrayPoint[];
  scalars: bigint[] | Uint32Array[];
};

onmessage = (event) => {
  const data = event.data as Data;
  const points = data.points;
  const pointBuffer = new Uint32Array(points.length * 32);
  const scalars = data.scalars;
  const scalarBuffer = new Uint32Array(scalars.length * 8);

  if (points.length > 0 && typeof points[0].x === "bigint") {
    // const mask = 0xffffffff;
    for (let i = 0; i < points.length; i++) {
      const p = points[i] as BigIntPoint;
      let idx = i * 32;
      for (let c of [p.x, p.y, p.t, p.z]) {
        for (let j = 7; j >= 0; j--) {
          pointBuffer[idx + j] = Number(BigInt.asUintN(32, c));
          c >>= 32n;
        }
        idx += 8;
      }
    }
  } else {
    for (let i = 0; i < points.length; i++) {
      const p = points[i] as U32ArrayPoint;
      pointBuffer.set(p.x, i * 32);
      pointBuffer.set(p.y, i * 32 + 8);
      pointBuffer.set(p.t, i * 32 + 16);
      pointBuffer.set(p.z, i * 32 + 24);
    }
  }

  if (scalars.length > 0 && typeof scalars[0] === "bigint") {
    const mask = 0xffffffffn;
    for (let i = 0; i < scalars.length; i++) {
      let s = scalars[i] as bigint;
      for (let j = 7; j >= 0; j--) {
        scalarBuffer[i * 8 + j] = Number(s & mask);
        s >>= 32n;
      }
    }
  } else {
    for (let i = 0; i < scalars.length; i++)
      scalarBuffer.set(scalars[i] as Uint32Array, i * 8);
  }

  // @ts-expect-error - TS doesn't know we are in a worker so uses the wrong type
  postMessage({ pointBuffer, scalarBuffer }, [
    pointBuffer.buffer,
    scalarBuffer.buffer,
  ]);
  console.timeStamp("data sent");
  close();
};
