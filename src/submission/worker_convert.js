/**
 * @typedef {{ x: Uint32Array, y: Uint32Array, t: Uint32Array, z: Uint32Array }} U32ArrayPoint
 * @typedef {{ x: BigInt, y: BigInt, t: BigInt, z: BigInt }} BigIntPoint
 */
onmessage = (event) => {
  /**
   * @type {{ points: BigIntPoint[] | U32ArrayPoint[], scalars: BigInt[] | Uint32Array }}
   */
  const data = event.data;
  const points = data.points;
  console.timeStamp("data received");
  const pointBuffer = new Uint32Array(points.length * 32);
  const scalars = data.scalars;
  const scalarBuffer = new Uint32Array(scalars.length * 8);

  if (points.length > 0 && typeof points[0].x === "bigint") {
    // const mask = 0xffffffff;
    for (let i = 0; i < points.length; i++) {
      /**
       * @type {BigIntPoint}
       */
      const p = points[i];
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
      /**
       * @type {U32ArrayPoint}
       */
      const p = points[i];
      pointBuffer.set(p.x, i * 32);
      pointBuffer.set(p.y, i * 32 + 8);
      pointBuffer.set(p.t, i * 32 + 16);
      pointBuffer.set(p.z, i * 32 + 24);
    }
  }

  if (scalars.length > 0 && typeof scalars[0] === "bigint") {
    const mask = 0xffffffffn;
    for (let i = 0; i < scalars.length; i++) {
      /**
       * @type {BigInt}
       */
      let s = scalars[i];
      for (let j = 7; j >= 0; j--) {
        scalarBuffer[i * 8 + j] = Number(s & mask);
        s >>= 32n;
      }
    }
  } else {
    for (let i = 0; i < scalars.length; i++)
      scalarBuffer.set(scalars[i], i * 8);
  }

  postMessage({ pointBuffer, scalarBuffer }, [
    pointBuffer.buffer,
    scalarBuffer.buffer,
  ]);
  console.timeStamp("data sent");
  close();
};
