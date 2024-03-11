import { gpuIntraBucketReduction } from "./gpu";

type Data = {
  pointBuffer: Uint32Array;
  splitScalars: Uint32Array;
};

onmessage = async (event) => {
  const data = event.data as Data;
  const result = await gpuIntraBucketReduction(
    data.pointBuffer,
    data.splitScalars
  );
  // @ts-expect-error - TS doesn't know we are in a worker so uses the wrong type
  postMessage(result, [result.buffer]);
};
