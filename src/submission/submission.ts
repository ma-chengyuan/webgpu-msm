import { BigIntPoint, U32ArrayPoint } from "../reference/types";
import { bigIntToU32Array, u32ArrayToBigInts } from "./utils";

import init, {
  compute_msm_js,
  split_16,
  inter_bucket_reduce_16,
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
  const pointBuffer = new Uint32Array(baseAffinePoints.length * 32);
  const scalarBuffer = new Uint32Array(scalars.length * 8);

  const concurrency = navigator.hardwareConcurrency;
  const chunkSize = Math.ceil(baseAffinePoints.length / concurrency);
  const promises = [];
  console.time("convert points");
  for (let i = 0; i < concurrency; i++) {
    const pointsChunk = baseAffinePoints.slice(
      i * chunkSize,
      (i + 1) * chunkSize
    );
    const scalarsChunk = scalars.slice(i * chunkSize, (i + 1) * chunkSize);
    const worker = new Worker(new URL("worker_convert.js", import.meta.url));
    let resolvePromise: (_: void) => void;
    promises.push(new Promise<void>((resolve) => (resolvePromise = resolve)));
    worker.onmessage = (e) => {
      pointBuffer.set(e.data.pointBuffer, i * chunkSize * 32);
      scalarBuffer.set(e.data.scalarBuffer, i * chunkSize * 8);
      resolvePromise();
    };
    worker.postMessage({
      points: pointsChunk,
      scalars: scalarsChunk,
    });
  }
  await Promise.all(promises);
  console.timeEnd("convert points");
  // if (concurrency > 1) return { x: BigInt(0), y: BigInt(0) };
  // console.time("convert scalars");
  // for (let i = 0; i < scalars.length; i++) {
  //   const s = scalars[i];
  //   scalarBuffer.set(typeof s === "bigint" ? bigIntToU32Array(s) : s, i * 8);
  // }
  // console.timeEnd("convert scalars");
  // const worker = new Worker(new URL("worker.js", import.meta.url));

  if (!initialized) {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
    initialized = true;
  }

  console.time("split (js)");
  const splitScalars = split_16(scalarBuffer);
  console.timeEnd("split (js)");
  console.time("intra bucket reduce (gpu)");
  const reduces = await gpuIntraBucketReduction(pointBuffer, splitScalars);
  console.timeEnd("intra bucket reduce (gpu)");
  console.time("inter bucket reduce (js)");
  const result = inter_bucket_reduce_16(reduces);
  console.timeEnd("inter bucket reduce (js)");
  const resultBigInts = u32ArrayToBigInts(result);
  return { x: resultBigInts[0], y: resultBigInts[1] };
  // return { x: BigInt(0), y: BigInt(0) };

  // const options: MSMOptions = {
  //   bucketImpl: "gpu",
  //   bucketSumImpl: "cpu",
  // };

  // console.time("ffi");
  // const result = await compute_msm_js(pointBuffer, scalarBuffer, options);
  // const resultBigInts = u32ArrayToBigInts(result);
  // return { x: resultBigInts[0], y: resultBigInts[1] };
};

import U256_WGSL from "./msm-wgpu/src/gpu/wgsl/u256.wgsl";
import FIELD_MODULUS_WGSL from "./msm-wgpu/src/gpu/wgsl/field_modulus.wgsl";
import CURVE_WGSL from "./msm-wgpu/src/gpu/wgsl/curve.wgsl";
import PADD_IDX_WGSL from "./msm-wgpu/src/gpu/wgsl/entry_padd_idx.wgsl";

async function gpuIntraBucketReduction(
  points: Uint32Array,
  splitScalars: Uint32Array
) {
  const WINDOW_SIZE = 16;

  const nBuckets = 1 << WINDOW_SIZE;
  const nWindows = Math.ceil(256 / WINDOW_SIZE);

  const nUint32PerPoint = 32;
  const nBytesPerPoint = nUint32PerPoint * 4;

  const nPoints = points.length / 32;

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "low-power",
  });
  if (!adapter) throw new Error("No adapter found");
  const device = await adapter.requestDevice();
  const shader = device.createShaderModule({
    code: [U256_WGSL, FIELD_MODULUS_WGSL, CURVE_WGSL, PADD_IDX_WGSL].join("\n"),
  });
  const bindingTypes: GPUBufferBindingType[] = [
    "read-only-storage",
    "read-only-storage",
    "storage",
    "storage",
    "read-only-storage",
  ];
  const bindGroupLayout = device.createBindGroupLayout({
    entries: bindingTypes.map((type: GPUBufferBindingType, index) => ({
      binding: index,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type, hasDynamicOffset: false, minBindingSize: 0 },
    })),
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });
  const pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: shader,
      entryPoint: "main",
    },
  });

  // Approximate VRAM usage:
  // 12 * (nPoints + 2 * nBuckets)  +  256 * batchSize  +  256 * nBuckets
  // +------indices buffers------+    +-point buffers-+   + result buffer (+staging) +
  // So having 2^16 buckets already costs 17.5 MB of VRAM. Not much, but not
  // nothing either.

  // The buffer to pass indices to the PADD kernel.
  // Each set of indices is 3 u32s: [input1, input2, output], so 12 bytes.
  // Max # of index sets is sum_{bucket} ceil(# of points in bucket / 2)
  // which, if you work out the math, is <= nPoints / 2 + N_BUCKETS
  const nIndicesBufferBytes = (Math.ceil(nPoints / 2) + nBuckets) * 3 * 4;
  const indicesBuffers = [
    createBuffer(device, nIndicesBufferBytes, "indices buffer 0"),
    createBuffer(device, nIndicesBufferBytes, "indices buffer 1"),
  ];
  const indicesLengthBuffers = [
    createBuffer(device, 4, "indices length buffer 0"),
    createBuffer(device, 4, "indices length buffer 1"),
  ];
  // const batchSize = nPoints;
  const batchSize = nPoints / 4;
  const nBytes = batchSize * nBytesPerPoint;
  const inOutBuffers = [
    createBuffer(device, nBytes, "in/out buffer 0"),
    createBuffer(device, nBytes, "in/out buffer 1"),
  ];
  // Initially the bucket is full of zero points
  const initialBucket = new Uint32Array(nUint32PerPoint * nBuckets);
  for (let i = 0; i < nBuckets; i++) {
    // Zero points, when marshalled in big-endian, have two entries that are 1.
    initialBucket[i * nUint32PerPoint + 15] = 1;
    initialBucket[i * nUint32PerPoint + 31] = 1;
  }
  const bucketBuffer = createBuffer(device, initialBucket.byteLength, "bucket");
  // const pointsBuffer = device.createBuffer({
  //   size: points.byteLength,
  //   usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
  //   mappedAtCreation: true,
  // });
  // new Uint32Array(pointsBuffer.getMappedRange()).set(points);
  // pointsBuffer.unmap();
  const stagingBuffer = createStagingBuffer(
    device,
    bucketBuffer.size,
    "staging"
  );
  // Create two binding groups so we can ping-pong between them
  const bindGroups = [0, 1].map((i) =>
    createBindGroup(device, pipeline, [
      indicesBuffers[i],
      inOutBuffers[i],
      inOutBuffers[1 - i],
      bucketBuffer,
      indicesLengthBuffers[i],
    ])
  );

  // Output staging buffer, only needed when we batch inputs
  let outputStagingBuffer: GPUBuffer | undefined = undefined;
  let inputStagingBuffer: GPUBuffer | undefined = undefined;

  const results = [];
  let totalTimeComputeNext = 0;
  let totalTimeWriteBuffer = 0;
  let totalTimeWaitMap = 0;

  // The PADD indices buffer is a list of indices into the input points & output points buffer.
  // Every PADD requires 3 indices: input1, input2, output.
  const pAddIndices = new Uint32Array(nIndicesBufferBytes / 4);
  // The number of PADD indices in the buffer, number of PADDs is nPAddIndices / 3
  let nPAddIndices = 0;
  for (let w = 0; w < nWindows; w++) {
    let currentBindGroup = 0;
    const idxByBucket: number[][] = [];
    for (let i = 0; i < nBuckets; i++) idxByBucket.push([]);
    for (let i = 0; i < nPoints; i++) {
      const s = splitScalars[w * nPoints + i];
      if (s === 0) continue;
      idxByBucket[s].push(i);
    }

    const PADD_INDEX_NO_INPUT_2 = 0xffffffff;
    const PADD_INDEX_OUTPUT_TO_BUCKET = 0x80000000;

    let nOutputPoints = 0;
    const computeNextPAddIndices = () => {
      nPAddIndices = 0;
      let nextOutputIdx = 0;
      const start = performance.now();
      for (let bucket = 0; bucket < nBuckets; bucket++) {
        const indices = idxByBucket[bucket];
        if (indices.length === 0) continue;
        const newIndices = [];
        if (indices.length === 1) {
          pAddIndices[nPAddIndices++] = indices[0];
          pAddIndices[nPAddIndices++] = PADD_INDEX_NO_INPUT_2;
          pAddIndices[nPAddIndices++] = PADD_INDEX_OUTPUT_TO_BUCKET | bucket;
        } else if (indices.length === 2) {
          pAddIndices[nPAddIndices++] = indices[0];
          pAddIndices[nPAddIndices++] = indices[1];
          pAddIndices[nPAddIndices++] = PADD_INDEX_OUTPUT_TO_BUCKET | bucket;
        } else {
          for (let i = 0; i < indices.length; i += 2) {
            pAddIndices[nPAddIndices++] = indices[i];
            pAddIndices[nPAddIndices++] =
              i + 1 < indices.length ? indices[i + 1] : PADD_INDEX_NO_INPUT_2;
            pAddIndices[nPAddIndices++] = nextOutputIdx;
            newIndices.push(nextOutputIdx++);
          }
        }
        idxByBucket[bucket] = newIndices;
      }
      nOutputPoints = nextOutputIdx;
      totalTimeComputeNext += performance.now() - start;
    };

    const start = performance.now();
    device.queue.writeBuffer(bucketBuffer, 0, initialBucket);
    // {
    //   const commandEncoder = device.createCommandEncoder();
    //   commandEncoder.copyBufferToBuffer(
    //     pointsBuffer,
    //     0,
    //     inOutBuffers[currentBindGroup],
    //     0,
    //     points.byteLength
    //   );
    // }
    let inputPointsBufer = points;
    let nInputPoints = nPoints;
    if (nInputPoints <= batchSize) {
      device.queue.writeBuffer(
        inOutBuffers[currentBindGroup],
        0,
        inputPointsBufer
      );
    }
    totalTimeWriteBuffer += performance.now() - start;
    computeNextPAddIndices();
    while (nPAddIndices > 0) {
      if (nInputPoints <= batchSize) {
        // When everything fits in one batch, it's easy mode.
        const nPAdds = nPAddIndices / 3;
        // prettier-ignore
        device.queue.writeBuffer(indicesBuffers[currentBindGroup], 0, pAddIndices, 0, nPAddIndices);
        // prettier-ignore
        device.queue.writeBuffer(indicesLengthBuffers[currentBindGroup], 0, Uint32Array.from([nPAdds]));
        const commandEncoder = device.createCommandEncoder();
        const nWorkgroups = Math.ceil(nPAdds / 64);
        {
          const computePass = commandEncoder.beginComputePass();
          computePass.setPipeline(pipeline);
          computePass.setBindGroup(0, bindGroups[currentBindGroup]);
          computePass.dispatchWorkgroups(nWorkgroups, 1, 1);
          computePass.end();
        }
        device.queue.submit([commandEncoder.finish()]);
      } else {
        // Otherwise things get much trickier.
        if (!inputStagingBuffer) {
          inputStagingBuffer = device.createBuffer({
            label: "input staging buffer",
            size: nBytes,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
          });
        } else {
          await inputStagingBuffer.mapAsync(GPUMapMode.WRITE);
        }
        let batchBuffer = new Uint32Array(inputStagingBuffer.getMappedRange());
        let nextBatchBufIdx = 0;
        let basdPAddIdx = 0;
        let baseOutputIdx = 0;
        let lastOutputIdx = 0;
        const batchedOutput = nOutputPoints > batchSize;
        let outputPointsBuffer: Uint32Array | undefined = undefined;
        const nTotalPAdds = nPAddIndices / 3;

        // eslint-disable-next-line @typescript-eslint/no-empty-function
        let finalizePreviousOutput: () => Promise<void> = async () => {};

        const reshuffle = (i: number) => {
          const inputIdx1 = pAddIndices[i * 3];
          batchBuffer.set(
            // prettier-ignore
            inputPointsBufer.slice(inputIdx1 * nUint32PerPoint, (inputIdx1 + 1) * nUint32PerPoint),
            nextBatchBufIdx * nUint32PerPoint
          );
          pAddIndices[i * 3] = nextBatchBufIdx++;

          const inputIdx2 = pAddIndices[i * 3 + 1];
          if (inputIdx2 !== PADD_INDEX_NO_INPUT_2) {
            batchBuffer.set(
              // prettier-ignore
              inputPointsBufer.slice(inputIdx2 * nUint32PerPoint, (inputIdx2 + 1) * nUint32PerPoint),
              nextBatchBufIdx * nUint32PerPoint
            );
            pAddIndices[i * 3 + 1] = nextBatchBufIdx++;
          }

          const outputIdx = pAddIndices[i * 3 + 2];
          if ((outputIdx & PADD_INDEX_OUTPUT_TO_BUCKET) === 0) {
            lastOutputIdx = outputIdx;
            pAddIndices[i * 3 + 2] -= baseOutputIdx;
          }
        };

        console.time("reshuffle input");
        for (let i = 0; i < nTotalPAdds; i++) {
          reshuffle(i);

          if (nextBatchBufIdx + 2 >= batchSize || i == nTotalPAdds - 1) {
            console.timeEnd("reshuffle input");
            // If we have almost no space for next set of PADD indices, or if
            // we've reached the end,
            const nPAdds = i + 1 - basdPAddIdx;
            inputStagingBuffer.unmap();
            await finalizePreviousOutput();
            // prettier-ignore
            device.queue.writeBuffer(indicesBuffers[currentBindGroup], 0, pAddIndices, basdPAddIdx * 3, nPAdds * 3);
            // prettier-ignore
            device.queue.writeBuffer(indicesLengthBuffers[currentBindGroup], 0, Uint32Array.from([nPAdds]));
            const commandEncoder = device.createCommandEncoder();
            const nWorkgroups = Math.ceil(nPAdds / 64);
            // prettier-ignore
            commandEncoder.copyBufferToBuffer(inputStagingBuffer, 0, inOutBuffers[currentBindGroup], 0, nextBatchBufIdx * nBytesPerPoint);
            {
              const computePass = commandEncoder.beginComputePass();
              computePass.setPipeline(pipeline);
              computePass.setBindGroup(0, bindGroups[currentBindGroup]);
              computePass.dispatchWorkgroups(nWorkgroups, 1, 1);
              computePass.end();
            }
            if (batchedOutput) {
              if (!outputStagingBuffer) {
                outputStagingBuffer = createStagingBuffer(
                  device,
                  (Math.ceil(batchSize / 2) + nBuckets) * nBytesPerPoint
                );
              }
              const nOutputPointsThisBatch = lastOutputIdx - baseOutputIdx + 1;
              const nOutputBytes = nOutputPointsThisBatch * nBytesPerPoint;
              // prettier-ignore
              // inOutBuffers[1 - currentBindGroup] is the output buffer
              commandEncoder.copyBufferToBuffer(inOutBuffers[1 - currentBindGroup], 0, outputStagingBuffer, 0, nOutputBytes);
              device.queue.submit([commandEncoder.finish()]);
              const outputOffset = baseOutputIdx * nUint32PerPoint;
              baseOutputIdx = lastOutputIdx + 1;
              finalizePreviousOutput = async () => {
                if (!outputStagingBuffer) throw new Error("unreachable");
                // prettier-ignore
                await outputStagingBuffer.mapAsync(GPUMapMode.READ, 0, nOutputBytes);
                if (!outputPointsBuffer)
                  outputPointsBuffer = new Uint32Array(
                    nPoints * nUint32PerPoint
                  );
                outputPointsBuffer.set(
                  new Uint32Array(
                    // No need to copy/slice, this array is transient anyway
                    outputStagingBuffer.getMappedRange(0, nOutputBytes)
                  ),
                  outputOffset
                );
                outputStagingBuffer.unmap();
              };
            } else {
              // If we don't need to stream the output, let the data stay in the GPU buffer and we can just
              // ping-pong between the two buffers.
              device.queue.submit([commandEncoder.finish()]);
            }
            basdPAddIdx = i + 1;
            nextBatchBufIdx = 0;

            if (i != nTotalPAdds - 1) {
              await inputStagingBuffer.mapAsync(GPUMapMode.WRITE);
              batchBuffer = new Uint32Array(
                inputStagingBuffer.getMappedRange()
              );
              console.time("reshuffle input");
            }
          }
        }
        if (batchedOutput) {
          if (!outputPointsBuffer) throw new Error("unreachable");
          await finalizePreviousOutput();
          inputPointsBufer = outputPointsBuffer;
        }
      }
      // Swap the input and output buffers
      currentBindGroup = 1 - currentBindGroup;
      // The inputs for the next iteration are the outputs from this iteration
      nInputPoints = nOutputPoints;
      // Updates nOutputPoints, nPAddIndices, and pAddIndices
      computeNextPAddIndices();
    }

    const commandEncoder = device.createCommandEncoder();
    // prettier-ignore
    commandEncoder.copyBufferToBuffer(bucketBuffer, 0, stagingBuffer, 0, bucketBuffer.size);
    device.queue.submit([commandEncoder.finish()]);
    const start1 = performance.now();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    results.push(
      new Uint32Array(
        stagingBuffer.getMappedRange().slice(0, bucketBuffer.size)
      )
    );
    stagingBuffer.unmap();

    totalTimeWaitMap += performance.now() - start1;
  }
  stagingBuffer.destroy();
  bucketBuffer.destroy();
  indicesBuffers.forEach((buffer) => buffer.destroy());
  indicesLengthBuffers.forEach((buffer) => buffer.destroy());
  inOutBuffers.forEach((buffer) => buffer.destroy());
  device.destroy();

  console.log("total time compute next", totalTimeComputeNext);
  console.log("total time write buffer", totalTimeWriteBuffer);
  console.log("total time wait map", totalTimeWaitMap);
  return results;
}

function createBuffer(
  device: GPUDevice,
  size: number,
  label?: string
): GPUBuffer {
  return device.createBuffer({
    label,
    size,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });
}

function createStagingBuffer(
  device: GPUDevice,
  size: number,
  label?: string
): GPUBuffer {
  return device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    mappedAtCreation: false,
  });
}

function createBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: GPUBuffer[]
) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, index) => ({
      binding: index,
      resource: { buffer },
    })),
  });
}
