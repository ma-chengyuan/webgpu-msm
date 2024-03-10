import { BigIntPoint, U32ArrayPoint } from "../reference/types";

import init, {
  split_dynamic,
  inter_bucket_reduce_dynamic,
  inter_bucket_reduce_last_dynamic,
  msm_end_to_end_dynamic,
  initThreadPool,
} from "./msm-wasm/pkg/msm_wasm.js";

import aleoInit, {
  initThreadPool as aleoInitThreadPool,
  msm as aleoMsm,
} from "./aleo-wasm-baseline/pkg/aleo_wasm_baseline.js";

let initialized = false;

const gpuPowerPreference: GPUPowerPreference = "high-performance";
const windowSize = parseInt(
  new URLSearchParams(location.search).get("window_size") || "13"
);
const nBuckets = 1 << windowSize;
const nWindows = Math.ceil(256 / windowSize);

export const compute_msm = async (
  baseAffinePoints: BigIntPoint[] | U32ArrayPoint[],
  scalars: bigint[] | Uint32Array[]
): Promise<{ x: bigint; y: bigint }> => {
  const pointBuffer = new Uint32Array(baseAffinePoints.length * 32);
  const scalarBuffer = new Uint32Array(scalars.length * 8);
  console.time("convert points");
  // Convert BigInts to Uint32Arrays

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
      const worker = new Worker(new URL("worker_convert.js", import.meta.url));
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
  if (new URLSearchParams(location.search).has("aleo")) {
    if (!initialized) {
      await aleoInit();
      await aleoInitThreadPool(navigator.hardwareConcurrency);
      initialized = true;
    }
    const result = aleoMsm(scalarBuffer, pointBuffer);
    const resultBigInts = u32ArrayToBigInts(result);
    return { x: resultBigInts[0], y: resultBigInts[1] };
  }

  if (!initialized) {
    await init();
    await initThreadPool(navigator.hardwareConcurrency);
    initialized = true;
  }

  const pureWasm = new URLSearchParams(location.search).has("cpu");
  let result: Uint32Array;
  if (pureWasm) {
    console.time("end to end rust msm (gpu)");
    result = msm_end_to_end_dynamic(windowSize, scalarBuffer, pointBuffer);
    console.timeEnd("end to end rust msm (gpu)");
  } else {
    console.time("scalar split (rust)");
    const splitScalars = split_dynamic(windowSize, scalarBuffer);
    console.timeEnd("scalar split (rust)");
    console.time("intra bucket reduction (gpu)");
    const reduced = await gpuIntraBucketReduction(pointBuffer, splitScalars);
    console.timeEnd("intra bucket reduction (gpu)");
    const useRustForReduction = !new URLSearchParams(location.search).has(
      "gpu_inter_bucket"
    );
    if (useRustForReduction) {
      console.time("inter bucket reduction (rust)");
      result = inter_bucket_reduce_dynamic(windowSize, reduced);
      console.timeEnd("inter bucket reduction (rust)");
    } else {
      // Not really used now.
      console.time("inter bucket reduction (gpu)");
      result = inter_bucket_reduce_last_dynamic(
        windowSize,
        await gpuInterBucketReduction(reduced)
      );
      console.timeEnd("inter bucket reduction (gpu)");
    }
  }
  const resultBigInts = u32ArrayToBigInts(result);
  return { x: resultBigInts[0], y: resultBigInts[1] };
};

import U256_WGSL from "./wgsl/u256.wgsl";
import FIELD_MODULUS_WGSL from "./wgsl/field_modulus.wgsl";
import CURVE_WGSL from "./wgsl/curve.wgsl";
import PADD_IDX_WGSL from "./wgsl/entry_padd_idx.wgsl";
import INTER_BUCKET_WGSL from "./wgsl/entry_inter_bucket.wgsl";
import MONT_WGSL from "./wgsl/entry_mont.wgsl";

// TODO: Detect this dynamically
const maxVRAM = 128 * (1 << 20); // 128 MB

const nUint32PerScalar = 8;
const nBytesPerScalar = 4 * nUint32PerScalar;
const nUint32PerPoint = 4 * nUint32PerScalar;
const nBytesPerPoint = 4 * nUint32PerPoint;

let device: GPUDevice | undefined = undefined;

async function initDevice(): Promise<GPUDevice> {
  if (device === undefined) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: gpuPowerPreference,
    });
    if (!adapter) throw new Error("No adapter found");
    device = await adapter.requestDevice();
  }
  return device;
}

async function gpuIntraBucketReduction(
  points: Uint32Array,
  splitScalars: Uint32Array
): Promise<Uint32Array> {
  const nPoints = points.length / nUint32PerPoint;

  const device = await initDevice();
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
  // - 12 * (nPoints + 2 * nBuckets) for the indices buffers
  // - 384 * batchSize for in/out buffers + input staging buffer
  // - 256 * nBuckets for the bucket buffer and the output staging buffer
  // - 8 bytes for in/out length buffers
  // So having 2^16 buckets already costs 17.5 MB of VRAM. Not much, but not
  // nothing either.
  const maxBatchSize = Math.floor(
    (maxVRAM - 280 * nBuckets - 8 - 12 * nPoints) / 384
  );
  // const batchSize = Math.min(nPoints, maxBatchSize);
  const batchSize =
    nPoints <= maxBatchSize
      ? nPoints
      : Math.ceil(nPoints / Math.ceil(nPoints / maxBatchSize));
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
  const nBytesPerBatch = batchSize * nBytesPerPoint;
  const inOutBuffers = [
    createBuffer(device, nBytesPerBatch, "in/out buffer 0"),
    createBuffer(device, nBytesPerBatch, "in/out buffer 1"),
  ];
  // Initially the bucket is full of zero points
  const initialBucket = new Uint32Array(nUint32PerPoint * nBuckets);
  for (let i = 0; i < nBuckets; i++) {
    // Zero points, when marshalled in big-endian, have two entries that are 1.
    initialBucket[i * nUint32PerPoint + 15] = 1;
    initialBucket[i * nUint32PerPoint + 31] = 1;
  }
  const bucketBuffer = createBuffer(device, initialBucket.byteLength, "bucket");
  const outputStagingBuffer = createOutputStagingBuffer(
    device,
    bucketBuffer.size,
    "staging"
  );
  // Create two binding groups so we can ping-pong between them
  const bindGroups = [0, 1].map((i) =>
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    createBindGroup(device!, pipeline, [
      indicesBuffers[i],
      inOutBuffers[i],
      inOutBuffers[1 - i],
      bucketBuffer,
      indicesLengthBuffers[i],
    ])
  );

  const results = new Uint32Array(nWindows * nBuckets * nUint32PerPoint);

  // Number of PADDs for this round.
  let nPAdds = 0;
  // The PADD indices buffer is a list of indices into the input points & output
  // points buffer. Every PADD requires 3 indices: input1, input2, output.
  const pAddIndices = new Uint32Array(nIndicesBufferBytes / 4);
  // When input2 is 0xffffffff, it means there is no input2, so input1 should be
  // copied to output.
  const PADD_INDEX_NO_INPUT_2 = 0xffffffff;
  // When the high bit of output is set, it means the output should be written
  // to the bucket instead of the output buffer.
  // When the high bit of input2 is set, it means input2 should be read from the
  // bucket instead of the input buffer.
  const PADD_INDEX_BUCKET = 0x80000000;
  const idxByBucket: number[][] = [];

  const inputStagingBuffer = device.createBuffer({
    label: "input staging buffer",
    size: nBytesPerBatch,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint32Array(inputStagingBuffer.getMappedRange()).set(
    points.subarray(0, batchSize * nUint32PerPoint)
  );
  inputStagingBuffer.unmap();

  let fillInputStagingBuffer: (() => Promise<void>) | undefined = undefined;
  for (let w = 0; w < nWindows; w++) {
    let currentBindGroup = 0;
    device.queue.writeBuffer(bucketBuffer, 0, initialBucket);
    if (fillInputStagingBuffer) await fillInputStagingBuffer();
    for (let batchStart = 0; batchStart < nPoints; batchStart += batchSize) {
      const batchEnd = Math.min(batchStart + batchSize, nPoints);
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(
        inputStagingBuffer,
        0,
        inOutBuffers[currentBindGroup],
        0,
        (batchEnd - batchStart) * nBytesPerPoint
      );
      device.queue.submit([commandEncoder.finish()]);
      idxByBucket.length = 0;
      for (let i = 0; i < nBuckets; i++) idxByBucket.push([]);
      for (let i = batchStart; i < batchEnd; i++) {
        const s = splitScalars[w * nPoints + i];
        if (s === 0) continue;
        idxByBucket[s].push(i - batchStart);
      }
      const firstBatch = batchStart === 0;
      const computeNextPAddIndices = () => {
        let nPAddIndices = 0;
        let nextOutputIdx = 0;
        for (let bucket = 0; bucket < nBuckets; bucket++) {
          const indices = idxByBucket[bucket];
          if (indices.length === 0) continue;
          const newIndices = [];
          if (indices.length === 1) {
            pAddIndices[nPAddIndices++] = indices[0];
            pAddIndices[nPAddIndices++] = firstBatch
              ? PADD_INDEX_NO_INPUT_2
              : PADD_INDEX_BUCKET | bucket;
            pAddIndices[nPAddIndices++] = PADD_INDEX_BUCKET | bucket;
          } else if (firstBatch && indices.length === 2) {
            pAddIndices[nPAddIndices++] = indices[0];
            pAddIndices[nPAddIndices++] = indices[1];
            pAddIndices[nPAddIndices++] = PADD_INDEX_BUCKET | bucket;
          } else {
            let i = 0;
            for (; i + 1 < indices.length; i += 2) {
              pAddIndices[nPAddIndices++] = indices[i];
              pAddIndices[nPAddIndices++] = indices[i + 1];
              pAddIndices[nPAddIndices++] = nextOutputIdx;
              newIndices.push(nextOutputIdx++);
            }
            if (i < indices.length) {
              pAddIndices[nPAddIndices++] = indices[i];
              if (firstBatch) {
                pAddIndices[nPAddIndices++] = PADD_INDEX_NO_INPUT_2;
                pAddIndices[nPAddIndices++] = nextOutputIdx;
                newIndices.push(nextOutputIdx++);
              } else {
                pAddIndices[nPAddIndices++] = PADD_INDEX_BUCKET | bucket;
                pAddIndices[nPAddIndices++] = PADD_INDEX_BUCKET | bucket;
              }
            }
          }
          idxByBucket[bucket] = newIndices;
        }
        nPAdds = nPAddIndices / 3;
      };
      computeNextPAddIndices();
      while (nPAdds > 0) {
        // When everything fits in one batch, it's easy mode.
        // prettier-ignore
        device.queue.writeBuffer(indicesBuffers[currentBindGroup], 0, pAddIndices, 0, nPAdds * 3);
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
        // Swap the input and output buffers
        currentBindGroup = 1 - currentBindGroup;
        // Updates nOutputPoints, nPAddIndices, and pAddIndices
        computeNextPAddIndices();
      }
      if (batchEnd < nPoints) {
        const nextBatchStart = batchEnd;
        const nextBatchEnd = Math.min(nextBatchStart + batchSize, nPoints);
        await inputStagingBuffer.mapAsync(GPUMapMode.WRITE);
        new Uint32Array(inputStagingBuffer.getMappedRange()).set(
          points.subarray(
            nextBatchStart * nUint32PerPoint,
            nextBatchEnd * nUint32PerPoint
          )
        );
        inputStagingBuffer.unmap();
      }
    }
    const commandEncoder = device.createCommandEncoder();
    // prettier-ignore
    commandEncoder.copyBufferToBuffer(bucketBuffer, 0, outputStagingBuffer, 0, bucketBuffer.size);
    device.queue.submit([commandEncoder.finish()]);
    if (w != nWindows - 1 && nPoints > batchSize) {
      fillInputStagingBuffer = async () => {
        await inputStagingBuffer.mapAsync(GPUMapMode.WRITE);
        new Uint32Array(inputStagingBuffer.getMappedRange()).set(
          points.subarray(0, batchSize * nUint32PerPoint)
        );
        inputStagingBuffer.unmap();
      };
    } else {
      fillInputStagingBuffer = undefined;
    }
    await outputStagingBuffer.mapAsync(GPUMapMode.READ);
    const range = outputStagingBuffer.getMappedRange();
    results.set(new Uint32Array(range), w * nBuckets * nUint32PerPoint);
    outputStagingBuffer.unmap();
  }
  outputStagingBuffer.destroy();
  inputStagingBuffer.destroy();
  bucketBuffer.destroy();
  indicesBuffers.forEach((buffer) => buffer.destroy());
  indicesLengthBuffers.forEach((buffer) => buffer.destroy());
  inOutBuffers.forEach((buffer) => buffer.destroy());
  // device.destroy();
  return results;
}

/**
 * Performs the inter-bucket reduction step of the MSM algorithm on the GPU.
 * This is slower than the CPU implementation, and may suffer instability due to
 * binding group exhaustion. Definitely use with caution.
 *
 * @param points The flattened array of points to reduce.
 * @returns The reduced points.
 */
async function gpuInterBucketReduction(
  points: Uint32Array
): Promise<Uint32Array> {
  const nPoints = points.length / nUint32PerPoint;
  if (nPoints !== nBuckets * nWindows) throw new Error("Invalid input length");
  const device = await initDevice();
  const shader = device.createShaderModule({
    code: [U256_WGSL, FIELD_MODULUS_WGSL, CURVE_WGSL, INTER_BUCKET_WGSL].join(
      "\n"
    ),
  });
  const bindingTypes: GPUBufferBindingType[] = [
    "read-only-storage",
    "storage",
    "read-only-storage",
    "storage",
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
  const pipelines = ["main_1", "main_2"].map((entryPoint) =>
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    device!.createComputePipeline({
      layout: pipelineLayout,
      compute: { module: shader, entryPoint },
    })
  );

  // VRAM usage:
  // - 384 * batchSize for in/out buffers + input staging buffer
  // - 256 * (batchSize / nBuckets) for output staging buffer
  // - 8 bytes for in/out length buffers
  const maxBatchSize = Math.floor((maxVRAM - 8) / (384 + 256 / nBuckets));
  const batchSize = Math.min(
    nPoints,
    Math.max(1, Math.floor(maxBatchSize / nBuckets)) * nBuckets
  );
  if (batchSize % nBuckets !== 0)
    throw new Error("Invalid batchSize for inter-bucket reduction");
  const nBytesPerBatch = batchSize * nBytesPerPoint;
  const inOutLengthBuffers = [
    createBuffer(device, 4, "in/out length buffer 0"),
    createBuffer(device, 4, "in/out length buffer 1"),
  ];
  const inOutBuffers = [
    createBuffer(device, nBytesPerBatch, "in/out buffer 0"),
    createBuffer(device, nBytesPerBatch, "in/out buffer 1"),
  ];
  const outputStagingBuffer = createOutputStagingBuffer(
    device,
    2 * (batchSize / nBuckets) * nBytesPerPoint,
    "output staging buffer"
  );
  const bindGroups = pipelines.map((pipeline) =>
    [0, 1].map((i) =>
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      createBindGroup(device!, pipeline, [
        inOutBuffers[i],
        inOutBuffers[1 - i],
        inOutLengthBuffers[i],
        inOutLengthBuffers[1 - i],
      ])
    )
  );
  let currentBindGroup = 0;

  const inputStagingBuffer = device.createBuffer({
    label: "input staging buffer",
    size: nBytesPerBatch,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint32Array(inputStagingBuffer.getMappedRange()).set(
    points.subarray(0, batchSize * nUint32PerPoint)
  );
  inputStagingBuffer.unmap();

  const output = new Uint32Array(nWindows * nUint32PerPoint);
  let fillInputStagingBuffer: (() => Promise<void>) | undefined = undefined;

  for (let batchStart = 0; batchStart < nPoints; batchStart += batchSize) {
    console.log("batchStart", batchStart);
    const batchEnd = Math.min(batchStart + batchSize, nPoints);
    const nPointsInBatch = batchEnd - batchStart;
    const nWindowsInBatch = nPointsInBatch / nBuckets;
    device.queue.writeBuffer(
      inOutLengthBuffers[currentBindGroup],
      0,
      Uint32Array.from([nPointsInBatch])
    );
    if (fillInputStagingBuffer) await fillInputStagingBuffer();
    let commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      inputStagingBuffer,
      0,
      inOutBuffers[currentBindGroup],
      0,
      nPointsInBatch * nBytesPerPoint
    );
    device.queue.submit([commandEncoder.finish()]);
    let first = true;
    let nOutputPoints = nPointsInBatch / 2;
    while (nOutputPoints >= nWindowsInBatch) {
      const nWorkgroups = Math.ceil(nOutputPoints / 64);
      commandEncoder = device.createCommandEncoder();
      {
        const computePass = commandEncoder.beginComputePass();
        const pipelineIdx = first ? 0 : 1;
        computePass.setPipeline(pipelines[pipelineIdx]);
        computePass.setBindGroup(0, bindGroups[pipelineIdx][currentBindGroup]);
        computePass.dispatchWorkgroups(nWorkgroups, 1, 1);
        computePass.end();
      }
      device.queue.submit([commandEncoder.finish()]);
      nOutputPoints = nOutputPoints / 2;
      currentBindGroup = 1 - currentBindGroup;
      first = false;
    }
    const copySize = 2 * nWindowsInBatch * nBytesPerPoint;
    commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      inOutBuffers[currentBindGroup],
      0,
      outputStagingBuffer,
      0,
      copySize
    );
    device.queue.submit([commandEncoder.finish()]);
    if (batchEnd < nPoints) {
      fillInputStagingBuffer = async () => {
        const nextBatchStart = batchEnd;
        const nextBatchEnd = Math.min(nextBatchStart + batchSize, nPoints);
        await inputStagingBuffer.mapAsync(GPUMapMode.WRITE);
        new Uint32Array(inputStagingBuffer.getMappedRange()).set(
          points.subarray(
            nextBatchStart * nUint32PerPoint,
            nextBatchEnd * nUint32PerPoint
          )
        );
        inputStagingBuffer.unmap();
      };
    } else {
      fillInputStagingBuffer = undefined;
    }
    await outputStagingBuffer.mapAsync(GPUMapMode.READ);
    const range = outputStagingBuffer.getMappedRange();
    // const output = new Uint32Array(range.slice(0, copySize));
    for (let j = 0; j < nWindowsInBatch; j++) {
      output.set(
        new Uint32Array(
          range.slice(
            (2 * j + 0) * nBytesPerPoint,
            (2 * j + 1) * nBytesPerPoint
          )
        ),
        (batchStart / nBuckets + j) * nUint32PerPoint
      );
    }
    outputStagingBuffer.unmap();
  }

  outputStagingBuffer.destroy();
  inOutBuffers.forEach((buffer) => buffer.destroy());
  inOutLengthBuffers.forEach((buffer) => buffer.destroy());
  inputStagingBuffer.destroy();
  // device.destroy();
  return output;
}

/**
 * Converts a Uint32Array of 256-bit scalars, encoded in big endian, to
 * Montgomery form using the GPU.
 * @param scalars The scalars to convert to Montgomery form.
 * @returns The scalars in Montgomery form.
 */
async function gpuConvertMontgomery(
  scalars: Uint32Array,
  direction: "to" | "from"
) {
  const nScalars = scalars.length / nUint32PerScalar;
  const device = await initDevice();

  const shader = device.createShaderModule({
    code: [U256_WGSL, FIELD_MODULUS_WGSL, CURVE_WGSL, MONT_WGSL].join("\n"),
  });
  const bindingTypes: GPUBufferBindingType[] = ["read-only-storage", "storage"];
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
      entryPoint: direction === "to" ? "to_mont" : "from_mont",
    },
  });

  // VRAM usage:
  // - 128 * batchSize for in/out buffers + input/output staging buffer
  const maxBatchSize = Math.floor(maxVRAM / 128);
  const batchSize = Math.min(
    nScalars,
    Math.max(1, Math.floor(maxBatchSize / nBuckets)) * nBuckets
  );
  const nBytesPerBatch = batchSize * nBytesPerScalar;
  const inputBuffer = createBuffer(device, nBytesPerBatch, "input buffer");
  const outputBuffer = createBuffer(device, nBytesPerBatch, "output buffer");
  const outputStagingBuffer = createOutputStagingBuffer(
    device,
    nBytesPerBatch,
    "output staging buffer"
  );
  const bindGroup = createBindGroup(device, pipeline, [
    inputBuffer,
    outputBuffer,
  ]);
  const inputStagingBuffer = device.createBuffer({
    label: "input staging buffer",
    size: nBytesPerBatch,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint32Array(inputStagingBuffer.getMappedRange()).set(
    scalars.subarray(0, batchSize * nUint32PerScalar)
  );
  inputStagingBuffer.unmap();

  let fillInputStagingBuffer: (() => Promise<void>) | undefined = undefined;

  for (let batchStart = 0; batchStart < nScalars; batchStart += batchSize) {
    const batchEnd = Math.min(batchStart + batchSize, nScalars);
    const nScalarsInBatch = batchEnd - batchStart;
    const nBytesInBatch = nScalarsInBatch * nBytesPerScalar;
    if (fillInputStagingBuffer) await fillInputStagingBuffer();
    const commandEncoder = device.createCommandEncoder();
    // prettier-ignore
    commandEncoder.copyBufferToBuffer(inputStagingBuffer, 0, inputBuffer, 0, nBytesInBatch);
    const nWorkgroups = Math.ceil(nScalarsInBatch / 64);
    {
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(nWorkgroups, 1, 1);
      computePass.end();
    }
    // prettier-ignore
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, outputStagingBuffer, 0, nBytesInBatch);
    device.queue.submit([commandEncoder.finish()]);
    if (batchEnd < nScalars) {
      fillInputStagingBuffer = async () => {
        const nextBatchStart = batchEnd;
        const nextBatchEnd = Math.min(nextBatchStart + batchSize, nScalars);
        await inputStagingBuffer.mapAsync(GPUMapMode.WRITE);
        new Uint32Array(inputStagingBuffer.getMappedRange()).set(
          scalars.subarray(
            nextBatchStart * nUint32PerScalar,
            nextBatchEnd * nUint32PerScalar
          )
        );
        inputStagingBuffer.unmap();
      };
    } else {
      fillInputStagingBuffer = undefined;
    }
    await outputStagingBuffer.mapAsync(GPUMapMode.READ, 0, nBytesInBatch);
    const range = outputStagingBuffer.getMappedRange(0, nBytesInBatch);
    scalars.set(new Uint32Array(range), batchStart * nUint32PerScalar);
    outputStagingBuffer.unmap();
  }
  inputBuffer.destroy();
  outputBuffer.destroy();
  outputStagingBuffer.destroy();
  inputStagingBuffer.destroy();
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

function createOutputStagingBuffer(
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
