import { nBytesPerPoint, nUint32PerPoint } from "./consts";

const gpuPowerPreference: GPUPowerPreference = "high-performance";

export let windowSize = 13;
let nBuckets = 1 << windowSize;
let nWindows = Math.ceil(256 / windowSize);

export function setWindowSize(size: number) {
  windowSize = size;
  nBuckets = 1 << windowSize;
  nWindows = Math.ceil(256 / windowSize);
}

import U256_WGSL from "./wgsl/u256.wgsl";
import FIELD_MODULUS_WGSL from "./wgsl/field_modulus.wgsl";
import CURVE_WGSL from "./wgsl/curve.wgsl";
import PADD_IDX_WGSL from "./wgsl/entry_padd_idx.wgsl";

// TODO: Detect this dynamically
const maxVRAM = 128 * (1 << 20); // 128 MB

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

export async function gpuIntraBucketReduction(
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
