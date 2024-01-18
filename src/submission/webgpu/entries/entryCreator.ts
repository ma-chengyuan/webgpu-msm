import { gpuU32Inputs } from "../utils";

export const entry = async (
  inputData: gpuU32Inputs[],
  shaderCode: string,
  u32SizePerOutput: number,
  device: GPUDevice
) => {
  // const time = console.time;
  // const timeEnd = console.timeEnd;

  // eslint-disable-next-line @typescript-eslint/no-empty-function, @typescript-eslint/no-unused-vars
  const time = (_: string) => {};
  // eslint-disable-next-line @typescript-eslint/no-empty-function, @typescript-eslint/no-unused-vars
  const timeEnd = (_: string) => {};

  time("GPU Prepare");
  const allBuffers: GPUBuffer[] = [];

  const numInputs =
    inputData[0].u32Inputs.length / inputData[0].individualInputSize;

  time("GPU Compile");
  const module = device.createShaderModule({
    code: shaderCode,
  });
  timeEnd("GPU Compile");

  time("GPU Buffer Creation");
  const gpuBufferInputs = inputData.map((data) =>
    createU32ArrayInputBuffer(device, data.u32Inputs)
  );

  // Result Matrix
  const resultBufferSize =
    Uint32Array.BYTES_PER_ELEMENT * numInputs * u32SizePerOutput;
  const resultBuffer = device.createBuffer({
    size: resultBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  timeEnd("GPU Buffer Creation");

  time("GPU Create Layouts");
  // Bind group layout and bind group
  const bindGroupLayout = createBindGroupLayout(device, gpuBufferInputs);
  const bindGroup = createBindGroup(
    device,
    bindGroupLayout,
    gpuBufferInputs,
    resultBuffer
  );

  // Pipeline setup

  const layout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });
  const computePipeline = await device.createComputePipelineAsync({
    layout: layout,
    compute: {
      module: module,
      entryPoint: "main",
    },
  });
  timeEnd("GPU Create Layouts");

  time("GPU Encode Commands");
  // Commands submission
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroupCount = Math.ceil(numInputs / 64);
  passEncoder.dispatchWorkgroups(workgroupCount);
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  allBuffers.push(...gpuBufferInputs);
  allBuffers.push(resultBuffer);
  allBuffers.push(gpuReadBuffer);

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    resultBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    resultBufferSize /* size */
  );

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  timeEnd("GPU Encode Commands");

  time("GPU Submit");
  device.queue.submit([gpuCommands]);
  timeEnd("GPU Submit");

  timeEnd("GPU Prepare");

  time("GPU Compute");
  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  timeEnd("GPU Compute");

  time("GPU Cleanup");
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  const result = new Uint32Array(arrayBuffer.slice(0));
  gpuReadBuffer.unmap();

  // Destroy all buffers
  for (const buffer of allBuffers) {
    buffer.destroy();
  }
  timeEnd("GPU Cleanup");

  return result;
};

export const getDevice = async () => {
  if (!("gpu" in navigator)) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    console.log("Failed to get GPU adapter.");
    return;
  }
  return await adapter.requestDevice();
};

const createU32ArrayInputBuffer = (device: GPUDevice, uint32s: Uint32Array) => {
  const gpuBufferU32Inputs = device.createBuffer({
    mappedAtCreation: true,
    size: uint32s.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });
  const arrayBufferInput = gpuBufferU32Inputs.getMappedRange();
  new Uint32Array(arrayBufferInput).set(uint32s);
  gpuBufferU32Inputs.unmap();
  return gpuBufferU32Inputs;
};

const createBindGroupLayout = (
  device: GPUDevice,
  gpuInputBuffers: GPUBuffer[]
) => {
  // Bind group layout and bind group
  const layoutEntries: GPUBindGroupLayoutEntry[] = [];
  for (let i = 0; i < gpuInputBuffers.length; i++) {
    layoutEntries.push({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage",
      },
    });
  }

  const resultLayoutEntry: GPUBindGroupLayoutEntry = {
    binding: gpuInputBuffers.length,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: "storage",
    },
  };

  layoutEntries.push(resultLayoutEntry);

  const layout = { entries: layoutEntries };

  return device.createBindGroupLayout(layout);
};

const createBindGroup = (
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  gpuInputBuffers: GPUBuffer[],
  gpuOutputBuffer: GPUBuffer
) => {
  const entriesToBind = gpuInputBuffers.map((gpuInputBuffer, i) => {
    return {
      binding: i,
      resource: {
        buffer: gpuInputBuffer,
      },
    };
  });

  entriesToBind.push({
    binding: gpuInputBuffers.length,
    resource: {
      buffer: gpuOutputBuffer,
    },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: entriesToBind,
  });

  return bindGroup;
};
