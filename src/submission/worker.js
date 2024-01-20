import init, { compute_msm } from "./msm-wgpu/pkg/msm_wgpu.js";

let initialized = false;

onmessage = async (event) => {
  if (!initialized) {
    await init();
    initialized = true;
  }

  /** @type {Uint32Array} */
  const points = event.data.points;
  /** @type {Uint32Array} */
  const scalars = event.data.scalars;
  /** @type {Uint32Array} */
  const result = await compute_msm(points, scalars, event.data.options);
  postMessage({ result: result }, [result.buffer]);
};
