// The full FLUX.2 decoder, off the page's thread (generator.html, and lab.js for the map): onnxruntime-web on its own
// WebGPU device. The decoder file is fetched and stored like the weights (lab.js), so storing, reading and compiling
// the 199 MB never block the page. Messages in: { load: { url } }, then { lat, sig }; out: { progress } (loading
// messages), { ready }, { img, sig } or { error }.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
import { cachedBytes } from './lab.js?v=20261123a';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
let session = null;
onmessage = async (e) => {
  try {
    if (e.data.load) {
      const model = await cachedBytes(e.data.load.url, (s) => postMessage({ progress: s }));
      postMessage({ progress: 'Starting the FLUX.2 decoder…' });
      session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ['webgpu'] });
      postMessage({ ready: true }); return;
    }
    const { image } = await session.run({ latent: new ort.Tensor('float32', e.data.lat, [1, 32, 32, 32]) });
    const img = new Float32Array(image.data); postMessage({ img, sig: e.data.sig }, [img.buffer]);
  } catch (err) { postMessage({ error: String((err && err.message) || err) }); }
};
