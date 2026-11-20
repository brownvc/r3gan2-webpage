// The live generator's full FLUX.2 decoder (generator.html), off the page's thread: onnxruntime-web on its own WebGPU
// device. The decoder file is fetched and stored like the weights (lab.js), so storing and reading 199 MB never blocks
// the page. Messages in: { load: { url } }, then { lat, sig }; out: { ready }, { img, sig } or { error }.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
import { cachedBytes } from './lab.js?v=20261121a';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
let session = null;
onmessage = async (e) => {
  try {
    if (e.data.load) {
      const model = await cachedBytes(e.data.load.url);
      session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ['webgpu'] });
      postMessage({ ready: true }); return;
    }
    const { image } = await session.run({ latent: new ort.Tensor('float32', e.data.lat, [1, 32, 32, 32]) });
    const img = new Float32Array(image.data); postMessage({ img, sig: e.data.sig }, [img.buffer]);
  } catch (err) { postMessage({ error: String((err && err.message) || err) }); }
};
