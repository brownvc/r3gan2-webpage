// For map.html: createLab() gives a WebGPU device and a pipelined generator that can switch model, decoder and
// precision; weights are kept in the same browser storage as the live generator.
import { createFastGAN } from './fastgan.js?v=20261118c';
import { DATA } from './config.js';
export { DATA };

export const Z = 64, E = 64;

// A small device: a phone or tablet, or one reporting 4 GB of memory or less. Caches are sized down for it and the
// FLUX.2 memory probe is skipped (a probe that big can lose the GPU there).
// iPadOS Safari sends a desktop Mac user agent and has no deviceMemory: a 'Mac' with a touch screen is an iPad.
export const SMALL_DEVICE = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent) || (navigator.deviceMemory || 8) <= 4
  || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);

// The GPU stopped for this page (a driver reset, a crash, or memory taken back): a notice over the page with a Reload
// button, since nothing can run on a lost device. Once per page.
// A GPU's name for people: 'amd rdna-3' -> 'AMD RDNA 3', 'nvidia ampere' -> 'NVIDIA Ampere'.
export function gpuLabel(info) {
  const V = { amd: 'AMD', nvidia: 'NVIDIA', intel: 'Intel', apple: 'Apple', arm: 'Arm', qualcomm: 'Qualcomm', microsoft: 'Microsoft' };
  const word = (w) => (/^[a-z]{1,4}$/.test(w) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1));
  const arch = (info.architecture || '').split('-').filter(Boolean).map(word).join(' ');
  return [V[info.vendor] || word(info.vendor || ''), arch].filter(Boolean).join(' ') || info.description || 'this GPU';
}
export function gpuLostNotice(why) {
  if (typeof document === 'undefined' || document.getElementById('gpulost')) return;
  const d = document.createElement('div'); d.id = 'gpulost'; d.setAttribute('role', 'alert');
  d.innerHTML = '<p style="margin:0 0 12px">The GPU stopped working for this page. This can happen when it runs out of memory or its driver resets.</p>'
    + (why ? `<p style="margin:0 0 12px;opacity:.7;font-size:13px">${String(why).replace(/[<&]/g, '')}</p>` : '')
    + '<button type="button" style="font:inherit;font-weight:600;padding:6px 16px;border:0;border-radius:6px;background:#C00404;color:#fff;cursor:pointer">Reload</button>';
  Object.assign(d.style, { position: 'fixed', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', zIndex: 1000, maxWidth: 'min(420px, 90vw)',
    padding: '18px 22px', borderRadius: '10px', background: '#fff', color: '#212529', boxShadow: '0 12px 40px rgba(0,0,0,.35)', textAlign: 'center', font: '15px/1.45 system-ui, sans-serif' });
  d.querySelector('button').onclick = () => location.reload();
  document.body.append(d);
}

// Builds with error scopes: WebGPU reports out-of-memory and invalid objects through them, not by throwing, so without
// this a network too big for the GPU would 'build' and then produce nothing. -> what build returned, or throws.
export async function scoped(device, label, build, destroy) {
  device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
  let out = null, err = null;
  try { out = await build(); } catch (e) { err = e; }
  const v = await device.popErrorScope(), m = await device.popErrorScope();
  if (m || v || err) {
    if (out) try { destroy(out); } catch (e) { /* half built */ }
    throw new Error(m ? `not enough GPU memory for ${label} here` : v ? `the GPU refused ${label} (${v.message})` : err.message);
  }
  return out;
}

// ---- random numbers: mulberry32 + Box-Muller, seeded, so every tile / image is reproducible ----
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296; };
}
export function gauss(n, seed) {
  const u = rng(seed), o = new Float32Array(n);
  for (let i = 0; i < n; i += 2) { const r = Math.sqrt(-2 * Math.log(u())), a = 2 * Math.PI * u(); o[i] = r * Math.cos(a); if (i + 1 < n) o[i + 1] = r * Math.sin(a); }
  return o;
}
export function hash2(i, j, salt) {
  let h = Math.imul(i | 0, 0x27D4EB2D) ^ Math.imul(j | 0, 0x165667B1) ^ Math.imul(salt | 0, 0x9E3779B1);
  h = Math.imul(h ^ (h >>> 15), 0x85EBCA6B); h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35); h ^= h >>> 16;
  return h >>> 0;
}

// ---- weights, kept in Cache Storage: one copy of this code for the map, the live generator and its decoder worker,
// so a file saved by one loads in the others ----
const get = (f) => fetch(f).then((r) => (r.ok ? r : Promise.reject(new Error(`${f}: ${r.status}`))));
// Reads a download to the end, reporting progress.
// Into one buffer of the announced size, so a 528 MB model never needs twice that in memory; chunks only when the size
// is unknown or wrong.
async function readAll(body, total, onStatus, what) {
  const reader = body.getReader();
  let out = total ? new Uint8Array(total) : null, chunks = total ? null : [], got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    if (out && got + value.byteLength <= out.length) out.set(value, got);
    else { if (out) { chunks = [out.slice(0, got)]; out = null; } chunks.push(value); }
    got += value.byteLength;
    onStatus(`Downloading ${what}: ${(got / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)} MB` : ' MB'}`);
  }
  if (out) return got === out.length ? out.buffer : out.slice(0, got).buffer;
  const all = new Uint8Array(got); let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.byteLength; }
  return all.buffer;
}
// Keyed by the server's ETag (or Last-Modified + size), so a re-upload is fetched again; any failure falls back to a
// download. onStatus: short progress messages; what: the file as messages name it; log: a fuller record.
export async function cachedBytes(name, onStatus = () => {}, { what = 'the model', log = () => {} } = {}) {
  const url = new URL(name, location.href).href;
  let cache = null, key = null;
  try {
    const h = await fetch(name, { method: 'HEAD', cache: 'no-store' });
    if (!h.ok) throw new Error(`HEAD ${h.status}`);
    key = `${url}?v=${encodeURIComponent(h.headers.get('etag') || `${h.headers.get('last-modified')}|${h.headers.get('content-length')}`)}`;
    cache = await caches.open('r3gan2-weights');
    const hit = await cache.match(key), size = +h.headers.get('content-length') || 0;
    if (hit) {
      onStatus(`Loading ${what} from this browser…`);
      let b = null;
      try { b = await hit.arrayBuffer(); } catch (e) { log(`${name}: stored copy unreadable (${e.message}), downloading again`); }
      if (!b) { await cache.delete(key).catch(() => {}); b = new ArrayBuffer(0); }
      // A stored copy cut short (say, by a browser crash while saving) is replaced, not trusted.
      if (b.byteLength && (!size || b.byteLength === size)) { log(`${name}: loaded from browser storage`); return b; }
      if (b.byteLength) { log(`${name}: stored copy incomplete (${b.byteLength} of ${size} bytes), downloading again`); await cache.delete(key); }
    }
  } catch (e) { log(`${name}: browser storage unavailable (${e.message}), downloading`); cache = null; }
  const r = await get(name), total = +r.headers.get('content-length') || 0;
  // The download is split in two as it arrives: one copy for the page, one streamed straight into browser storage.
  // (Storing the finished buffer instead makes Chrome fail on the 528 MB L model with "Unexpected internal error".)
  const notSaved = (e) => { log(`${name}: not saved in this browser (${e.name}: ${e.message}), so it downloads again next time`); if (cache) cache.delete(key).catch(() => {}); return false; };
  let body = r.body, saving = null;
  if (cache) {
    try {
      for (const k of await cache.keys()) if (k.url.startsWith(url + '?v=')) await cache.delete(k);
      const [mine, store] = body.tee();
      body = mine;
      saving = cache.put(key, new Response(store, { headers: { 'content-type': 'application/octet-stream' } })).catch(notSaved);
    } catch (e) { notSaved(e); }
  }
  const buf = await readAll(body, total, onStatus, what);
  const stored = saving ? (await saving) !== false : false;  // stored before the caller asks whether it is
  log(`${name}: downloaded${stored ? ' and stored for next time' : ''}`);
  return buf;
}

// The image buffer is FP32 RGB in [0, 1]; this packs it to RGBA8 on the GPU, so a read-back is 256 KB and needs no
// per-pixel work on the CPU.
const PACK = `
@group(0) @binding(0) var<storage, read> I: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= 65536u) { return; }
  O[i] = pack4x8unorm(vec4f(clamp(vec3f(I[3u * i], I[3u * i + 1u], I[3u * i + 2u]), vec3f(0.0), vec3f(1.0)), 1.0));
}`;

// Latent preview, no decoder: each latent pixel's K x K neighbourhood (32 channels each, zeros past the tile's edge) ->
// RGB by a fixed linear map (preview.json: fitted by least squares against the TAEF2 decode
// box-shrunk to 32 px, then brightness and saturation stretched back to the target's spread, which least squares
// shrinks; the same for S, M and L), packed to RGBA8. M: weights [(channel, ky, kx), colour], then 3 biases.
// The fit's numbers (held-out tiles, PSNR vs the shrunk decode): 1x1 17.4 dB, 3x3 22.4 dB, 5x5 24.4 dB.
const previewCode = (K) => `
@group(0) @binding(0) var<storage, read> L: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<u32>;
@group(0) @binding(2) var<storage, read> M: array<f32>;
const K = ${K}u; const R = ${K >> 1};
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let p = g.x; if (p >= 1024u) { return; }
  let y = i32(p / 32u); let x = i32(p % 32u); let nb = 32u * K * K * 3u;
  var c = vec3f(M[nb], M[nb + 1u], M[nb + 2u]);
  for (var ky = 0u; ky < K; ky++) { for (var kx = 0u; kx < K; kx++) {
    let yy = y + i32(ky) - R; let xx = x + i32(kx) - R;
    if (yy < 0 || yy >= 32 || xx < 0 || xx >= 32) { continue; }
    let q = (u32(yy) * 32u + u32(xx)) * 32u;
    for (var ch = 0u; ch < 32u; ch++) { let w = ((ch * K + ky) * K + kx) * 3u; c += L[q + ch] * vec3f(M[w], M[w + 1u], M[w + 2u]); }
  } }
  O[p] = pack4x8unorm(vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0));
}`;

// The ImageNet models the demos can run (the same files and sizes as index.html).
export const MODELS = {
  s: { size: 'S', label: 'R3GAN2-S', params: '17 M', fid: '2.12', mb: 75 },
  m: { size: 'M', label: 'R3GAN2-M', params: '61 M', fid: '1.60', mb: 249 },
  l: { size: 'L', label: 'R3GAN2-L', params: '131 M', fid: '1.52', mb: 528 },
};
export const weightsFile = (code) => `${DATA}imagenet_${code}_weights.bin`;
export async function fileExists(name) {
  try { return (await fetch(name, { method: 'HEAD', cache: 'no-store' })).ok; } catch (e) { return false; }
}
export async function savedInBrowser(name) {  // any stored version of this file (see cachedBytes)
  try { const url = new URL(name, location.href).href; return (await (await caches.open('r3gan2-weights')).keys()).some((k) => k.url.startsWith(url + '?v=')); }
  catch (e) { return false; }
}

const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const SLOTS = 3;

// A generator that can switch model, decoder and precision while running:
//   lab.setModel(code)                         s | m | l: loads the weights (browser storage first)
//   lab.setOptions({ half, gHalf, decoder })   FP16 TAEF2, FP16 generator, 'taef2' | 'flux2'
//   lab.generate(z, c, noise) -> ImageData     up to SLOTS images in flight
// A switch waits for the images in flight to finish and holds new ones until it is done, so every image comes from
// one configuration.
export async function createLab({ onStatus = () => {}, onLost = () => {} } = {}) {
  if (!navigator.gpu) throw new Error('This browser does not offer WebGPU. Recent desktop Chrome, Edge and Firefox have it.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU found no usable GPU on this system.');
  // Firefox reports shader-f16 but its FP16 decoder never finishes (see index.html), so it runs FP32.
  // FP16 weights are uploaded as Float16Array (Chrome 135+): without it, FP32 even if the GPU has shader-f16.
  const hasF16 = adapter.features.has('shader-f16') && !/Firefox\//.test(navigator.userAgent) && typeof Float16Array === 'function';
  const device = await adapter.requestDevice({ requiredFeatures: hasF16 ? ['shader-f16'] : [] });
  let lost = false, gpuErrors = 0;
  device.lost.then((info) => { lost = true; gpuLostNotice(info.message || info.reason); onLost(info); });
  device.addEventListener('uncapturederror', (e) => { if (gpuErrors++ < 3) console.warn('WebGPU error:', e.error.message); });  // a few, not every frame's
  const gpuName = gpuLabel(adapter.info || {});  // older Chromium: no adapter.info
  const names = await get('classes.json').then((r) => r.json());

  const packPipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: PACK }), entryPoint: 'main' } });
  const packed = device.createBuffer({ size: 65536 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  // Read-back buffers, one per image in flight: RGBA8 images (TAEF2) and latents (for the FLUX.2 decoder).
  const pool = (bytes) => ({ free: Array.from({ length: SLOTS }, () => device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })), waiting: [] });
  const imgPool = pool(65536 * 4), latPool = pool(32768 * 4);
  // Latent previews (when preview.json is here): many in flight, as each is only the generator and an 8 KB read-back.
  // Each is made in both colourings, the fitted map and raw (the latent itself: the centre pixel's channels 1-3 as RGB,
  // 0.5 + x / 2, each channel's spread being ~0.5), so the page can switch between them without the GPU.
  const PREV_SLOTS = 8, prevFit = await fetch('preview.json?v=k5b').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const PK = prevFit ? prevFit.k || 1 : 1, PN = 32 * PK * PK * 3 + 3;  // neighbourhood size; weights and biases
  let prevPipe = null, prevPacked = null, prevW = null, prevPool = null;
  if (prevFit) {
    prevPipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: previewCode(PK) }), entryPoint: 'main' } });
    prevPacked = device.createBuffer({ size: 2048 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });  // fitted, then raw
    const raw = new Float32Array(PN), ctr = (PK >> 1) * PK + (PK >> 1);  // M[((channel * K + ky) * K + kx) * 3 + colour]
    for (let j = 0; j < 3; j++) raw[(j * PK * PK + ctr) * 3 + j] = 0.5;
    raw[PN - 3] = raw[PN - 2] = raw[PN - 1] = 0.5;
    prevW = [new Float32Array([...prevFit.w.flat(), ...prevFit.b]), raw].map((m) => {
      const b = device.createBuffer({ size: PN * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, m); return b;
    });
    prevPool = { free: Array.from({ length: PREV_SLOTS }, () => device.createBuffer({ size: 2048 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })), waiting: [] };
  }
  async function take(p) { while (!p.free.length) await new Promise((r) => p.waiting.push(r)); return p.free.pop(); }
  function give(p, b) { p.free.push(b); const w = p.waiting.shift(); if (w) w(); }

  let model = null;                 // { code, man, wbin, embed }
  const nets = new Map();           // `${half}${gHalf}` -> { net, packBG }, for the current model
  const opts = { half: hasF16, gHalf: false, decoder: 'taef2' };
  async function buildNet() {
    const key = `${opts.half && hasF16}${opts.gHalf && hasF16}`;
    if (!nets.has(key)) {
      for (const [k, n] of nets) { n.net.destroy(); nets.delete(k); }  // one network at a time: each holds its own copy of the weights
      const net = await scoped(device, MODELS[model.code].label, () => createFastGAN(device, model.man, model.wbin, { half: opts.half && hasF16, gHalf: opts.gHalf && hasF16 }), (n) => n.destroy());
      const packBG = device.createBindGroup({ layout: packPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: net.img } }, { binding: 1, resource: { buffer: packed } }] });
      const prevBG = prevPipe && prevW.map((w, n) => device.createBindGroup({ layout: prevPipe.getBindGroupLayout(0), entries: [  // fitted, raw
        { binding: 0, resource: { buffer: net.lat } }, { binding: 1, resource: { buffer: prevPacked, offset: n * 4096, size: 4096 } }, { binding: 2, resource: { buffer: w } }] }));
      nets.set(key, { net, packBG, prevBG });
    }
  }

  // ---- latents kept on the GPU after a preview, so its decode can skip the generator ----
  // Keyed by the page's id for the tile; 128 KB each, at most LAT_KEEP (1,024: 128 MB), the oldest dropped first. All
  // are dropped when the network changes (dropLatents; the page also drops them when a tile's inputs change).
  const LAT_BYTES = 32768 * 4, LAT_KEEP = SMALL_DEVICE ? 256 : 1024, keptLat = new Map(), spareLat = [];
  function keepLatent(id, enc, net) {
    let b = keptLat.get(id);
    if (b) keptLat.delete(id);
    else if (keptLat.size >= LAT_KEEP) { const [k0, b0] = keptLat.entries().next().value; keptLat.delete(k0); b = b0; }
    else b = spareLat.pop() || device.createBuffer({ size: LAT_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(net.lat, 0, b, 0, LAT_BYTES); keptLat.set(id, b);
  }
  function takeLatent(id) { const b = keptLat.get(id); if (b) keptLat.delete(id); return b; }  // give back to spareLat after use
  function dropLatents() { for (const b of keptLat.values()) b.destroy(); keptLat.clear(); for (const b of spareLat) b.destroy(); spareLat.length = 0; }  // freed, not kept for reuse

  // ---- switching: one at a time, after the images in flight ----
  const inflight = new Set();
  let gate = null, chain = Promise.resolve();
  function reconfigure(fn) {
    const run = chain.then(async () => {
      let open; gate = new Promise((r) => { open = r; });
      try { while (inflight.size) await Promise.allSettled([...inflight]); dropLatents(); await fn(); }
      finally { gate = null; open(); }
    });
    chain = run.catch(() => {});
    return run;
  }
  async function fetchModel(code) {  // every file the model needs, before anything is freed
    const pre = `imagenet_${code}_`, what = `${MODELS[code].label} (${MODELS[code].mb} MB)`;
    const man = await get(DATA + pre + 'weights.json').then((r) => r.json());
    const wbin = await cachedBytes(DATA + pre + 'weights.bin', (s) => onStatus(s.replace('the model', what)));
    const embed = new Float32Array(await get(DATA + pre + 'embed.bin').then((r) => r.arrayBuffer()));
    return { code, man, wbin, embed };
  }
  async function useModel(m) {
    model = m; onStatus('Building the network…'); await buildNet(); onStatus('');
    convRate = await timeConv();
  }
  // The networks (GPU memory) go before the new model arrives, but the old model's files stay until the new one's are
  // all here: if they can't be fetched (offline) or the new network can't be built (say, not enough GPU memory), the old
  // network is built again from them, with no download, and the error goes to the caller.
  const setModel = (code) => reconfigure(async () => {
    if (model && model.code === code) return;
    const old = model;
    for (const n of nets.values()) n.net.destroy();
    nets.clear();
    try { await useModel(await fetchModel(code)); }
    catch (e) {
      for (const n of nets.values()) n.net.destroy();
      nets.clear(); model = null; onStatus('');
      if (old && old.code !== code) await useModel(old).catch(() => { model = null; });
      throw e;
    }
  });
  const setOptions = (o) => reconfigure(async () => {
    const before = { ...opts };
    Object.assign(opts, o);
    onStatus(opts.decoder === 'flux2' ? 'Loading the FLUX.2 decoder…' : 'Building the network…');
    try { if (opts.decoder === 'flux2') await vae(); await buildNet(); }
    catch (e) {  // keep running on the settings that worked: buildNet freed their network, so it is built again
      Object.assign(opts, before);
      try { await buildNet(); } catch (e2) { console.warn('could not rebuild the previous network:', e2); }
      throw e;
    }
    finally { onStatus(''); }
  });

  // ---- the full FLUX.2 decoder: onnxruntime-web on its own WebGPU device, one decode at a time ----
  let session = null, starting = null, ortChain = Promise.resolve();
  // quiet: no status messages (for refining in the background, see map.html); one load however many ask at once.
  function vae(quiet = false) {
    if (session) return Promise.resolve(session);
    const say = quiet ? () => {} : onStatus;
    return (starting ??= (async () => {
      if (!globalThis.ort) await new Promise((ok, err) => { const s = document.createElement('script'); s.src = ORT + 'ort.webgpu.min.js'; s.onload = ok; s.onerror = () => err(new Error('could not load onnxruntime-web')); document.head.appendChild(s); });
      ort.env.wasm.wasmPaths = ORT;
      const bytes = new Uint8Array(await cachedBytes(DATA + 'vae_dec.onnx', (s) => say(s.replace('the model', 'the FLUX.2 decoder (200 MB)'))));
      say('Starting the FLUX.2 decoder…');
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });
      return session;
    })().finally(() => { starting = null; }));
  }
  // Whether the GPU looks able to hold the full FLUX.2 decoder beside the map: ~200 MB of weights and a few hundred MB
  // of working memory. WebGPU does not say how much memory is free, so this asks for that much (3 x 256 MB) on this
  // device, sees whether it gets it (an out-of-memory error says no), and frees it at once. Devices reporting under
  // 4 GB of system memory (navigator.deviceMemory) are not tried.
  async function fullFits() {
    if (SMALL_DEVICE) return false;
    const bufs = [];
    device.pushErrorScope('out-of-memory');
    for (let k = 0; k < 3; k++) bufs.push(device.createBuffer({ size: 256 * 2 ** 20, usage: GPUBufferUsage.STORAGE }));
    const err = await device.popErrorScope();
    for (const b of bufs) b.destroy();
    return !err;
  }
  const decode = (lat) => {  // latent NCHW [1, 32, 32, 32] -> image NCHW [1, 3, 256, 256] in [0, 1]
    const run = ortChain.then(() => session.run({ latent: new ort.Tensor('float32', lat, [1, 32, 32, 32]) }));
    ortChain = run.catch(() => {});
    return run.then((r) => r.image.data);
  };

  // ---- smooth mode, while the user drags or zooms (the page sets lab.smooth) ----
  // Each image's GPU work is submitted in pieces of about a sixth, one per animation frame, so a frame never waits
  // behind a whole tile (25-50 ms on a laptop GPU); otherwise it goes in one submit, for speed. Images share the net's
  // buffers, so runs take turns (exclusive): a run in pieces holds the others until its last piece is submitted (the
  // page also keeps just one image in flight while smooth).
  let smooth = false, runChain = Promise.resolve(), genRuns = 0;  // genRuns: generator passes run (the readout's Generated)
  const nextFrame = () => new Promise((r) => requestAnimationFrame(r));
  function runOps(net, from, upTo, { before, head, tail } = {}) {  // before(): queue writes; head/tail(enc): around the ops
    const run = runChain.then(async () => {
      const step = smooth ? Math.max(4, Math.ceil((upTo - from) / 6)) : upTo - from;
      if (before) before();
      for (let a = from; a < upTo; a += step) {
        const b = Math.min(upTo, a + step), enc = device.createCommandEncoder();
        if (a === from && head) head(enc);
        net.encodeOps(enc, a, b);
        if (b === upTo && tail) tail(enc);
        device.queue.submit([enc.finish()]);
        if (b < upTo) await nextFrame();
      }
    });
    runChain = run.catch(() => {});
    return run;
  }

  // keepId: a latent kept for this tile (see keepLatent) is decoded instead of running the generator again.
  async function generateOnce(z, c, noise, decoder = opts.decoder, keepId) {
    const built = nets.get(`${opts.half && hasF16}${opts.gHalf && hasF16}`);
    if (!built) throw new Error('no network for these settings (it failed to build)');
    const { net, packBG } = built;
    if (decoder !== 'flux2') {
      const st = await take(imgPool);
      const kb = keepId != null && takeLatent(keepId);  // taken after waiting for a slot, so nothing else uses net.lat now
      try {
        await runOps(net, kb ? net.nG : 0, net.ops.length, {
          before: kb ? null : () => { net.setInputs(z, c, noise); genRuns++; },
          head: kb ? (e) => e.copyBufferToBuffer(kb, 0, net.lat, 0, LAT_BYTES) : null,
          tail: (e) => {
            const pass = e.beginComputePass(); pass.setPipeline(packPipe); pass.setBindGroup(0, packBG); pass.dispatchWorkgroups(256); pass.end();
            e.copyBufferToBuffer(packed, 0, st, 0, 65536 * 4);
          },
        });
        if (kb) spareLat.push(kb);  // queued work runs in order, so it can be reused for later copies
        await st.mapAsync(GPUMapMode.READ);
        const px = new Uint8ClampedArray(st.getMappedRange().slice(0)); st.unmap();
        return new ImageData(px, 256, 256);
      } finally { give(imgPool, st); }
    }
    // FLUX.2: the latent (kept, or from the generator) read back, decoded with ORT.
    const st = await take(latPool);
    const kb = keepId != null && takeLatent(keepId);
    let lat;
    try {
      let enc;
      if (kb) { enc = device.createCommandEncoder(); enc.copyBufferToBuffer(kb, 0, st, 0, LAT_BYTES); }
      else { net.setInputs(z, c, noise); genRuns++; enc = net.run(net.nG); enc.copyBufferToBuffer(net.lat, 0, st, 0, LAT_BYTES); }
      device.queue.submit([enc.finish()]);
      if (kb) spareLat.push(kb);
      await st.mapAsync(GPUMapMode.READ);
      lat = new Float32Array(st.getMappedRange().slice(0)); st.unmap();
    } finally { give(latPool, st); }
    const nchw = new Float32Array(32768);
    for (let p = 0; p < 1024; p++) for (let ch = 0; ch < 32; ch++) nchw[ch * 1024 + p] = lat[p * 32 + ch];
    const img = await decode(nchw), px = new Uint8ClampedArray(65536 * 4);
    for (let p = 0; p < 65536; p++) { px[4 * p] = img[p] * 255; px[4 * p + 1] = img[65536 + p] * 255; px[4 * p + 2] = img[131072 + p] * 255; px[4 * p + 3] = 255; }
    return new ImageData(px, 256, 256);
  }
  async function previewOnce(z, c, noise, keepId) {  // keepId: keep the latent on the GPU for this tile's decode
    const built = nets.get(`${opts.half && hasF16}${opts.gHalf && hasF16}`);
    if (!built) throw new Error('no network for these settings (it failed to build)');
    const { net, prevBG } = built, st = await take(prevPool);
    try {
      await runOps(net, 0, net.nG, {  // the generator only
        before: () => { net.setInputs(z, c, noise); genRuns++; },
        tail: (e) => {
          if (keepId != null) keepLatent(keepId, e, net);
          const pass = e.beginComputePass(); pass.setPipeline(prevPipe);
          for (const bg of prevBG) { pass.setBindGroup(0, bg); pass.dispatchWorkgroups(16); }  // fitted, then raw
          pass.end();
          e.copyBufferToBuffer(prevPacked, 0, st, 0, 2048 * 4);
        },
      });
      await st.mapAsync(GPUMapMode.READ);
      const px = new Uint8ClampedArray(st.getMappedRange().slice(0)); st.unmap();
      return { fit: new ImageData(px.slice(0, 4096), 32, 32), raw: new ImageData(px.slice(4096), 32, 32) };
    } finally { give(prevPool, st); }
  }
  // The Conv decoder alone (the latent preview's linear map), tiles per second: a preview also runs the generator, so
  // its own rate is timed here, on the GPU with nothing else queued: 256 dispatches on one latent, best of three.
  let convRate = 0;
  async function timeConv(n = 256) {
    const built = nets.get(`${opts.half && hasF16}${opts.gHalf && hasF16}`);
    if (!built || !prevPipe) return 0;
    await device.queue.onSubmittedWorkDone();
    let best = Infinity;
    for (let r = 0; r < 3; r++) {
      const e = device.createCommandEncoder(), pass = e.beginComputePass(); pass.setPipeline(prevPipe);
      for (let k = 0; k < n; k++) { pass.setBindGroup(0, built.prevBG[0]); pass.dispatchWorkgroups(16); }
      pass.end();
      const t0 = performance.now(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone();
      best = Math.min(best, performance.now() - t0);
    }
    return n * 1000 / best;
  }
  // Each call waits while a switch is under way, and counts as in flight until it finishes (see reconfigure).
  async function track(fn, ...args) {
    while (gate) await gate;
    const p = fn(...args);
    inflight.add(p); const done = () => inflight.delete(p); p.then(done, done);
    return p;
  }
  // -> ImageData 256x256; decoder: this one image with another decoder than the current ('flux2' needs loadFull first);
  // keepId: decode the latent kept for it by preview, if still kept (else the generator runs as usual)
  const generate = (z, c, noise, decoder, keepId) => track(generateOnce, z, c, noise, decoder, keepId);
  // -> { fit, raw }: ImageData 32x32, the latent in the fitted colours and raw
  const preview = (z, c, noise, keepId) => track(previewOnce, z, c, noise, keepId);

  // Class vector for a weighted mix of classes ({ index: weight }), scaled like the main demo's blend (norm sqrt(E)).
  const cbuf = new Float32Array(E);
  function classVec(mix) {
    cbuf.fill(0);
    for (const k in mix) { const w = mix[k], o = +k * E; for (let i = 0; i < E; i++) cbuf[i] += w * model.embed[o + i]; }
    let s = 0; for (let i = 0; i < E; i++) s += cbuf[i] * cbuf[i];
    const f = Math.sqrt(E / s); for (let i = 0; i < E; i++) cbuf[i] *= f;
    return cbuf;
  }
  return {
    device, hasF16, gpuName, names, slots: SLOTS, setModel, setOptions, generate, classVec,
    preview: prevFit ? preview : null, previewSlots: PREV_SLOTS, loadFull: () => vae(true), fullFits, dropLatents,
    get smooth() { return smooth; }, set smooth(v) { smooth = !!v; }, get generated() { return genRuns; }, get convRate() { return convRate; }, get lost() { return lost; },
    get N() { return model ? model.man.noise_total : 0; }, get model() { return model && model.code; }, get options() { return { ...opts }; },
  };
}

const NF = [];  // one formatter per digit count: toLocaleString with options builds a new one every call
export const fmt = (x, d = 0) => (NF[d] ||= new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })).format(x);
