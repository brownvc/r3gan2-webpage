// The class globe as a view of map.html, as a module. The page owns the panel, the lab (G) and
// the settings; this view reads them live through env.sh and is told when they change (settings, remake, loadModel).
import { gauss, Z, SMALL_DEVICE } from './lab.js?v=20261122d';
import { geodesic, classMix, tierFor, neighbours, gridTargets, anchorWeights, nearestSpacing } from './globe.js?v=20261122m';
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

// env: G (the lab), Q (the link's parameters), $ (element by id), GD (the globe files' folder), S100 (CSS px per tile
// at 100% zoom), blocked (classes left out), sh (the page's settings, live: psi, soft, span (1, 2, 4 or 8), noiseSeed,
// noiseStr, NOISE, held, paused, latRaw, convOnly, arrows), ok() and fail(e) (a tile made, or failed), refineOK(),
// refineReady() and refineFailed(e) (the full FLUX.2 decoder, when zoomed in and idle).
export async function createGlobeView(env) {
  const { G, Q, $, GD, S100, blocked, sh, NG } = env, dev = G.device;
  async function globeFiles(code) {  // where this model's class codes put each class on the globe
    const lay = await fetch(GD + `imagenet_${code}_classlayout.json`).then((r) => r.json());
    return { points: Float32Array.from(lay.points.flat()), n10: lay.neighbours10 };
  }
  let model = G.model, GF = await globeFiles(model);  // the model as asked for: G.model may change during the fetch

  // ---- the tiling ----
  // Morphing: anchors every 1, 2, 4 or 8 cells are the tiling's vertices 0 to 3 levels coarser
  const anchorLevel = () => 7 - Math.round(Math.log2(clamp(sh.span, 1, 8)));
  let gLevel = anchorLevel();
  const g = geodesic(7, gLevel), N = g.n;
  const anchorTables = new Map([[gLevel, { anchorOf: g.anchorOf, anchorTri: g.anchorTri }]]);
  const rad = new Float32Array(N);  // each cell's circumradius: the image is scaled so the cell fits it at any rotation
  const cellData = new Float32Array(N * 32);  // per cell: centre + radius, then 6 corners (a pentagon repeats its last), 8 vec4
  for (let i = 0; i < N; i++) {
    const s = g.cellStart[i], k = g.cellStart[i + 1] - s; let r = 0;
    cellData.set(g.centres.subarray(3 * i, 3 * i + 3), i * 32);
    for (let j = 0; j < 6; j++) {
      const q = 3 * (s + Math.min(j, k - 1));
      cellData.set(g.corners.subarray(q, q + 3), i * 32 + 4 + 4 * j);
      r = Math.max(r, Math.hypot(g.corners[q] - g.centres[3 * i], g.corners[q + 1] - g.centres[3 * i + 1], g.corners[q + 2] - g.centres[3 * i + 2]));
    }
    rad[i] = r; cellData[i * 32 + 3] = r;
  }

  const unit = (v) => { const r = Math.hypot(...v) || 1; return v.map((x) => x / r); };

  // Per tile: its class mix (cached per cell) and its z.
  let mixCache = new Array(N);
  const centre = (i) => [g.centres[3 * i], g.centres[3 * i + 1], g.centres[3 * i + 2]];
  // As on the map, anchors: an anchor image (class mix, z) every 1, 2, 4 or 8 cells (the vertices of the tiling 0 to 3
  // levels coarser: Morphing), and each cell blends the 3 around it (anchorWeights): the class mix, and z, put back at the typical length
  // sqrt(Z). So a region's tiles morph from one sample to the next, and across a border into the next class.
  let ancMix = new Map(), ancZ = new Map();
  const anchorMix = (a) => { let m = ancMix.get(a); if (!m) { m = classMix(GF.points, centre(a), sh.soft, blocked, GF.spacing ??= nearestSpacing(GF.points, blocked)); ancMix.set(a, m); } return m; };
  const anchorZ = (a) => { let z = ancZ.get(a); if (!z) { z = gauss(Z, (sh.noiseSeed * 10243 + a) % 2147483647); ancZ.set(a, z); } return z; };
  const mixAt = (i) => {
    if (mixCache[i]) return mixCache[i];
    const mix = {}; let t = 0;
    for (const [a, w] of anchorWeights(g, i)) { if (!w) continue; const m = anchorMix(a); for (const c in m) { mix[c] = (mix[c] || 0) + w * m[c]; t += w * m[c]; } }
    for (const c in mix) { mix[c] /= t; if (mix[c] < 0.005) delete mix[c]; }
    return (mixCache[i] = mix);
  };
  function z0At(i) {
    const z = new Float32Array(Z); let s = 0;
    for (const [a, w] of anchorWeights(g, i)) { if (!w) continue; const n = anchorZ(a); for (let k = 0; k < Z; k++) z[k] += w * n[k]; }
    for (let k = 0; k < Z; k++) s += z[k] * z[k];
    s = Math.sqrt(Z / s); for (let k = 0; k < Z; k++) z[k] *= s;
    return z;
  }
  // Each tile's z: its base z, with the noise globe's point for it when the noise globe is on (noiseglobe.js). qCache:
  // each cell's point of the noise field.
  let qCache = new Array(N);
  function tileInputs(i) {
    const mix = mixAt(i), q = (qCache[i] ??= NG.field(centre(i)));
    return { z: NG.z(z0At(i), mix, q), mix };
  }

  // A path's frame at any point p of the globe: as a tile's inputs, blended at p among the anchors round the cell
  // nearest p, so frames morph smoothly between cells.
  // ponytail: the nearest cell by brute force over all cells (~0.3 ms, at most 30 frames a second); a spatial index if it shows
  function inputsAt(p) {
    let i = 0, best = -2;
    for (let k = 0; k < N; k++) { const d = g.centres[3 * k] * p[0] + g.centres[3 * k + 1] * p[1] + g.centres[3 * k + 2] * p[2]; if (d > best) { best = d; i = k; } }
    const mix = {}, z0 = new Float32Array(Z); let t = 0, s = 0;
    for (const [a, w] of anchorWeights(g, i, p)) {
      if (!w) continue;
      const m = anchorMix(a), n = anchorZ(a);
      for (const c in m) { mix[c] = (mix[c] || 0) + w * m[c]; t += w * m[c]; }
      for (let k = 0; k < Z; k++) z0[k] += w * n[k];
    }
    for (const c in mix) { mix[c] /= t; if (mix[c] < 0.005) delete mix[c]; }
    for (let k = 0; k < Z; k++) s += z0[k] * z0[k];
    s = Math.sqrt(Z / s); for (let k = 0; k < Z; k++) z0[k] *= s;
    return { z: NG.z(z0, mix, NG.field(p)), mix };
  }

  // ---- tile images: GPU texture arrays at 16, 64 and 256 px. Every tile made keeps a 16 px copy; 64 and 256 px keep
  // the most recently made (the oldest made is replaced first). A cell draws from the largest it has that its size on
  // screen needs, and only a tier that holds every cell in view (tierFor), or tiles in view would be remade forever. ----
  const SIDE = 2048;
  function tier(px, layers) {
    const per = SIDE / px, cap = per * per * layers;
    const tex = dev.createTexture({ size: [SIDE, SIDE, layers], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    return { px, per, cap, tex, slot: new Map(), owner: new Int32Array(cap).fill(-1), next: 0 };
  }
  const T16 = tier(16, Math.ceil(N / 16384)), T64 = tier(64, SMALL_DEVICE ? 2 : 8), T256 = tier(256, SMALL_DEVICE ? 2 : 8);
  const TIERS = [T16, T64, T256];
  function put(T, cell, bmp) {
    let s = T.slot.get(cell);
    if (s === undefined) {
      s = T.next; T.next = (s + 1) % T.cap;
      if (T.owner[s] >= 0) T.slot.delete(T.owner[s]);
      T.owner[s] = cell; T.slot.set(cell, s);
    }
    const p2 = T.per * T.per, r = s % p2;
    dev.queue.copyExternalImageToTexture({ source: bmp }, { texture: T.tex, origin: [(r % T.per) * T.px, Math.floor(r / T.per) * T.px, Math.floor(s / p2)] }, [T.px, T.px]);
  }
  const made = new Int32Array(N).fill(-1);  // the epoch each cell's images were made for (-1: none)
  const ver = new Int32Array(N);  // bumped when one cell's inputs change (its class's noise spun) without a new epoch
  let epoch = 0;
  function clearTiles() { for (const T of TIERS) { T.slot.clear(); T.owner.fill(-1); T.next = 0; } made.fill(-1); }
  async function store(cell, img) {
    const bmps = await Promise.all([16, 64].map((s) => createImageBitmap(img, { resizeWidth: s, resizeHeight: s, resizeQuality: 'high' })).concat(createImageBitmap(img)));
    TIERS.forEach((T, t) => { put(T, cell, bmps[t]); bmps[t].close(); });
  }
  async function storePreview(cell, img) {  // the 32 px latent preview: the smallest tier only, until the decode replaces it
    const b = await createImageBitmap(img, { resizeWidth: 16, resizeHeight: 16, resizeQuality: 'high' });
    put(T16, cell, b); b.close();
  }

  // ---- drawing ----
  const cv = $('globe'), lc = $('labels'), lctx = lc.getContext('2d'), wrap = $('wrap');
  const ctx = cv.getContext('webgpu'), fmtTex = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device: dev, format: fmtTex, alphaMode: 'opaque' });
  let dpr = 1, W = 0, H = 0;
  let cw = 0, ch = 0;  // the canvas's CSS size
  const resize = () => { dpr = devicePixelRatio || 1; cw = wrap.clientWidth; ch = wrap.clientHeight; W = cv.width = lc.width = Math.round(cw * dpr); H = cv.height = lc.height = Math.round(ch * dpr); };
  new ResizeObserver(resize).observe(wrap); resize();
  const cellBuf = dev.createBuffer({ size: cellData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(cellBuf, 0, cellData);
  const MAXI = Math.ceil(N / 2) + 1024;
  const instBuf = dev.createBuffer({ size: MAXI * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const uniBuf = dev.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const morphBuf = dev.createBuffer({ size: MAXI * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const shader = dev.createShaderModule({ code: `
// rotation rows; sc = (px per unit x, y in clip units, side); gr = the grid's row direction on screen and its cells'
// half width and half height (unit-sphere units)
struct U { r0: vec4f, r1: vec4f, r2: vec4f, sc: vec4f, gr: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(7) var<storage, read> morph: array<vec4f>;  // per instance: its brick place (x, y), how far reshaped, its row slide (+-1)
@group(0) @binding(1) var<storage, read> cells: array<vec4f>;
@group(0) @binding(2) var<storage, read> inst: array<vec4u>;     // cell, tier, slot, -
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var t16: texture_2d_array<f32>;
@group(0) @binding(5) var t64: texture_2d_array<f32>;
@group(0) @binding(6) var t256: texture_2d_array<f32>;
struct O { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) ts: vec2u, @location(2) shade: f32 };
fn rot(p: vec3f) -> vec3f { return vec3f(dot(u.r0.xyz, p), dot(u.r1.xyz, p), dot(u.r2.xyz, p)); }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> O {
  let it = inst[ii]; let b = it.x * 8u; let c4 = cells[b]; let c = rot(c4.xyz);
  let tri = vi / 3u; let k = vi % 3u;
  var q = c;
  if (k == 1u) { q = rot(cells[b + 1u + tri].xyz); }
  if (k == 2u) { q = rot(cells[b + 1u + (tri + 1u) % 6u].xyz); }
  // the tile's upright frame on the surface: its top towards the top of the screen
  var up = vec3f(0.0, 1.0, 0.0) - c.y * c;
  if (length(up) < 1e-4) { up = vec3f(0.0, 0.0, 1.0) - c.z * c; }
  up = normalize(up); let rt = cross(up, c); let d = q - c;
  var p = q.xy; var uv = vec2f(0.5) + vec2f(dot(d, rt), -dot(d, up)) / (2.0 * c4.w);
  // close in: the hexagon reshapes into its grid rectangle, in two phases, neither opening a gap. First (s1) it
  // flattens into a brick where the lattice puts it (m.xy): its corners lie between its neighbours (30, 90, 150...
  // degrees from the row direction); the four diagonal ones go to the brick's corners, the two across the row to its top
  // and bottom middles, and since every corner is shared by three cells that all send it to the same place, the cells
  // stay edge to edge (a brick wall). Then (s2) alternate rows slide a quarter cell (m.w: which way) along their now
  // straight seams, lining the columns up. The image goes to the whole brick (full width, cropped top and bottom).
  let m = morph[ii];
  if (m.z > 0.0) {
    let e = u.gr.xy; let f = vec2f(-e.y, e.x); let B = m.xy;
    let s1 = smoothstep(0.0, 1.0, clamp(2.0 * m.z, 0.0, 1.0)); let s2 = smoothstep(0.0, 1.0, clamp(2.0 * m.z - 1.0, 0.0, 1.0));
    var s = B;
    if (k != 0u) {
      let dd = q.xy - c.xy; let a = dot(dd, e) / max(length(dd), 1e-9); let sb = select(-1.0, 1.0, dot(dd, f) > 0.0);
      if (abs(a) < 0.25) { s = B + sb * u.gr.w * f; } else { s = B + sign(a) * u.gr.z * e + sb * u.gr.w * f; }
    }
    let uvs = vec2f(0.5) + vec2f(dot(s - B, e), -dot(s - B, f)) / (2.0 * u.gr.z);
    p = mix(p, s, s1) + s2 * m.w * 0.5 * u.gr.z * e; uv = mix(uv, uvs, s1);
  }
  var o: O;
  o.pos = vec4f(p.x * u.sc.x, p.y * u.sc.y, 0.5, 1.0);
  o.uv = uv;
  o.ts = it.yz; o.shade = 0.55 + 0.45 * max(c.z, 0.0);
  return o;
}
fn at(px: f32, s: u32, uv: vec2f) -> vec3f {  // atlas position of slot s's uv, and its layer
  let per = u32(u.sc.z / px); let p2 = per * per; let r = s % p2;
  let m = clamp(uv, vec2f(0.5 / px), vec2f(1.0 - 0.5 / px));
  return vec3f((vec2f(f32(r % per), f32(r / per)) * px + m * px) / u.sc.z, f32(s / p2));
}
@fragment fn fs(i: O) -> @location(0) vec4f {
  let a = at(16.0, i.ts.y, i.uv); let b = at(64.0, i.ts.y, i.uv); let c = at(256.0, i.ts.y, i.uv);
  let x16 = textureSampleLevel(t16, smp, a.xy, i32(a.z), 0.0);
  let x64 = textureSampleLevel(t64, smp, b.xy, i32(b.z), 0.0);
  let x256 = textureSampleLevel(t256, smp, c.xy, i32(c.z), 0.0);
  var col = x16;
  if (i.ts.x == 1u) { col = x64; }
  if (i.ts.x == 2u) { col = x256; }
  return vec4f(col.rgb * i.shade, 1.0);
}` });
  const pipe = dev.createRenderPipeline({ layout: 'auto', vertex: { module: shader, entryPoint: 'vs' }, fragment: { module: shader, entryPoint: 'fs', targets: [{ format: fmtTex }] }, primitive: { topology: 'triangle-list' } });
  const smp = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const bind = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: { buffer: cellBuf } }, { binding: 2, resource: { buffer: instBuf } },
    { binding: 3, resource: smp }, ...TIERS.map((T, t) => ({ binding: 4 + t, resource: T.tex.createView({ dimension: '2d-array' }) })),
    { binding: 7, resource: { buffer: morphBuf } }] });

  // ---- view: rotation (3x3, rows) and zoom (CSS px per unit radius) ----
  const facing = (zr) => {  // a rotation (rows) that brings unit vector zr to the front, upright
    const a = Math.abs(zr[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const xr = unit([a[1] * zr[2] - a[2] * zr[1], a[2] * zr[0] - a[0] * zr[2], a[0] * zr[1] - a[1] * zr[0]]);
    return [...xr, zr[1] * xr[2] - zr[2] * xr[1], zr[2] * xr[0] - zr[0] * xr[2], zr[0] * xr[1] - zr[1] * xr[0], ...zr];
  };
  // Start as the map does: at the golden retriever, at the map's starting zoom (84 CSS px per tile).
  const gr = G.names.indexOf('golden retriever');
  let Rm = facing([GF.points[3 * gr], GF.points[3 * gr + 1], GF.points[3 * gr + 2]]);
  if (Q.has('rot')) { const r = Q.get('rot').split(',').map(Number); if (r.length === 9 && r.every(Number.isFinite)) Rm = r; }
  const fitR = () => 0.42 * Math.min(wrap.clientWidth, wrap.clientHeight);
  const meanRad = rad.reduce((s, x) => s + x, 0) / N;
  let zoom = (Q.get('layout') === 'globe' && +Q.get('zoom') > 0 ? +Q.get('zoom') : S100) / (2 * meanRad), zoomGoal = zoom, lastFrame = performance.now();
  const maxZoom = () => 300 / (2 * meanRad);  // a tile ~300 CSS px across
  const mul = (A, B) => [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]));
  function turn(ax, ay) {  // rotate the globe: ax about the screen's vertical axis, ay about its horizontal axis
    const cy = Math.cos(ax), sy = Math.sin(ax), cx = Math.cos(ay), sx = Math.sin(ay);
    Rm = mul(mul([1, 0, 0, 0, cx, -sx, 0, sx, cx], [cy, 0, sy, 0, 1, 0, -sy, 0, cy]), Rm);
    // re-orthonormalise now and then against drift
    const r0 = unit(Rm.slice(0, 3)), r1u = Rm.slice(3, 6), d = r1u.reduce((s, x, i) => s + x * r0[i], 0), r1 = unit(r1u.map((x, i) => x - d * r0[i]));
    const r2 = [r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2], r0[0] * r1[1] - r0[1] * r1[0]];
    Rm = [...r0, ...r1, ...r2];
  }
  // A flick, as on the map: the drag's velocity (smoothed, px/s) carries on after the release and dies away (e^-4t);
  // a release more than 80 ms after the last move does not fling. apply(dx, dy) moves by pixels.
  function flinger(apply) {
    let vx = 0, vy = 0, t = 0, on = false, last = 0;
    return {
      down() { vx = vy = 0; on = false; t = performance.now(); },
      move(dx, dy) { const now = performance.now(), dt = Math.max(1, now - t) / 1000; vx = 0.8 * vx + 0.2 * dx / dt; vy = 0.8 * vy + 0.2 * dy / dt; t = now; },
      up() { on = performance.now() - t <= 80 && Math.hypot(vx, vy) > 30; last = performance.now(); },
      get on() { return on; },
      step() {
        const now = performance.now(), dt = Math.min(0.05, (now - last) / 1000); last = now; if (!on) return;
        apply(vx * dt, vy * dt); const k = Math.exp(-4 * dt); vx *= k; vy *= k; if (Math.hypot(vx, vy) < 4) on = false;
      },
    };
  }
  const globeFling = flinger((dx, dy) => turn(dx / zoom, dy / zoom));
  const pointers = new Map(); let pinch = 0;
  // The page's path tool is offered each pointer first (env.pen: Draw mode, or a drag on the path's dot); one it
  // takes draws or scrubs, and does not turn the globe. Positions for it are CSS px within the canvas.
  let penId = null, penAt = [0, 0];
  const at = (e) => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId);
    if (penId !== null) { env.pen.cancel(); pointers.set(penId, penAt); penId = null; }  // a second finger: the stroke ends and the two pinch, as on the flat maps
    else if (!pointers.size && env.pen.down(...at(e), e)) { penId = e.pointerId; penAt = [e.clientX, e.clientY]; return; }
    pointers.set(e.pointerId, [e.clientX, e.clientY]); cv.classList.add('drag'); globeFling.down();
  });
  cv.addEventListener('pointermove', (e) => {
    if (e.pointerId === penId) { penAt = [e.clientX, e.clientY]; env.pen.move(...at(e), e); return; }
    const p = pointers.get(e.pointerId); if (!p) { env.pen.hover(...at(e)); return; }
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()], d0 = Math.hypot(a[0] - b[0], a[1] - b[1]); pointers.set(e.pointerId, [e.clientX, e.clientY]);
      const [c, dd] = [...pointers.values()], d1 = Math.hypot(c[0] - dd[0], c[1] - dd[1]);
      if (d0 > 0) zoom = zoomGoal = clamp(zoom * d1 / d0, fitR() * 0.5, maxZoom());
      globeFling.down(); return;  // a pinch does not fling
    }
    const dx = e.clientX - p[0], dy = e.clientY - p[1];
    turn(dx / zoom, dy / zoom); globeFling.move(dx, dy); pointers.set(e.pointerId, [e.clientX, e.clientY]);
  });
  const up = (e) => {
    if (e.pointerId === penId) { penId = null; env.pen.up(e); return; }
    pointers.delete(e.pointerId); if (!pointers.size) { cv.classList.remove('drag'); globeFling.up(); }
  };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  // The wheel sets a goal; the frame glides the zoom there (time constant 0.1 s), so a notch is a short glide, not a jump.
  cv.addEventListener('wheel', (e) => { e.preventDefault(); env.pen.zoomed(); zoomGoal = clamp(zoomGoal * Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)), fitR() * 0.5, maxZoom()); }, { passive: false });

  // ---- generation: the tiles in view that lack an image for this epoch (or a large enough one), nearest the centre
  // first, as many in flight as the lab has slots ----
  // As on the map: a cell showing nothing gets its latent preview first (G.preview, 32 px, kept on the GPU for its
  // decode), then is decoded; previews and decodes each have their own slots, and both go nearest the centre first.
  let inflight = 0, inflightP = 0, madeN = 0, madeFlux = 0, previewsN = 0, remadeN = 0, want = [], on = false, looping = false, movedAt = -1e9;
  let left = 0, leftP = 0, busy = 0, busyP = 0, busyAny = 0;  // for the readout: in view still to decode, still to preview; the stages' clocks
  const pending = new Set();
  // rePv: cells whose latent preview is made again (Conv switched on or off). refined: cells whose image is the full
  // FLUX.2 decoder's. When every tile in view is made, the view is close in and left alone (env.refineOK), the tiles
  // in view are made again with the full decoder, nearest the centre first, one at a time, as on the flat maps.
  const rePv = new Set(), refined = new Set(); let refining = false, refinedN = 0;
  // Conv alone: a cell's latent preview is all it gets, so it is made again whenever its inputs change. pvFor: the
  // epoch each cell's preview was made for. needPv: the cell needs a latent preview (first, or again).
  const pvFor = new Int32Array(N).fill(-1);
  const needPv = (i) => !T16.slot.has(i) || rePv.has(i) || (sh.convOnly && made[i] !== epoch && pvFor[i] !== epoch);
  function refine(cells) {
    const cell = cells.find((c) => made[c] === epoch && !refined.has(c)); if (cell === undefined) return;
    refining = true; const ep = epoch, v = ver[cell];
    env.refineReady()
      .then(() => { const { z, mix } = tileInputs(cell); return G.generate(z, Float32Array.from(G.classVec(mix)), sh.NOISE, 'flux2'); })
      .then((img) => (ep === epoch && v === ver[cell] ? store(cell, img).then(() => { refined.add(cell); refinedN++; }) : null))
      .catch(env.refineFailed).finally(() => { refining = false; });
  }
  function pump() {
    if (!on || sh.held || sh.paused || G.lost || document.hidden) return;
    // While the view moves (drag, flick, arrow keys, a zoom under way) only latents are made: each is cheap and gives the
    // GPU back quickly, so the motion stays smooth; the decodes wait until 200 ms after it stops.
    const cheap = sh.cheapMove && G.preview && performance.now() - movedAt < 200;
    G.smooth = !!cheap;  // and each latent's GPU work is spread over frames (the lab's smooth mode)
    const slow = sh.pathSlow, slots = slow ? 1 : G.slots, pvSlots = !G.preview ? 0 : slow ? 1 : G.previewSlots;  // a path animating short of 30 frames a second goes first
    for (let k = 0; k < want.length && (inflight < slots || inflightP < pvSlots);) {
      const cell = want[k]; if (pending.has(cell)) { want.splice(k, 1); continue; }
      const pv = pvSlots > 0 && needPv(cell);
      if (!pv && sh.convOnly) { want.splice(k, 1); continue; }  // Conv alone: the latents are all there is
      if (!pv && cheap) { k++; continue; }  // the view is moving: decodes wait
      if (pv ? inflightP >= pvSlots : inflight >= slots) { k++; continue; }
      want.splice(k, 1); pending.add(cell); if (pv) inflightP++; else inflight++;
      if (!pv && made[cell] === epoch) remadeN++;  // a sharper copy of a cell already made (its larger one was evicted)
      // keep: the id the lab keeps this cell's preview latent under; it carries ver, so a stale latent is never decoded
      const ep = epoch, v = ver[cell], dec = G.options.decoder, { z, mix } = tileInputs(cell), c = Float32Array.from(G.classVec(mix)), keep = pvSlots ? cell + N * v : undefined;
      (pv ? G.preview(z, c, sh.NOISE, keep) : G.generate(z, c, sh.NOISE, undefined, keep))
        .then((r) => {
          env.ok(); if (ep !== epoch || v !== ver[cell]) return null;
          if (pv) { previewsN++; rePv.delete(cell); pvFor[cell] = ep; return made[cell] === ep ? null : storePreview(cell, sh.latRaw ? r.raw : r.fit); }
          return store(cell, r).then(() => { made[cell] = ep; madeN++; if (dec === 'flux2') { madeFlux++; refined.add(cell); } else refined.delete(cell); });
        })
        .catch(env.fail)
        .finally(() => { pending.delete(cell); if (pv) inflightP--; else inflight--; pump(); });
    }
  }
  // keepFull: the change does not alter the full FLUX.2 decoder's images, so the cells it made stay as they are
  function remake(keepFull = false) { epoch++; want = []; G.dropLatents(); rePv.clear(); if (keepFull) for (const c of refined) made[c] = epoch; else refined.clear(); }  // new inputs: every tile is made again; old images show until replaced
  let stale = 0, madeCells = 0;  // the last spin: cells holding its classes, of all cells with inputs so far
  function remakeClasses(cs) {  // these classes' noise turned: only the cells holding them are made again
    stale = 0; madeCells = 0;
    for (let i = 0; i < N; i++) {
      const m = mixCache[i]; if (!m) continue; madeCells++;
      if (!cs.some((c) => c in m)) continue;
      ver[i]++; made[i] = -1; pvFor[i] = -1; stale++; refined.delete(i);
    }
    want = [];
  }

  // ---- the frame ----
  const inst = new Uint32Array(MAXI * 4), CAPS = TIERS.map((T, t) => (t ? T.cap : Infinity)); let drawn = 0, inView = 0, lastInView = 0, centreCell = -1;
  // Close in, the cells reshape into a rectangular grid (gridTargets): fully from 200 CSS px tiles, not at all under 110,
  // smoothly between. gst keeps the cells' grid places from frame to frame.
  const NB = neighbours(g), morphArr = new Float32Array(MAXI * 4); let gst = {}, morphT = 0, gridded = 0;
  const smooth = (x) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };
  const WANT = 256, candD = new Float32Array(WANT), candI = new Int32Array(WANT);
  function frame() {
    if (!on) { looping = false; return; }  // another layout is showing: show() starts the frames again
    globeFling.step();
    const now = performance.now(), dtf = Math.min(1, (now - lastFrame) / 1000), kz = 1 - Math.exp(-(now - lastFrame) / 100); lastFrame = now;
    // each stage's clock runs while it has work in flight, as on the flat maps (for the readout's rates)
    if (inflight > 0) busy += dtf; if (inflightP > 0) busyP += dtf; if (inflight > 0 || inflightP > 0) busyAny += dtf;
    // arrow keys, held: the view moves 300 px a second, as on the flat maps (left: towards what is on the left)
    for (const k of sh.arrows) { const a = ARROWS[k], d = 300 * Math.min(0.1, dtf) / zoom; if (a) turn(-a[0] * d, -a[1] * d); }
    // while the pen is down near an edge the globe turns that way (up to 600 px a second), and the stroke follows the pen
    const ed = env.pen.edge(); if (ed) { const d = 600 * Math.min(0.1, dtf) / zoom; turn(-ed[0] * d, -ed[1] * d); }
    if (ed || sh.arrows.size) env.pen.carry();
    zoom = Math.abs(zoomGoal / zoom - 1) < 1e-4 ? zoomGoal : zoom * (zoomGoal / zoom) ** kz;
    const S = zoom * dpr, sx = 2 * S / W, sy = 2 * S / H;
    morphT = smooth((2 * meanRad * zoom - 110) / 90);
    const vis = morphT > 0 ? [] : null, vpos = morphT > 0 ? new Map() : null;
    if (pointers.size > 0 || globeFling.on || zoom !== zoomGoal || sh.arrows.size > 0) movedAt = now;
    let n = 0, best = -2, cn = 0; inView = 0; left = 0; leftP = 0;
    const wantRefine = !refining && now - movedAt > 200 && env.refineOK(), near2 = [];  // not while the globe moves (a flick, a zoom)
    const hw = W / 2, hh = H / 2;
    for (let i = 0; i < N; i++) {
      const x = g.centres[3 * i], y = g.centres[3 * i + 1], zc = g.centres[3 * i + 2];
      const vz = Rm[6] * x + Rm[7] * y + Rm[8] * zc; if (vz <= 0) continue;
      const vx = (Rm[0] * x + Rm[1] * y + Rm[2] * zc) * S, vy = (Rm[3] * x + Rm[4] * y + Rm[5] * zc) * S, rp = rad[i] * S;
      const mgn = rp * (1 + 3 * morphT);  // while reshaping, cells just past the edges too: drifting in, they fill them
      if (Math.abs(vx) > hw + mgn || Math.abs(vy) > hh + mgn) continue;
      inView++;
      const nPv = needPv(i);
      if (nPv) { leftP++; left++; } else if (!sh.convOnly && made[i] !== epoch) left++;
      if (vis) { vis.push(i); vpos.set(i, [vx, vy]); }
      if (vz > best) { best = vz; centreCell = i; }  // the point facing the viewer
      const cellPx = 2 * rp, need = tierFor(cellPx, lastInView, CAPS);
      const have = T256.slot.has(i) && need >= 2 ? 2 : T64.slot.has(i) && need >= 1 ? 1 : T16.slot.has(i) ? 0 : -1;
      if (wantRefine && have === 2) near2.push([vx * vx + vy * vy, i]);
      if (have >= 0 && n < MAXI) { const s = TIERS[have].slot.get(i); inst[4 * n] = i; inst[4 * n + 1] = have; inst[4 * n + 2] = s; n++; }
      // Conv alone: only cells that need a latent are wanted (the rest would fill the queue and never leave it)
      if ((sh.convOnly ? nPv : made[i] !== epoch || have < need) && !pending.has(i)) {
        // the nearest WANT to the centre, kept in order as they are found: no list of every candidate, no sort
        const d = vx * vx + vy * vy;
        if (cn < WANT || d < candD[cn - 1]) {
          let q = cn < WANT ? cn++ : cn - 1;
          for (; q > 0 && candD[q - 1] > d; q--) { candD[q] = candD[q - 1]; candI[q] = candI[q - 1]; }
          candD[q] = d; candI[q] = i;
        }
      }
    }
    drawn = n; lastInView = inView; gridded = 0;
    let gr = [1, 0, 0, 0];
    if (vis && centreCell >= 0) {
      const tg = gridTargets(vis, vpos, NB, gst, centreCell), L = Math.hypot(...gst.e1);
      // the view rolls (about its own axis) towards level rows, as fast as the morph has set in; the next frame's
      // positions follow
      const th = Math.atan2(gst.e1[1], gst.e1[0]);
      if (Math.abs(th) > 1e-4) { const a = -th * Math.min(1, 0.2 * morphT), c = Math.cos(a), s = Math.sin(a); Rm = mul([c, -s, 0, s, c, 0, 0, 0, 1], Rm); }
      gr = [gst.e1[0] / L, gst.e1[1] / L, 0.5 * L / S, (Math.sqrt(3) / 4) * L / S];
      for (let j = 0; j < n; j++) {
        const t = tg.get(inst[4 * j]); const w = t ? t.w : 0; if (w > 0) gridded++;
        morphArr[4 * j] = t ? t.B[0] / S : 0; morphArr[4 * j + 1] = t ? t.B[1] / S : 0; morphArr[4 * j + 2] = morphT * w; morphArr[4 * j + 3] = t ? t.slide : 0;
      }
    } else { gst = {}; morphArr.fill(0, 0, 4 * n); }
    dev.queue.writeBuffer(uniBuf, 0, new Float32Array([...Rm.slice(0, 3), 0, ...Rm.slice(3, 6), 0, ...Rm.slice(6, 9), 0, sx, sy, SIDE, 0, ...gr]));
    if (n) { dev.queue.writeBuffer(instBuf, 0, inst, 0, 4 * n); dev.queue.writeBuffer(morphBuf, 0, morphArr, 0, 4 * n); }
    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }] });
    if (n) { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.draw(18, n); }
    pass.end(); dev.queue.submit([enc.finish()]);
    want = Array.from(candI.subarray(0, cn));
    pump();
    if (wantRefine && !want.length && !pending.size && near2.length) { near2.sort((a, b) => a[0] - b[0]); refine(near2.map((c) => c[1])); }
    drawLabels(S);
    window.__globe = { drawn, inView, made: madeN, model: G.model, rot: Rm.slice(), zoom, epoch, psi: sh.psi, on, labels: nLabels, previews: previewsN, remade: remadeN, morph: morphT, gridded, gridAngle: gst.e1 ? (Math.atan2(gst.e1[1], gst.e1[0]) * 180) / Math.PI : 0, stale, madeCells };
    requestAnimationFrame(frame);
  }

  // Over the tiles: the globe's edge, a grid fixed to the globe (strongest far out, where it shows the sphere), and
  // class names at their points, nearest the view's centre first, each only where it overlaps no other.
  let nLabels = 0, labelDpr = 0; const labelWs = new Map();
  const labelW = (c, t) => { if (labelDpr !== dpr) { labelWs.clear(); labelDpr = dpr; } let w = labelWs.get(c); if (w === undefined) { w = lctx.measureText(t).width; labelWs.set(c, w); } return w; };  // measured once per name
  function drawLabels(S) {
    lctx.clearRect(0, 0, W, H);
    lctx.strokeStyle = 'rgba(255,255,255,0.25)'; lctx.lineWidth = dpr;
    lctx.beginPath(); lctx.arc(W / 2, H / 2, S, 0, 2 * Math.PI); lctx.stroke();
    const tilePx = 2 * meanRad * S;
    lctx.strokeStyle = `rgba(255,255,255,${0.08 + 0.22 * clamp((60 * dpr - tilePx) / (50 * dpr), 0, 1)})`;
    const line = (f) => { lctx.beginPath(); let on = false;
      for (let t = 0; t <= 360; t++) { const p = f(2 * Math.PI * t / 360), vz = Rm[6] * p[0] + Rm[7] * p[1] + Rm[8] * p[2];
        const x = W / 2 + S * (Rm[0] * p[0] + Rm[1] * p[1] + Rm[2] * p[2]), y = H / 2 - S * (Rm[3] * p[0] + Rm[4] * p[1] + Rm[5] * p[2]);
        if (vz > 0) { if (on) lctx.lineTo(x, y); else lctx.moveTo(x, y); on = true; } else on = false; }
      lctx.stroke(); };
    for (let k = 1; k < 6; k++) { const la = Math.PI * (k / 6 - 0.5); line((t) => [Math.cos(la) * Math.cos(t), Math.sin(la), Math.cos(la) * Math.sin(t)]); }
    for (let k = 0; k < 6; k++) { const lo = Math.PI * k / 6; line((t) => [Math.cos(t) * Math.cos(lo), Math.sin(t), Math.cos(t) * Math.sin(lo)]); }
    lctx.font = `600 ${13 * dpr}px system-ui, sans-serif`; lctx.textAlign = 'center'; lctx.textBaseline = 'middle';
    const pts = GF.points, cand = [], boxes = [], wr = wrap.getBoundingClientRect(); nLabels = 0;
    for (const id of ['stats', 'pathwin', 'minicap', 'gmini', 'rail', 'railtab']) {  // no label under the overlays
      const e = $(id); if (e.hidden || !e.offsetWidth) continue; const r = e.getBoundingClientRect();
      boxes.push([(r.left - wr.left) * dpr, (r.top - wr.top) * dpr, (r.right - wr.left) * dpr, (r.bottom - wr.top) * dpr]);
    }
    for (let c = 0; c < 1000; c++) {
      if (blocked.has(c)) continue;
      const x = pts[3 * c], y = pts[3 * c + 1], zc = pts[3 * c + 2];
      const vz = Rm[6] * x + Rm[7] * y + Rm[8] * zc; if (vz < 0.3) continue;
      const px = W / 2 + (Rm[0] * x + Rm[1] * y + Rm[2] * zc) * S, py = H / 2 - (Rm[3] * x + Rm[4] * y + Rm[5] * zc) * S;
      if (px < 0 || px > W || py < 0 || py > H) continue;
      cand.push([(px - W / 2) ** 2 + (py - H / 2) ** 2, c, px, py]);
    }
    cand.sort((a, b) => a[0] - b[0]);
    const gap = 6 * dpr, th = 20 * dpr;
    for (const [, c, px, py] of cand) {
      const t = G.names[c], tw = labelW(c, t) + 14 * dpr, b = [px - tw / 2 - gap, py - th / 2 - gap, px + tw / 2 + gap, py + th / 2 + gap];
      if (b[0] < 0 || b[1] < 0 || b[2] > W || b[3] > H || boxes.some((o) => b[0] < o[2] && o[0] < b[2] && b[1] < o[3] && o[1] < b[3])) continue;
      boxes.push(b); nLabels++;
      lctx.fillStyle = 'rgba(33,37,41,0.78)'; lctx.beginPath(); lctx.roundRect(px - tw / 2, py - th / 2, tw, th, th / 2); lctx.fill();
      lctx.fillStyle = '#fff'; lctx.fillText(t, px, py + 0.5 * dpr);
    }
    env.overlay(lctx, dpr);  // the page's path, over the labels
  }

  // ---- the noise globe (noiseglobe.js): this view is its host while it shows ----
  // These classes' noise turned (null: every class's): only the cells holding them are made again. A spin calls this on
  // every pointer event, and marking the cells is a loop over all of them, so it runs at most every 120 ms.
  let spunAt = 0, spunWait = false, spunSet = new Set(), spunAll = false;
  function spun(cs) {
    if (cs) for (const c of cs) spunSet.add(c); else spunAll = true;
    const go = () => {
      spunWait = false; if (!spunAll && !spunSet.size) return;  // already done by a call that came first
      spunAt = performance.now(); if (spunAll) remake(); else remakeClasses([...spunSet]); spunSet = new Set(); spunAll = false;
    };
    if (performance.now() - spunAt > 120) go(); else if (!spunWait) { spunWait = true; setTimeout(go, 120); }
  }
  const ngHost = {
    centre: () => (centreCell < 0 ? null : { mix: mixAt(centreCell), q: NG.field(Rm.slice(6, 9)), id: centreCell, z0: () => z0At(centreCell) }),
    spun(cs) { spun(cs); env.spun(cs, true); },
    idle: () => !inflight && !inflightP && !want.length,
  };

  // ---- what the page drives ----
  return {
    get model() { return model; },
    get n10() { return GF.n10; },
    show() {  // each loop is started once: a hide and show within one frame must not start a second
      if (on) return; on = true; lastFrame = performance.now();
      if (!looping) { looping = true; requestAnimationFrame(frame); }
      NG.attach(ngHost);
    },
    hide() { on = false; want = []; NG.detach(ngHost); },
    remake,
    spun,  // the noise globe was spun while another layout showed
    settings(what) {  // a setting changed ('soft', 'span', 'seed' or 'all'): drop what it feeds; every tile is made again
      if (what === 'soft' || what === 'all') { mixCache = new Array(N); ancMix = new Map(); }
      if ((what === 'span' || what === 'all') && anchorLevel() !== gLevel) {
        gLevel = anchorLevel(); let a = anchorTables.get(gLevel);
        if (!a) { const t = geodesic(7, gLevel); a = { anchorOf: t.anchorOf, anchorTri: t.anchorTri }; anchorTables.set(gLevel, a); }  // built once per stop (0.8 s), then kept
        g.anchorOf = a.anchorOf; g.anchorTri = a.anchorTri; mixCache = new Array(N);
      }
      if (what === 'seed' || what === 'all') { qCache = new Array(N); ancZ = new Map(); }  // the page gives the noise globe its new field first
      remake();
    },
    async loadModel(code) {  // each model places the classes its own way: its files, and a globe made afresh
      if (code === model) return;
      GF = await globeFiles(code); model = code;
      mixCache = new Array(N); ancMix = new Map(); epoch++; clearTiles(); refined.clear(); rePv.clear();
    },
    // for the page's path tool: the point of the globe under a canvas position (null: off the globe), where a point
    // shows ([x, y] CSS px in the canvas, and its depth: > 0 facing the viewer), a frame's inputs there, the tile sizes
    pick(px, py) {
      const x = (px - cw / 2) / zoom, y = -(py - ch / 2) / zoom, r2 = x * x + y * y; if (r2 >= 1) return null;
      const zc = Math.sqrt(1 - r2); return [Rm[0] * x + Rm[3] * y + Rm[6] * zc, Rm[1] * x + Rm[4] * y + Rm[7] * zc, Rm[2] * x + Rm[5] * y + Rm[8] * zc];
    },
    project: (p) => [cw / 2 + (Rm[0] * p[0] + Rm[1] * p[1] + Rm[2] * p[2]) * zoom, ch / 2 - (Rm[3] * p[0] + Rm[4] * p[1] + Rm[5] * p[2]) * zoom, Rm[6] * p[0] + Rm[7] * p[1] + Rm[8] * p[2]],
    inputsAt,
    tileAngle: 2 * meanRad,  // a tile's size on the unit sphere
    tilePx: () => 2 * meanRad * zoom,
    radius: () => zoom,  // the globe's radius, CSS px
    get epoch() { return epoch; },
    zoomBy(f) { zoomGoal = clamp(zoomGoal * f, fitR() * 0.5, maxZoom()); },
    zoomPct: () => 200 * meanRad * zoom / S100,
    zoomReset() { zoomGoal = clamp(S100 / (2 * meanRad), fitR() * 0.5, maxZoom()); },
    teleport() {  // a random turn: any point of the globe, equally likely
      const u = 2 * Math.random() - 1, t = 2 * Math.PI * Math.random(), r = Math.sqrt(1 - u * u);
      Rm = facing([r * Math.cos(t), r * Math.sin(t), u]); globeFling.down();  // and a flick under way stops
    },
    repreview() { for (const c of T16.slot.keys()) if (made[c] !== epoch) rePv.add(c); },  // Conv on or off: the latents shown, in the other colouring
    counters: () => ({ viewTotal: inView, left, leftP, made: madeN, madeFlux, previews: previewsN, busy, busyP, busyAny, refinedN, refining }),
    writeURL(q) {
      const r = (x) => +x.toFixed(4);
      q.set('rot', Rm.map(r).join(',')); q.set('zoom', (2 * meanRad * zoom).toFixed(2));
    },
  };
}
