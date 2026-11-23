// The noise globe of mapv2.html, for every layout (off by default: Advanced > Noise). On, each tile's noise along its
// class's 3 strongest directions comes from a point of a small globe: a smooth random field over the map gives each
// tile a point, and each class turns that point by its own rotation. Spinning the small globe turns the rotation of
// the class its caption names (the class at the view's centre); with Shift, of every class. Off, tiles keep their
// purely random noise and nothing here runs.
import { geodesic, directions, tileZ, noiseField, noisePoint } from './globe.js?v=20261122m';

const unit = (v) => { const r = Math.hypot(...v) || 1; return v.map((x) => x / r); };
const mul = (A, B) => [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]));
const tr = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const ortho = (M) => {  // re-orthonormalise rows against drift
  const r0 = unit(M.slice(0, 3)), r1u = M.slice(3, 6), d = r1u.reduce((s, x, i) => s + x * r0[i], 0), r1 = unit(r1u.map((x, i) => x - d * r0[i]));
  return [...r0, ...r1, r0[1] * r1[2] - r0[2] * r1[1], r0[2] * r1[0] - r0[0] * r1[2], r0[0] * r1[1] - r0[1] * r1[0]];
};
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const mixKey = (mix) => Object.keys(mix).map((k) => `${k}:${mix[k].toFixed(3)}`).join(',');
// A flick, as on the maps: the drag's velocity (smoothed, px/s) carries on after the release and dies away (e^-4t); a
// release more than 80 ms after the last move does not fling.
function flinger(apply) {
  let vx = 0, vy = 0, t = 0, on = false, last = 0;
  return {
    down() { vx = vy = 0; on = false; t = performance.now(); },
    move(dx, dy) { const now = performance.now(), dt = Math.max(1, now - t) / 1000; vx = 0.8 * vx + 0.2 * dx / dt; vy = 0.8 * vy + 0.2 * dy / dt; t = now; },
    up() { on = performance.now() - t <= 80 && Math.hypot(vx, vy) > 30; last = performance.now(); },
    step() {
      const now = performance.now(), dt = Math.min(0.05, (now - last) / 1000); last = now; if (!on) return;
      apply(vx * dt, vy * dt); const k = Math.exp(-4 * dt); vx *= k; vy *= k; if (Math.hypot(vx, vy) < 4) on = false;
    },
  };
}

// env: G (the lab), Q (the link's parameters), $ (element by id), GD (the folder of each model's noise directions),
// sh (the page's settings, live: psi, noiseSeed, noiseStr, NOISE, held, paused, latRaw, convOnly).
// A layout that shows attaches itself as the host: { centre(): null or { mix, q, id, z0() } (the class mix, field
// point, an id and the base z at the view's centre), spun(classes or null) (these classes' noise turned, null: every
// class's; the host makes their tiles again), idle() (the layout has no work of its own) }.
export function createNoiseGlobe(env) {
  const { G, Q, $, GD, sh } = env;
  const wrap = $('wrap'), mini = $('gmini'), mctx = mini.getContext('2d');
  let on = false, host = null, hiddenByM = false, model = null, axes = null, R0 = null, dirCache = new Map();

  // ---- each model's noise directions: per class, its 8 strongest directions of variation (noiseaxes.bin) ----
  async function loadModel(code) {
    if (code === model) return;
    const pre = GD + `imagenet_${code}_`;
    const [aj, ab] = await Promise.all([fetch(pre + 'noiseaxes.json').then((r) => r.json()), fetch(pre + 'noiseaxes.bin').then((r) => r.arrayBuffer())]);
    const rec = new Float32Array(ab), C = rec.length / 521;
    if (C !== 1000) throw new Error(`noiseaxes.bin holds ${C} classes, not 1000`);
    const a = { trace: new Float32Array(C), lambda: new Float32Array(C * 8), U: new Float32Array(C * 512) };
    for (let c = 0; c < C; c++) { a.trace[c] = rec[c * 521]; a.lambda.set(rec.subarray(c * 521 + 1, c * 521 + 9), c * 8); a.U.set(rec.subarray(c * 521 + 9, c * 521 + 521), c * 512); }
    axes = a; R0 = Float32Array.from(aj.R0.flat()); model = code; dirCache = new Map();
  }
  function dirsFor(mix) {
    const key = mixKey(mix); let d = dirCache.get(key);
    if (!d) { d = directions(mix, axes, R0); dirCache.set(key, d); if (dirCache.size > 20000) dirCache.delete(dirCache.keys().next().value); }
    return d;
  }

  // ---- the rotations: Mm turns every class's noise; Mc holds a class whose noise was spun on its own. The link carries
  // both (nm; nc=class:9 numbers;...). A tile blending classes blends their turned points by its class weights, so a
  // spin reaches into the next class only through the border tiles. ----
  let NF = noiseField(sh.noiseSeed), Mm = I3.slice();
  const Mc = new Map();
  const nm = (Q.get('nm') || '').split(',').map(Number);
  if (nm.length === 9 && nm.every(Number.isFinite)) Mm = ortho(nm);
  for (const part of (Q.get('nc') || '').split(';')) {
    const [c, m] = part.split(':'), M = (m || '').split(',').map(Number);
    if (Number.isInteger(+c) && +c >= 0 && +c < G.names.length && M.length === 9 && M.every(Number.isFinite)) Mc.set(+c, ortho(M));
  }
  function noiseOf(mix, q) {
    const n = [0, 0, 0];
    for (const c in mix) { const M = Mc.get(+c) ?? Mm, w = mix[c]; for (let t = 0; t < 3; t++) n[t] += w * (M[t] * q[0] + M[3 + t] * q[1] + M[6 + t] * q[2]); }
    return unit(n);
  }
  const ready = () => on && axes && model === G.model;
  // A tile's z from its base z0 (N(0, I) at the typical length): on, its part along the class mix's 3 directions is
  // the noise point (tileZ); off, z0 as it is. Either way scaled by psi.
  function z(z0, mix, q) {
    if (ready()) return tileZ(dirsFor(mix).V, noiseOf(mix, q), sh.psi, z0);
    const out = new Float32Array(z0.length); for (let k = 0; k < out.length; k++) out[k] = sh.psi * z0[k];
    return out;
  }
  // the classes the caption names, whose noise a spin turns: the strongest at the view centre, and the next if >= 0.3
  const named = (mix) => { const top = Object.keys(mix).sort((a, b) => mix[b] - mix[a]).map(Number); return top.length > 1 && mix[top[1]] >= 0.3 ? top.slice(0, 2) : top.slice(0, 1); };

  // ---- the small globe. It is drawn turned by Vm, which only a drag on it changes: it never moves on its own.
  // Underneath, Dm turns, unseen, so the noise at the view's centre is always at the front, under the ring: the point
  // drawn at s holds the noise Dm^T s. Moving the map changes what the small globe's tiles hold, not where it points. ----
  let Dm = I3.slice(), Vm = I3.slice(), cen = null, cenAt = -1;
  const centre = () => {  // the host's view centre, asked for at most every 8 ms
    const now = performance.now(); if (now - cenAt > 8) { cenAt = now; cen = host ? host.centre() : null; }
    return cen;
  };
  const noiseCentre = () => { const c = centre(); return c ? noiseOf(c.mix, c.q) : null; };
  function faceMini() {  // the smallest turn of Dm that brings the centre's noise to the front
    const n = noiseCentre(); if (!n) return;
    const v = [0, 1, 2].map((a) => Dm[3 * a] * n[0] + Dm[3 * a + 1] * n[1] + Dm[3 * a + 2] * n[2]);
    if (v[2] > 1 - 1e-9) return;
    if (v[2] < -0.9999) { Dm = mul([1, 0, 0, 0, -1, 0, 0, 0, -1], Dm); return; }
    const k = [v[1], -v[0], 0], K = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0], K2 = mul(K, K), f = 1 / (1 + v[2]);
    Dm = ortho(mul([0, 1, 2, 3, 4, 5, 6, 7, 8].map((j) => (j % 4 === 0) + K[j] + K2[j] * f), Dm));
  }
  let mAll = false;  // this drag (and its flick) turns every class's noise: Shift was down at its start
  function spinMini(dx, dy) {  // a drag (or flick) of dx, dy CSS px
    const c = centre(); if (!c || !ready()) return;
    const r = mini.clientWidth / 2, ax = dx / r, ay = dy / r;
    const cy = Math.cos(ax), sy = Math.sin(ax), cx = Math.cos(ay), sx = Math.sin(ay);
    const R = mul([1, 0, 0, 0, cx, -sx, 0, sx, cx], [cy, 0, sy, 0, 1, 0, -sy, 0, cy]);
    // the drawn globe turns by R; the view centre's noise becomes the point now at the front: M <- M Dm^T R Dm, for the
    // named classes (each from its own rotation, or Mm's), or with Shift for every class
    const X = mul(tr(Dm), mul(R, Dm));
    if (mAll) { Mm = ortho(mul(Mm, X)); for (const [k, M] of Mc) Mc.set(k, ortho(mul(M, X))); host.spun(null); }
    else { const cs = named(c.mix); for (const k of cs) Mc.set(k, ortho(mul(Mc.get(k) ?? Mm, X))); host.spun(cs); }
    Dm = ortho(mul(R, Dm)); Vm = ortho(mul(R, Vm)); cenAt = -1;
  }
  const miniFling = flinger(spinMini);
  let mdrag = null;
  mini.addEventListener('pointerdown', (e) => { mini.setPointerCapture(e.pointerId); mdrag = [e.clientX, e.clientY]; mAll = e.shiftKey; miniFling.down(); });
  mini.addEventListener('pointermove', (e) => {
    if (!mdrag) return;
    const dx = e.clientX - mdrag[0], dy = e.clientY - mdrag[1]; mdrag = [e.clientX, e.clientY];
    spinMini(dx, dy); miniFling.move(dx, dy);
  });
  const mup = () => { if (mdrag) miniFling.up(); mdrag = null; }; mini.addEventListener('pointerup', mup); mini.addEventListener('pointercancel', mup);

  // Its tiles: 162 points of the noise sphere, for the class mix at the view's centre, which its caption names. When
  // that mix changes, the old mix's tiles fade out and the new ones fill in.
  const mg = geodesic(2); let miniImg = [], miniKey = '', miniMix = '', miniMixSince = 0;
  const miniMixOf = [];
  const miniRad = Array.from({ length: mg.n }, (_, i) => {  // each cell's circumradius
    let m = 0; for (let j = mg.cellStart[i]; j < mg.cellStart[i + 1]; j++) m = Math.max(m, Math.hypot(...[0, 1, 2].map((t) => mg.corners[3 * j + t] - mg.centres[3 * i + t])));
    return m;
  });
  $('minisize').onclick = () => {
    const big = wrap.classList.toggle('minibig'), b = $('minisize'), t = big ? 'Smaller noise globe' : 'Larger noise globe';
    b.setAttribute('aria-pressed', String(big)); b.title = t; b.setAttribute('aria-label', t);
  };
  let looping = false;
  const vis = () => on && !!host && !hiddenByM;
  function drawMini() {
    if (!vis()) { looping = false; return; }
    const dpr = devicePixelRatio || 1, s = Math.round(mini.clientWidth * dpr); if (mini.width !== s || mini.height !== s) { mini.width = s; mini.height = s; }
    miniFling.step();
    const r = s / 2 - 2 * dpr; mctx.clearRect(0, 0, s, s); faceMini(); const D = Vm;
    const gr = mctx.createRadialGradient(s / 2 - r / 3, s / 2 - r / 3, r / 10, s / 2, s / 2, r);
    gr.addColorStop(0, '#3a4a66'); gr.addColorStop(1, '#0e1420');
    mctx.fillStyle = gr; mctx.beginPath(); mctx.arc(s / 2, s / 2, r, 0, 2 * Math.PI); mctx.fill();
    // a coordinate grid fixed to the noise globe, so its turning shows
    mctx.strokeStyle = 'rgba(255,255,255,0.28)'; mctx.lineWidth = dpr;
    const line = (f) => { mctx.beginPath(); let pen = false;
      for (let t = 0; t <= 96; t++) { const p = f(2 * Math.PI * t / 96), vz = D[6] * p[0] + D[7] * p[1] + D[8] * p[2];
        const x = s / 2 + r * (D[0] * p[0] + D[1] * p[1] + D[2] * p[2]), y = s / 2 - r * (D[3] * p[0] + D[4] * p[1] + D[5] * p[2]);
        if (vz > 0) { if (pen) mctx.lineTo(x, y); else mctx.moveTo(x, y); pen = true; } else pen = false; }
      mctx.stroke(); };
    for (let k = 1; k < 6; k++) { const la = Math.PI * (k / 6 - 0.5); line((t) => [Math.cos(la) * Math.cos(t), Math.sin(la), Math.cos(la) * Math.sin(t)]); }
    for (let k = 0; k < 6; k++) { const lo = Math.PI * k / 6; line((t) => [Math.cos(t) * Math.cos(lo), Math.sin(t), Math.cos(t) * Math.sin(lo)]); }
    // each tile is a cell of the sphere with its image laid upright on the cell's tangent plane and shaded
    const rot = (p) => [D[0] * p[0] + D[1] * p[1] + D[2] * p[2], D[3] * p[0] + D[4] * p[1] + D[5] * p[2], D[6] * p[0] + D[7] * p[1] + D[8] * p[2]];
    const fade = Math.max(0, 1 - (performance.now() - miniMixSince) / 500);  // another class's tiles fade out over 0.5 s
    for (let i = 0; i < mg.n; i++) {
      const img = miniImg[i]; if (!img) continue;
      const a = miniMixOf[i] === miniMix ? 1 : fade; if (a <= 0) continue;
      mctx.globalAlpha = a;
      const c = rot([mg.centres[3 * i], mg.centres[3 * i + 1], mg.centres[3 * i + 2]]); if (c[2] <= 0) continue;
      let up = [-c[1] * c[0], 1 - c[1] * c[1], -c[1] * c[2]]; if (Math.hypot(...up) < 1e-4) up = [-c[2] * c[0], -c[2] * c[1], 1 - c[2] * c[2]];
      up = unit(up); const rt = [up[1] * c[2] - up[2] * c[1], up[2] * c[0] - up[0] * c[2], up[0] * c[1] - up[1] * c[0]];
      const R = miniRad[i], w = img.width, k = (r * 2 * R) / w;
      mctx.save(); mctx.beginPath();
      for (let j = mg.cellStart[i]; j < mg.cellStart[i + 1]; j++) { const q = rot([mg.corners[3 * j], mg.corners[3 * j + 1], mg.corners[3 * j + 2]]); mctx.lineTo(s / 2 + r * q[0], s / 2 - r * q[1]); }
      mctx.closePath(); mctx.clip();
      mctx.setTransform(k * rt[0], -k * rt[1], -k * up[0], k * up[1], s / 2 + r * (c[0] - R * rt[0] + R * up[0]), s / 2 - r * (c[1] - R * rt[1] + R * up[1]));
      mctx.drawImage(img, 0, 0); mctx.setTransform(1, 0, 0, 1, 0, 0);
      mctx.fillStyle = `rgba(0,0,0,${0.45 * (1 - c[2])})`; mctx.fill(); mctx.restore();
    }
    mctx.globalAlpha = 1;
    // the ring: the noise at the view's centre, which the small globe always turns to its front
    mctx.strokeStyle = '#E8B04A'; mctx.lineWidth = 2 * dpr;
    mctx.beginPath(); mctx.arc(s / 2, s / 2, 6 * dpr, 0, 2 * Math.PI); mctx.stroke();
    requestAnimationFrame(drawMini);
  }
  // A tile drawn at Vm y holds the noise Dm^T Vm y, so its key carries that turn (rounded: a small turn keeps the
  // tiles; a drag turns Dm and Vm together, so it keeps them all). Every tile, back included, gets its Conv-decoded
  // latent (32 px, cheap), 3 at a time even while the map has work, so the small globe is whole and can be spun at
  // once; a tile is decoded with TAEF2 only when the map has nothing to do. Front first; old tiles stay drawn until
  // replaced. While the map moves, the small globe holds what it has: a new key (class or noise) is taken only once it
  // has stayed the same for SETTLE_MS, so a move across classes causes one switch, not many.
  const miniOf = [], miniDec = [], miniPend = new Set(), MINI_SLOTS = 3, SETTLE_MS = 400;
  let nextKey = '', nextSince = 0;
  function miniPump() {
    if (!vis() || !ready() || sh.held || sh.paused || G.lost || document.hidden || miniPend.size >= MINI_SLOTS) return;
    const c = centre(); if (!c) return;
    const H = mul(tr(Dm), Vm), mix = c.mix, mk = mixKey(mix);
    const key = [mk, c.id, sh.psi, sh.noiseSeed, sh.noiseStr, G.model, G.options.half, G.options.gHalf, sh.latRaw, H.map((x) => Math.round(x * 20)).join()].join('|');  // id: its base z
    if (key !== miniKey) {
      if (key !== nextKey) { nextKey = key; nextSince = performance.now(); }
      if (performance.now() - nextSince < SETTLE_MS) return;
    }
    miniKey = key;
    if (mk !== miniMix) {
      miniMix = mk; miniMixSince = performance.now();
      const top = named(mix);
      $('minicapt').textContent = `Noise for ${G.names[top[0]]}${top.length > 1 ? ` and ${G.names[top[1]]}` : ''}: spin to change`;
    }
    const idle = host.idle(), front = (k) => Vm[6] * mg.centres[3 * k] + Vm[7] * mg.centres[3 * k + 1] + Vm[8] * mg.centres[3 * k + 2];
    let i = -1, best = -2, dec = false;
    for (let k = 0; k < mg.n; k++) if (miniOf[k] !== key && !miniPend.has(k) && front(k) > best) { best = front(k); i = k; }
    // the full decoder is too slow for 162 small tiles; Conv alone: latents are all there is
    if (i < 0 && idle && !miniPend.size && !sh.convOnly && G.options.decoder !== 'flux2') { dec = true; for (let k = 0; k < mg.n; k++) if (!miniDec[k] && front(k) > best) { best = front(k); i = k; } }
    if (i < 0) return;
    const pv = !dec && !!G.preview;
    miniPend.add(i);
    const y = [mg.centres[3 * i], mg.centres[3 * i + 1], mg.centres[3 * i + 2]], q = [0, 1, 2].map((a) => H[3 * a] * y[0] + H[3 * a + 1] * y[1] + H[3 * a + 2] * y[2]);
    const zz = tileZ(dirsFor(mix).V, q, sh.psi, c.z0()), cv = Float32Array.from(G.classVec(mix));
    (pv ? G.preview(zz, cv, sh.NOISE).then((r) => (sh.latRaw ? r.raw : r.fit)) : G.generate(zz, cv, sh.NOISE))
      .then((img) => createImageBitmap(img, pv ? {} : { resizeWidth: 64, resizeHeight: 64, resizeQuality: 'high' }))
      .then((b) => { if (key === miniKey) { if (miniImg[i]) miniImg[i].close(); miniImg[i] = b; miniOf[i] = key; miniDec[i] = !pv; miniMixOf[i] = mk; } else b.close(); })
      .catch((e) => console.error(e)).finally(() => { miniPend.delete(i); miniPump(); });
    miniPump();
  }
  setInterval(miniPump, 50);

  function refresh() {  // shown while it is on, a layout is attached and M has not hidden it
    const v = vis();
    for (const id of ['gmini', 'minicap', 'minisize']) $(id).hidden = !v;
    wrap.classList.toggle('ng', v);
    if (v && !looping) { looping = true; requestAnimationFrame(drawMini); }
  }
  // The named classes' rotations as a code to read and copy: each as a rotation vector in degrees (axis times angle).
  function code() {
    const c = centre(); if (!ready() || !c) return '';
    const deg = (M) => {
      const a = Math.acos(Math.max(-1, Math.min(1, (M[0] + M[4] + M[8] - 1) / 2))), s = 2 * Math.sin(a);
      const v = a < 1e-6 ? [0, 0, 0] : Math.abs(s) < 1e-6 ? [180, 0, 0] : [M[7] - M[5], M[2] - M[6], M[3] - M[1]].map((x) => (x / s) * a * 180 / Math.PI);
      return v.map((x) => (Math.abs(x) < 0.05 ? '0' : x.toFixed(1))).join(', ');
    };
    return named(c.mix).map((k) => `${G.names[k]}: ${deg(Mc.get(k) ?? Mm)}${Mc.has(k) ? '' : ' (as every class)'}`).join('; ');
  }
  return {
    get on() { return on; },
    get model() { return model; },
    loadModel,
    async setOn(b) { if (b && model !== G.model) await loadModel(G.model); on = b; cenAt = -1; refresh(); },
    attach(h) { host = h; cenAt = -1; refresh(); },
    detach(h) { if (host === h) { host = null; refresh(); } },
    setHidden(h) { hiddenByM = h; refresh(); },
    seed() { NF = noiseField(sh.noiseSeed); },  // a new noise seed: a new field (the hosts drop the points they cached)
    field: (p) => noisePoint(NF, p),
    z,
    lineHTML() {  // for the readout: the class at the centre and how much its 3 directions carry
      const c = centre(); if (!ready() || !c) return '';
      const top = Object.keys(c.mix).sort((a, b) => c.mix[b] - c.mix[a])[0];
      return `<div class="cap" style="white-space: normal">Centre: ${G.names[top]}. These 3 noise directions carry ${Math.round(100 * dirsFor(c.mix).share)}% of the variation here${sh.psi !== 1 ? ' (measured at ψ = 1)' : ''}.</div>`;
    },
    code,
    writeURL(q) {
      const r = (x) => +x.toFixed(4);
      if (on) q.set('ng', 1); else q.delete('ng');
      if (Mm.some((x, i) => Math.abs(x - I3[i]) > 1e-6)) q.set('nm', Mm.map(r).join(',')); else q.delete('nm');
      if (Mc.size) q.set('nc', [...Mc].map(([c, M]) => `${c}:${M.map(r).join(',')}`).join(';')); else q.delete('nc');
    },
    state: () => ({ on, shown: vis(), tiles: miniImg.filter(Boolean).length, n: mg.n, Mm: Mm.slice(), spun: Object.fromEntries(Mc), view: Vm.slice(), front: Dm.slice(6, 9), centre: noiseCentre(), caption: $('minicapt').textContent }),
  };
}
