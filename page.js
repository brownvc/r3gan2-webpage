// index.html: the samples (the paper's Figure 8), the "waiting" speed widget, and the live generator's frame.
const $ = (id) => document.getElementById(id);

// The shared bar lists every h2; end matter (acknowledgements, references) stays off it. The bar arrives after this
// script, so watch for its links.
const trimNav = () => {
  const end = document.querySelectorAll('#navmenu a[href="#acknowledgements"], #navmenu a[href="#references"]');
  end.forEach((a) => a.remove());
  return end.length > 0;
};
if (!trimNav()) {
  const mo = new MutationObserver(() => { if (trimNav()) mo.disconnect(); });
  mo.observe(document.body, { childList: true, subtree: true });
}

// Paper, Table F.2. tf: TFLOPs per image, with guidance and the decoder; passes: network evaluations per image;
// pub: FID as published, the rest from our evaluation runs; n: samples in images/samples/<key>/207_<nn>.jpg;
// params: parameter count; type: the kind of model; ref: its entry in REFS.
// guided: the FID is with guidance; unguided: the published FID without it, where there is one.
const M = {
  'r3gan2-s':        { name: 'R3GAN2-S',      params: '17 M', type: 'GAN',            tf: 0.64,   fid: 2.12, passes: 1,   n: 32, ours: true },
  'r3gan2-m':        { name: 'R3GAN2-M',      params: '61 M', type: 'GAN',            tf: 0.67,   fid: 1.60, passes: 1,   n: 32, ours: true },
  'r3gan2-l':        { name: 'R3GAN2-L',      params: '131 M', type: 'GAN',            tf: 0.73,   fid: 1.52, passes: 1,   n: 32, ours: true },
  'gat':             { name: 'GAT-XL/2',      params: '602 M', type: 'GAN',            tf: 0.86,   fid: 2.11, passes: 1,   n: 16, ref: 'gat' },
  'alphaflow-2step': { name: 'α-Flow-XL/2+',  params: '676 M', type: 'Flow map',      tf: 1.10,   fid: 2.06, passes: 2,   n: 16, ref: 'alphaflow', guided: true },
  'stylegan-xl':     { name: 'StyleGAN-XL',   params: '166 M', type: 'GAN',            tf: 3.14,   fid: 2.34, passes: 1,   n: 16, ref: 'stylegan-xl', guided: true },
  'var-d30':         { name: 'VAR-d30',       params: '2,000 M', type: 'Autoregressive', tf: 5.54,   fid: 1.95, passes: 20,  n: 16, ref: 'var', guided: true },
  'dit':             { name: 'DiT-XL/2',      params: '675 M', type: 'Diffusion',      tf: 119.12, fid: 2.27, passes: 500, n: 16, pub: true, ref: 'dit', guided: true, unguided: '9.62' },
  'facm':            { name: 'FACM-XL/2',     params: '675 M', type: 'Consistency',    tf: 0.86,   fid: 1.72, passes: 1,   n: 16, ref: 'facm', guided: true },
  'imf-xl-2step':    { name: 'iMF-XL/2',      params: '610 M', type: 'Flow map',      tf: 1.32,   fid: 1.61, passes: 2,   n: 16, ref: 'imf', guided: true },
  'imf-xl':          { name: 'iMF-XL/2',      params: '610 M', type: 'Flow map',      tf: 0.97,   fid: 1.82, passes: 1,   n: 0, ref: 'imf', guided: true },
  'imf-xl-noguid':   { name: 'iMF-XL/2',      params: '610 M', type: 'Flow map',      tf: 0.97,   fid: 10.61, passes: 1,  n: 0, ref: 'imf', guidanceOff: true },
  'rar-xxl':         { name: 'RAR-XXL',       params: '1,500 M', type: 'Autoregressive', tf: 1.73,   fid: 1.49, passes: 512, n: 16, ref: 'rar', guided: true },
  'rae-noguid':      { name: 'RAE',           params: '839 M', type: 'Diffusion',      tf: 14.65,  fid: 1.51, passes: 50,  n: 16, ref: 'rae' },
  'repa':            { name: 'SiT-XL/2+REPA', params: '675 M', type: 'Diffusion',      tf: 100.64, fid: 1.42, passes: 500, n: 16, pub: true, ref: 'repa', guided: true, unguided: '5.90' },
  'mar-h':           { name: 'MAR-H',         params: '943 M', type: 'Autoregressive', tf: 189.52, fid: 1.55, passes: 512, n: 16, pub: true, ref: 'mar', guided: true },
};
// The references, in the order the page first cites them; the list at the bottom of the page is drawn from this.
// The last field is an arXiv id, or a full URL for work that is not on arXiv.
const REFS = {
  dit: ['W. Peebles, S. Xie', 'Scalable Diffusion Models with Transformers', 'ICCV 2023', '2212.09748'],
  imf: ['Z. Geng, Y. Lu, Z. Wu, E. Shechtman, J. Z. Kolter, K. He', 'Improved Mean Flows: On the Challenges of Fastforward Generative Models', 'CVPR 2026', '2512.02012'],
  facm: ['Y. Peng, K. Zhu, Y. Liu, P. Wu, H. Li, X. Sun, F. Wu', 'FACM: Flow-Anchored Consistency Models', 'ICLR 2026', '2507.03738'],
  gat: ['S. Hyun, M. Lee, J.-P. Heo', 'Scalable GANs with Transformers', 'ICML 2026', '2509.24935'],
  'stylegan-xl': ['A. Sauer, K. Schwarz, A. Geiger', 'StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets', 'SIGGRAPH 2022', '2202.00273'],
  taesd: ['O. Boer Bohan', 'TAESD: Tiny AutoEncoder for Stable Diffusion', 'GitHub, 2023', 'https://github.com/madebyollin/taesd'],
  taef2: ['O. Boer Bohan', 'TAEF2: Tiny AutoEncoder for FLUX.2', 'Hugging Face, 2026', 'https://huggingface.co/madebyollin/taef2'],
  alphaflow: ['H. Zhang, A. Siarohin, W. Menapace, M. Vasilkovsky, S. Tulyakov, Q. Qu, I. Skorokhodov', 'AlphaFlow: Understanding and Improving MeanFlow Models', 'ICLR 2026', '2510.20771'],
  var: ['K. Tian, Y. Jiang, Z. Yuan, B. Peng, L. Wang', 'Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction', 'NeurIPS 2024', '2404.02905'],
  rar: ['Q. Yu, J. He, X. Deng, X. Shen, L.-C. Chen', 'Randomized Autoregressive Visual Generation', 'ICCV 2025', '2411.00776'],
  rae: ['B. Zheng, N. Ma, S. Tong, S. Xie', 'Diffusion Transformers with Representation Autoencoders', 'ICLR 2026', '2510.11690'],
  repa: ['S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, S. Xie', 'Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think', 'ICLR 2025', '2410.06940'],
  mar: ['T. Li, Y. Tian, H. Li, M. Deng, K. He', 'Autoregressive Image Generation without Vector Quantization', 'NeurIPS 2024', '2406.11838'],
};
const refNo = (k) => Object.keys(REFS).indexOf(k) + 1;
const cite = (k) => (k ? ` <a class="ref" href="#ref-${k}">[${refNo(k)}]</a>` : '');
$('refs').innerHTML = Object.entries(REFS).map(([k, [who, title, venue, arxiv]], i) =>
  `<li id="ref-${k}"><span class="n">[${i + 1}]</span> ${who}. <a href="${arxiv.startsWith('http') ? arxiv : `https://arxiv.org/abs/${arxiv}`}">${title}</a>. ${venue}.</li>`).join('');
document.querySelectorAll('[data-ref]').forEach((a) => { a.href = `#ref-${a.dataset.ref}`; a.textContent = `[${refNo(a.dataset.ref)}]`; a.className = 'ref'; });
// Several works at one mark: <span data-refs="imf facm"></span> prints [2, 3], each number its own link.
document.querySelectorAll('[data-refs]').forEach((s) => { s.innerHTML = '[' + s.dataset.refs.split(' ').map((k) => `<a class="ref" href="#ref-${k}">${refNo(k)}</a>`).join(', ') + ']'; s.className = 'ref'; });
// Every label: the name; then type and size; then FID and guidance only. The narrow speed panels give each fact its
// own line (type, parameters, passes, FID), so the rows line up across panels like a table.
// Non-breaking spaces keep a number with its unit when a narrow label wraps.
const size = (m) => `${M[m].type} · ${M[m].params} params`;
const passes = (m) => `${M[m].passes} pass${M[m].passes > 1 ? 'es' : ''}`;
const fid = (m) => `FID ${M[m].fid.toFixed(2)}` + (M[m].guided ? ', guided' : M[m].guidanceOff ? ', unguided' : '') +
  (M[m].unguided ? ` (${M[m].unguided} unguided)` : '');
const label = (m) => (M[m].ours ? `${M[m].name} (ours)` : M[m].name);
const seg = (box, items, cur, onpick) => {
  box.innerHTML = items.map(([v, t]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(cur)}">${t}</button>`).join('');
  box.querySelectorAll('button').forEach((b) => b.onclick = () => {
    box.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); onpick(b.dataset.v);
  });
};

// ---- samples: the paper's Figure 8, hand-picked, 5 per class (images/curated/<folder>/<class>_<nn>.jpg) ----
// Folders under images/: the paper's Figure 8 (hand-picked) and, for iMF without guidance, our evaluation run's samples.
const FIG8 = [['r3gan2-l', 'curated/ours'], ['imf-xl', 'curated/imf'], ['imf-xl-noguid', 'samples/imf-xl-noguid'], ['gat', 'curated/gat'], ['stylegan-xl', 'curated/stylegan-xl']];  // check.py reads this line
const CLASSES = [[207, 'golden retriever'], [88, 'macaw'], [1, 'goldfish'], [980, 'volcano'], [985, 'daisy']];  // and this one
// Under each name: its type, parameters, and its guidance and pretrained features.
const FIG8_INFO = {
  'r3gan2-l':    { uses: 'No guidance or pretrained features' },
  'stylegan-xl': { uses: 'Classifier guidance; pretrained discriminator features' },
  'imf-xl':      { uses: 'Guidance trained in (default ω)' },
  'imf-xl-noguid': { uses: 'Guidance off (ω = 1)' },
  'gat':         { uses: 'DINOv2-aligned discriminator' },
};
let cls = 207;
function drawSamples() {
  $('fig8').innerHTML = FIG8.map(([m, dir]) => { const I = FIG8_INFO[m]; return `<div class="who${M[m].ours ? ' ours' : ''}">${label(m)}${cite(M[m].ref)}
    <small>${size(m)}</small><small>${fid(m)}</small><small>${I.uses}</small></div>`; }).map((who, i) => who +
    Array.from({ length: 5 }, (_, k) => `<img src="images/${FIG8[i][1]}/${cls}_${String(k).padStart(2, '0')}.jpg" alt="${M[FIG8[i][0]].name}, ${CLASSES.find(([c]) => c === cls)[1]}" loading="lazy">`).join('')).join('');
}
seg($('fig8cls'), CLASSES, cls, (v) => { cls = +v; drawSamples(); });
drawSamples();

// ---- speed: like an image app, each panel shows its last finished image and waits for the next ----
// Methods within about 10% of our FID at each size; times from compute per image at one GPU speed.
const TIERS = {
  s: { label: 'FID ≈ 2.1', ours: 'r3gan2-s', others: ['gat', 'alphaflow-2step', 'stylegan-xl', 'var-d30', 'dit'] },
  m: { label: 'FID ≈ 1.6', ours: 'r3gan2-m', others: ['facm', 'imf-xl-2step', 'rar-xxl', 'rae-noguid', 'mar-h'] },
  l: { label: 'FID ≈ 1.5', ours: 'r3gan2-l', others: ['imf-xl-2step', 'rar-xxl', 'rae-noguid', 'repa', 'mar-h'] },
};
const GPU = 20;  // TFLOP/s
// Every latent model costed with a small distilled decoder (TAEF2, 0.036 TFLOPs) in place of its own, so the
// generators are what differ. DEC: each model's own decoder in tf (paper, Table F.2 comments); default SD-VAE-sized.
// Pixel-space models (0) keep their cost.
const TAE = 0.0356, DEC = { 'stylegan-xl': 0, 'var-d30': 0.1, 'rar-xxl': 0.187, 'mar-h': 0.25, 'rae-noguid': 0.2 };
const tfSmall = (m) => { const d = DEC[m] ?? 0.622; return d ? M[m].tf - d + TAE : M[m].tf; };
const secs = (m) => tfSmall(m) / GPU;
const fmtT = (s) => (s < 1 ? `${(s * 1000).toPrecision(2) * 1} ms` : `${s < 10 ? s.toFixed(1) : s.toFixed(0)} s`);
const tile = (m, k) => (M[m].n ? `<img src="images/samples/${m}/207_${String(k % M[m].n).padStart(2, '0')}.jpg" alt="">` : `<div class="ph">${M[m].name}<small>samples to come</small></div>`);
let tier = 's', D = [];
const preload = {};
const tierList = (t) => [TIERS[t].ours, ...TIERS[t].others].sort((a, b) => tfSmall(a) - tfSmall(b));
const panelsHTML = (L) => L.map((m, i) => `<div class="wp${M[m].ours ? ' ours' : ''}"><div class="who">${label(m)}${cite(M[m].ref)}<small>${M[m].type}</small><small>${M[m].params} params</small><small>${passes(m)}</small><small>${fid(m)}</small></div>
    <div class="stage" id="st${i}"></div><div class="meta"><span><b id="dc${i}">0</b> <span id="dw${i}">images</span></span><span>${fmtT(secs(m))} each</span></div></div>`).join('');
// The labels get the height of the tallest at any FID level, so switching levels moves nothing below the panels.
function lockLabels() {
  const box = $('panels'); box.style.removeProperty('--who-h'); let h = 0;
  for (const t in TIERS) { box.innerHTML = panelsHTML(tierList(t)); for (const w of box.querySelectorAll('.who')) h = Math.max(h, w.offsetHeight); }
  box.style.setProperty('--who-h', `${h}px`);
}
function restartPanels() {
  const L = tierList(tier);
  for (const m of L) for (let k = 0; k < M[m].n; k++) (preload[`${m}${k}`] ||= new Image()).src = `images/samples/${m}/207_${String(k).padStart(2, '0')}.jpg`;  // no blank tiles on a first visit
  $('panels').innerHTML = panelsHTML(L);
  D = L.map((m, i) => ({ m, p: 0, n: 0, T: secs(m), st: $('st' + i), cnt: $('dc' + i), word: $('dw' + i), ring: null }));
}
function stepPanels(dt) {
  for (const P of D) {
    P.p += dt / P.T; let made = false;
    while (P.p >= 1) { P.p -= 1; P.n++; made = true; }
    const fast = P.T < 0.1;  // a new image every few frames: nothing to wait for
    if (made) { P.cnt.textContent = P.n.toLocaleString('en-US'); P.word.textContent = P.n === 1 ? 'image' : 'images'; }
    if (made || !P.drawn) {  // the last finished image underneath, the progress ring on top
      P.drawn = true;
      const last = P.n ? tile(P.m, P.n - 1) : '<div></div>';
      P.st.innerHTML = fast ? last : `${last}<div class="dim"></div><div class="ring"><svg viewBox="0 0 48 48">
        <circle class="rt" cx="24" cy="24" r="20"/><circle class="rf" cx="24" cy="24" r="20" pathLength="100" stroke-dasharray="0 100"/></svg><span></span></div>`;
      P.arc = P.st.querySelector('.rf'); P.pct = P.st.querySelector('.ring span');
    }
    if (!fast) { P.arc.setAttribute('stroke-dasharray', `${(100 * P.p).toFixed(2)} 100`); P.pct.textContent = `${Math.floor(100 * P.p)}%`; }
  }
}
seg($('tiers'), Object.entries(TIERS).map(([k, t]) => [k, t.label]), tier, (v) => { tier = v; restartPanels(); });
$('restart').onclick = restartPanels;
lockLabels(); restartPanels();
let lastW = innerWidth, rz = 0;
addEventListener('resize', () => {  // label heights depend on the column width; phones also resize while scrolling
  if (innerWidth === lastW) return;
  lastW = innerWidth; clearTimeout(rz); rz = setTimeout(() => { lockLabels(); restartPanels(); }, 200);
});
document.fonts && document.fonts.ready.then(() => { lockLabels(); restartPanels(); });  // the lab fonts change the heights
// The panels run only while the speed section is on screen (scrolled away, they pause where they are, as in a hidden
// tab), so they take nothing from the generator and the video further down.
let last = performance.now(), speedSeen = true, looping = false;
function loop(t) {
  const dt = Math.min(0.1, (t - last) / 1000); last = t;  // a hidden tab pauses rather than jumping ahead
  stepPanels(dt);
  if (speedSeen) requestAnimationFrame(loop); else looping = false;
}
function startLoop() { if (looping) return; looping = true; last = performance.now(); requestAnimationFrame(loop); }
if ('IntersectionObserver' in window) {
  new IntersectionObserver((es) => { speedSeen = es.some((e) => e.isIntersecting); if (speedSeen) startLoop(); }).observe($('speed-section'));
}
startLoop();

// ---- the live generator: its frame loads as it nears the screen; it asks before downloading a model ----
const demo = $('demo-frame');
if (!navigator.gpu) {
  demo.outerHTML = '<p class="nogpu">The live sample generator needs WebGPU, which this browser does not offer. Recent desktop Chrome, Edge and Firefox have it.</p>';
} else {
  const shared = new URLSearchParams(location.search);  // a link copied from the generator carries its state
  // Loaded when it nears the screen (a link to the generator's state: at once), so the page itself starts no GPU work.
  const go = () => { if (!demo.src) demo.src = shared.has('m') ? `generator.html?${shared}` : 'generator.html'; };
  if (shared.has('m') || !('IntersectionObserver' in window)) go();
  else { const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); go(); } }, { rootMargin: '600px 0px' }); io.observe(demo); }
  demo.addEventListener('load', () => {  // same origin: the frame takes its page's height, so the page scrolls, not the frame
    const body = demo.contentDocument && demo.contentDocument.body;
    // rounded up: fractional layout heights (e.g. at 125% display scale) would otherwise leave a sliver to scroll
    const root = demo.contentDocument.documentElement;
    if (body) new ResizeObserver(() => { demo.style.height = `${Math.ceil(root.getBoundingClientRect().height)}px`; }).observe(body);
  });
}
