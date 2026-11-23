// R3GAN2 ImageNet-S + TAEF2 in hand-written WGSL. Activations are channels-last ([pixels, channels]), FP32.
// Weights come pre-folded from dump_wgsl.py (gains, LReLU gains, alphas, weight normalization), so every kernel is
// plain conv/GEMM arithmetic with a fused epilogue. One command buffer per frame.

const lrelu = 'select(0.2 * v, v, v >= 0.0)';

// Y[P,N] = X[P,K] W[K,N] (+bias) (+noise[p]*s[n]) (lrelu) (+Y, i.e. residual in place) (relu)
function gemmCode(f) {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;  // P, N, K, noise offset
${f.bias ? '@group(0) @binding(4) var<storage, read> Bv: array<f32>;' : ''}
${f.noise ? '@group(0) @binding(5) var<storage, read> Sv: array<f32>;\n@group(0) @binding(6) var<storage, read> NZ: array<fp32>;' : ''}
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let P = d.x; let N = d.y; let K = d.z;
  let m0 = wg.y * 64u; let n0 = wg.x * 64u;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    for (var e = li; e < 1024u; e += 256u) {
      let m = e / 16u; let k = e % 16u;
      var a = f32(0.0); if (m0 + m < P && k0 + k < K) { a = X[(m0 + m) * K + k0 + k]; }
      As[k * 64u + m] = a;
      let kb = e / 64u; let n = e % 64u;
      var b = f32(0.0); if (k0 + kb < K && n0 + n < N) { b = Wt[(k0 + kb) * N + n0 + n]; }
      Bs[e] = b;
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      var a: array<f32, 4>; var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = As[k * 64u + l.y * 4u + i]; b[i] = Bs[k * 64u + l.x * 4u + i]; }
      for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]); } }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) { for (var j = 0u; j < 4u; j++) {
    let m = m0 + l.y * 4u + i; let n = n0 + l.x * 4u + j;
    if (m < P && n < N) {
      var v = acc[i * 4u + j];
      ${f.bias ? 'v += Bv[n];' : ''}
      ${f.noise ? 'v += f32(NZ[d.w + m]) * Sv[n];' : ''}
      ${f.lrelu ? `v = ${lrelu};` : ''}
      ${f.resid ? 'v += Y[m * N + n];' : ''}
      ${f.relu ? 'v = max(v, 0.0);' : ''}
      ${f.clamp ? 'v = clamp(v, 0.0, 1.0);' : ''}
      Y[m * N + n] = v;
    }
  } }
}`;
}

// gemmCode with vec4 loads/stores; needs P and N multiples of 64, K of 16. Thread = 4 rows x one vec4 of columns.
// A tile kept row-major as vec4 over k; per 4 k: 4 + 4 vec4 loads feed 16 vec4 FMAs.
// FP16 (th): products in f16, summed in fp32 per 16 k (sums of all K in f16 made the FP16 generator's latent 4x less
// accurate); the epilogue runs in fp32.
function gemmVCode(f) {
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
@group(0) @binding(3) var<uniform> d: vec4u;  // P, N, K, noise offset
${f.bias ? '@group(0) @binding(4) var<storage, read> Bv: array<vec4f>;' : ''}
${f.noise ? '@group(0) @binding(5) var<storage, read> Sv: array<vec4f>;\n@group(0) @binding(6) var<storage, read> NZ: array<fp32>;' : ''}
var<workgroup> As: array<wv4, 256>;  // 64 rows x 4 vec4 (16 k)
var<workgroup> Bs: array<wv4, 256>;  // 16 k x 16 vec4 (64 n)
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let N4 = d.y / 4u; let K4 = d.z / 4u;
  let m0 = wg.y * 64u; let n04 = wg.x * 16u;
  var acc: array<vec4<fp32>, 4>;
  for (var k0 = 0u; k0 < d.z; k0 += 16u) {
    As[li] = wput(X[(m0 + li / 4u) * K4 + k0 / 4u + li % 4u]);
    Bs[li] = wput(Wt[(k0 + li / 16u) * N4 + n04 + li % 16u]);
    workgroupBarrier();
    var part: array<vec4f, 4>;
    for (var kq = 0u; kq < 4u; kq++) {
      var a: array<vec4f, 4>; var b: array<vec4f, 4>;
      for (var i = 0u; i < 4u; i++) { a[i] = wget(As[(l.y * 4u + i) * 4u + kq]); b[i] = wget(Bs[(kq * 4u + i) * 16u + l.x]); }
      for (var i = 0u; i < 4u; i++) { for (var kk = 0u; kk < 4u; kk++) { part[i] = fma(vec4f(a[i][kk]), b[kk], part[i]); } }
    }
    for (var i = 0u; i < 4u; i++) { acc[i] += vec4<fp32>(part[i]); }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + l.y * 4u + i; let n4 = n04 + l.x; let o = m * N4 + n4;
    var v = acc[i];
    ${f.bias ? 'v += vec4<fp32>(Bv[n4]);' : ''}
    ${f.noise ? 'v += NZ[d.w + m] * vec4<fp32>(Sv[n4]);' : ''}
    ${f.lrelu ? 'v = select(0.2 * v, v, v >= vec4<fp32>(0.0));' : ''}
    ${f.resid ? 'v += vec4<fp32>(Y[o]);' : ''}
    ${f.relu ? 'v = max(v, vec4<fp32>(0.0));' : ''}
    Y[o] = vec4f(v);
  }
}`;
}

// The generator's 1x1 convs (gemmVCode's job), shapes baked in. A workgroup of 64 threads makes a 32 x 64 tile; a thread
// 4 rows x 2 vec4 (columns strided by 32). k in steps of 8, the next step's X and W loaded into registers before this
// step's FMAs. Split-K S > 1 (small P, few tiles): workgroup z sums K / S of k into FP32 partials, gemmTReduce adds them
// and applies the epilogue. FP32 only: in FP16 the whole generator was ~5% slower with it than with gemmVCode, though
// it won at 8x8. Tile sizes from a sweep (32-128 tiles, 2-8 x 1-4 per thread, k 8-32, split 1-8) on an AMD
// 880M: the best varied by shape, this was near the top on all. Needs P % 32, N % 64 and K % (8 S) == 0.
function gemmTCode(P, N, K, S, f) {
  const N4 = N / 4, K4 = K / 4, ks = K / S;
  const epi = (i, j) => S > 1 ? `Part[(wg.z * ${P}u + mr${i}) * ${N4}u + nc${j}] = acc${i}${j};` : `{ var v = acc${i}${j}; let o = mr${i} * ${N4}u + nc${j};
    ${gemmTEpi(f, `nc${j}`, `mr${i}`)} }`;
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4f>;
${S > 1 ? '@group(0) @binding(7) var<storage, read_write> Part: array<vec4<fp32>>;' : gemmTBinds(f)}
var<workgroup> As: array<wv4, 64>;   // 32 rows x 2 vec4 (8 k)
var<workgroup> Bs: array<wv4, 128>;  // 8 k x 16 vec4 (64 n)
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let tx = li % 8u; let ty = li / 8u;
  let m0 = wg.y * 32u; let n04 = wg.x * 16u; let kb = wg.z * ${ks}u;
  ${U(4, (i) => U(2, (j) => `var acc${i}${j} = vec4<fp32>(0.0);`))}
  var ra = X[(m0 + li / 2u) * ${K4}u + kb / 4u + li % 2u];
  var rb0 = Wt[(kb + li / 16u) * ${N4}u + n04 + li % 16u];
  var rb1 = Wt[(kb + 4u + li / 16u) * ${N4}u + n04 + li % 16u];
  for (var k0 = kb; k0 < kb + ${ks}u; k0 += 8u) {
    As[li] = wput(ra); Bs[li] = wput(rb0); Bs[li + 64u] = wput(rb1);
    workgroupBarrier();
    if (k0 + 8u < kb + ${ks}u) {
      let k1 = k0 + 8u;
      ra = X[(m0 + li / 2u) * ${K4}u + k1 / 4u + li % 2u];
      rb0 = Wt[(k1 + li / 16u) * ${N4}u + n04 + li % 16u];
      rb1 = Wt[(k1 + 4u + li / 16u) * ${N4}u + n04 + li % 16u];
    }
    for (var kq = 0u; kq < 2u; kq++) {
      ${U(4, (i) => `let a${i} = wget(As[(ty * 4u + ${i}u) * 2u + kq]);`)}
      ${U(4, (k) => U(2, (j) => `let b${k}${j} = wget(Bs[(kq * 4u + ${k}u) * 16u + tx + ${8 * j}u]);`))}
      ${U(4, (i) => U(2, (j) => U(4, (k) => `acc${i}${j} = fma(vec4f(a${i}[${k}]), b${k}${j}, acc${i}${j});`)))}
    }
    workgroupBarrier();
  }
  ${U(4, (i) => `let mr${i} = m0 + ty * 4u + ${i}u;`)}
  ${U(2, (j) => `let nc${j} = n04 + tx + ${8 * j}u;`)}
  ${U(4, (i) => U(2, (j) => epi(i, j)))}
}`;
}
const gemmTBinds = (f) => `@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
${f.bias ? '@group(0) @binding(4) var<storage, read> Bv: array<vec4f>;' : ''}
${f.noise ? '@group(0) @binding(3) var<uniform> d: vec4u;  // noise offset in d.w\n@group(0) @binding(5) var<storage, read> Sv: array<vec4f>;\n@group(0) @binding(6) var<storage, read> NZ: array<fp32>;' : ''}`;
const gemmTEpi = (f, n, m) => `${f.bias ? `v += vec4<fp32>(Bv[${n}]);` : ''}
    ${f.noise ? `v += NZ[d.w + ${m}] * vec4<fp32>(Sv[${n}]);` : ''}
    ${f.lrelu ? 'v = select(0.2 * v, v, v >= vec4<fp32>(0.0));' : ''}
    ${f.resid ? 'v += vec4<fp32>(Y[o]);' : ''}
    Y[o] = vec4f(v);`;
// Adds gemmTCode's S partial sums and applies the epilogue; a thread per output vec4 (P * N / 4 is a multiple of 64).
const gemmTReduce = (P, N, S, f) => `
${gemmTBinds(f)}
@group(0) @binding(7) var<storage, read_write> Part: array<vec4<fp32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let o = gid.x; let m = o / ${N / 4}u; let n = o % ${N / 4}u;
  var v = Part[o];
  for (var s = 1u; s < ${S}u; s++) { v += Part[s * ${P * N / 4}u + o]; }
  ${gemmTEpi(f, 'n', 'm')}
}`;

// The generator's last layer, Y[P, 32] = X[P, K] W[K, 32]. gemmCode's 64 x 64 tiles leave half of each tile idle at
// N = 32 and give only P / 64 = 16 workgroups for the whole GPU (~3 ms of S's ~22 ms generator on an AMD 880M).
// Here: a workgroup per 8 pixels (128), thread = (output, pixel), the 8 rows of X staged 128 k at a time.
// Needs P % 8 == 0 and K % 128 == 0 (S, M, L: K = 512, 1024, 1536).
const head32Code = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;  // P, 32, K
var<workgroup> Xs: array<f32, 1024>;  // 8 rows x 128 k
@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let K = d.z; let m0 = wg.x * 8u;
  var acc = fp32(0.0);  // FP16: sums of 16 terms in f16, added up in fp32
  for (var k0 = 0u; k0 < K; k0 += 128u) {
    for (var e = li; e < 1024u; e += 256u) { Xs[e] = X[(m0 + e / 128u) * K + k0 + e % 128u]; }
    workgroupBarrier();
    for (var k1 = 0u; k1 < 128u; k1 += 16u) {
      var part = f32(0.0);
      for (var k = k1; k < k1 + 16u; k++) { part = fma(Xs[l.y * 128u + k], Wt[(k0 + k) * 32u + l.x], part); }
      acc += fp32(part);
    }
    workgroupBarrier();
  }
  Y[(m0 + l.y) * 32u + l.x] = f32(acc);
}`;

// Grouped 3x3 conv, 32 channels per group, LReLU. Workgroup: one group x an 8x8 pixel tile; thread = (co, row).
const gconvCode = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;   // [g, tap, ci, co]
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;               // H, W, channels
var<workgroup> T: array<f32, 3200>;                        // 10 x 10 x 32
@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let H = d.x; let Wd = d.y; let Ct = d.z; let g = wg.y;
  let tilesX = Wd / 8u; let ty0 = (wg.x / tilesX) * 8u; let tx0 = (wg.x % tilesX) * 8u;
  for (var e = li; e < 3200u; e += 256u) {
    let c = e % 32u; let p = e / 32u;
    let yy = i32(ty0) + i32(p / 10u) - 1; let xx = i32(tx0) + i32(p % 10u) - 1;
    var v = f32(0.0);
    if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(Wd)) { v = X[(u32(yy) * Wd + u32(xx)) * Ct + g * 32u + c]; }
    T[e] = v;
  }
  workgroupBarrier();
  let co = l.x; let ry = l.y;
  var acc: array<f32, 8>;
  for (var tap = 0u; tap < 9u; tap++) {
    let ky = tap / 3u; let kx = tap % 3u;
    for (var ci = 0u; ci < 32u; ci++) {
      let w = Wt[((g * 9u + tap) * 32u + ci) * 32u + co];
      for (var x = 0u; x < 8u; x++) { acc[x] = fma(T[((ry + ky) * 10u + x + kx) * 32u + ci], w, acc[x]); }
    }
  }
  for (var x = 0u; x < 8u; x++) { let v = acc[x]; Y[((ty0 + ry) * Wd + tx0 + x) * Ct + g * 32u + co] = ${lrelu}; }
}`;

// Grouped 3x3 conv, vec4 version: workgroup = one 32-channel group x a TW x TH pixel tile; thread = PX pixels of a row
// x 4 output channels (8 threads across the 32). Input tile in shared memory as vec4; weights read as vec4 (each feeds
// 4 PX FMAs). The tile's shared memory (12.8 KB FP32 at 8 x 8) limits how many workgroups fit on a CU, so fewer pixels
// per thread (more threads per tile) mostly wins: on an AMD 880M, 8 x 4 tiles with 2 pixels per thread were 1.1-2x
// faster than 8 x 8 with 8 per thread, except FP16 at 32 x 32, where 8 x 8 / 8 stays best (gconvCfg).
// Sums are unrolled into named variables: the looped array version gave wrong results at x = 0 and 4.
// FP16: sums per tap (32 terms) in f16, added up in fp32 (FP16 sums over all 288 terms lost too much accuracy).
const gconvCfg = (half, P) => (half && P >= 1024 ? { TW: 8, TH: 8, PX: 8 } : { TW: 8, TH: 4, PX: 2 });
function gconvVCode({ TW, TH, PX }, half) {
  const HW = TW + 2, NP = (TH + 2) * HW, XS = TW / PX, NT = TH * XS * 8;
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4f>;  // [g, tap, ci, co/4]
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
@group(0) @binding(3) var<uniform> d: vec4u;                // H, W, channels
var<workgroup> T: array<wv4, ${NP * 8}>;                    // ${TH + 2} x ${HW} pixels x 8 vec4 (32 channels)
@compute @workgroup_size(${NT})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let H = d.x; let Wd = d.y; let C4 = d.z / 4u; let g = wg.y;
  let tilesX = Wd / ${TW}u; let ty0 = (wg.x / tilesX) * ${TH}u; let tx0 = (wg.x % tilesX) * ${TW}u;
  for (var e = li; e < ${NP * 8}u; e += ${NT}u) {
    let q = e % 8u; let p = e / 8u;
    let yy = i32(ty0) + i32(p / ${HW}u) - 1; let xx = i32(tx0) + i32(p % ${HW}u) - 1;
    var v = vec4f(0.0);
    if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(Wd)) { v = X[(u32(yy) * Wd + u32(xx)) * C4 + g * 8u + q]; }
    T[e] = wput(v);
  }
  workgroupBarrier();
  let cq = li % 8u; let row = li / ${8 * XS}u; let x0 = ((li / 8u) % ${XS}u) * ${PX}u;
  ${U(PX, (x) => `var acc${x} = vec4<fp32>(0.0);`)}
  for (var tap = 0u; tap < 9u; tap++) {
    let ky = tap / 3u; let kx = tap % 3u;
    let rb = (row + ky) * ${HW}u + x0 + kx;
    ${half ? U(PX, (x) => `var p${x} = vec4f(0.0);`) : ''}
    for (var q = 0u; q < 8u; q++) {
      let wb = ((g * 9u + tap) * 32u + q * 4u) * 8u + cq;
      let w0 = Wt[wb]; let w1 = Wt[wb + 8u]; let w2 = Wt[wb + 16u]; let w3 = Wt[wb + 24u];
      ${U(PX, (x) => { const s = half ? `p${x}` : `acc${x}`; return `{ let a = wget(T[(rb + ${x}u) * 8u + q]); ${s} = fma(vec4f(a.x), w0, ${s}); ${s} = fma(vec4f(a.y), w1, ${s}); ${s} = fma(vec4f(a.z), w2, ${s}); ${s} = fma(vec4f(a.w), w3, ${s}); }`; })}
    }
    ${half ? U(PX, (x) => `acc${x} += vec4<fp32>(p${x});`) : ''}
  }
  let ob = ((ty0 + row) * Wd + tx0 + x0) * C4 + g * 8u + cq;
  ${U(PX, (x) => `{ let v = vec4f(acc${x}); Y[ob + ${x}u * C4] = select(0.2 * v, v, v >= vec4f(0.0)); }`)}
}`;
}

// Unrolled code: U(n, f) writes f(0) ... f(n-1) on separate lines. The TAEF2 convs keep their per-thread sums in named
// variables, never in arrays indexed by a loop counter: Intel's shader compiler put such arrays in memory, which made
// TAEF2 13x slower on Intel Xe2 than on AMD.
const U = (n, f) => Array.from({ length: n }, (_, i) => f(i)).join('\n');
const XYZW = 'xyzw';

// Final TAEF2 layer, 64 -> RGB 3x3 conv with bias and clamp to [0,1].
// Workgroup: 8x8 output pixels, one thread per pixel with three FP32 sums; the input tile and the weights of one
// 16-channel chunk sit in shared memory. The sums and the image stay FP32 also when inputs and weights are FP16
// (vec3f is untouched by th()). Against the 64-lane kernel it replaces: S +5% images/s, image RMSE vs PyTorch
// -6% on S and -22% on M with the FP16 decoder.
function convRGBCode() {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;   // [tap, ci, 3]
@group(0) @binding(2) var<storage, read_write> Y: array<fp32>;  // the image stays FP32
@group(0) @binding(3) var<uniform> d: vec4u;               // output H, W, Cin
@group(0) @binding(4) var<storage, read> Bv: array<f32>;
var<workgroup> T: array<f32, 1600>;                        // 10 x 10 x 16
var<workgroup> Ws: array<f32, 432>;                        // 9 taps x 16 ci x 3
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let H = d.x; let Wd = d.y; let Cin = d.z;
  let tilesX = Wd / 8u; let ty0 = (wg.x / tilesX) * 8u; let tx0 = (wg.x % tilesX) * 8u;
  var acc = vec3f(0.0);
  for (var c0 = 0u; c0 < Cin; c0 += 16u) {
    for (var e = li; e < 1600u; e += 64u) {
      let p = e / 16u;
      let yy = i32(ty0) + i32(p / 10u) - 1; let xx = i32(tx0) + i32(p % 10u) - 1;
      var v = f32(0.0);
      if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(Wd)) { v = X[(u32(yy) * Wd + u32(xx)) * Cin + c0 + e % 16u]; }
      T[e] = v;
    }
    for (var e = li; e < 432u; e += 64u) { Ws[e] = Wt[((e / 48u) * Cin + c0 + (e % 48u) / 3u) * 3u + e % 3u]; }
    workgroupBarrier();
    for (var tap = 0u; tap < 9u; tap++) {
      for (var ci = 0u; ci < 16u; ci++) {
        let a = T[((l.y + tap / 3u) * 10u + l.x + tap % 3u) * 16u + ci];
        let wi = (tap * 16u + ci) * 3u;
        acc = fma(vec3f(vec3<f32>(a)), vec3f(vec3<f32>(Ws[wi], Ws[wi + 1u], Ws[wi + 2u])), acc);
      }
    }
    workgroupBarrier();
  }
  acc = clamp(acc + vec3f(vec3<f32>(Bv[0], Bv[1], Bv[2])), vec3f(0.0), vec3f(1.0));
  let o = ((ty0 + l.y) * Wd + tx0 + l.x) * 3u;
  Y[o] = acc.x; Y[o + 1u] = acc.y; Y[o + 2u] = acc.z;
}`;
}

// Dense 3x3 conv (TAEF2), Cin multiple of 16, Cout = 64. Optional nearest 2x upsample of the input folded into the
// reads, bias, in-place residual, ReLU, clamp. Register-blocked: workgroup = 8x8 pixels x 64 co, thread = 8 pixels
// (one row) x 8 co.
// Input tile per 16-channel chunk and the weights of one tap at a time sit in shared memory.
function conv3gCode(f) {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;   // [tap, ci, 64]
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;               // output H, W, Cin
${f.bias ? '@group(0) @binding(4) var<storage, read> Bv: array<f32>;' : ''}
var<workgroup> T: array<f32, 1600>;                        // 10 x 10 x 16
var<workgroup> Ws: array<f32, 1024>;                       // 16 ci x 64 co
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let H = d.x; let Wd = d.y; let Cin = d.z;
  let tilesX = Wd / 8u; let ty0 = (wg.x / tilesX) * 8u; let tx0 = (wg.x % tilesX) * 8u;
  let Win = ${f.ups ? 'Wd / 2u' : 'Wd'};
  let ty = l.y; let cb = l.x * 8u;
  var acc: array<f32, 64>;
  for (var c0 = 0u; c0 < Cin; c0 += 16u) {
    for (var e = li; e < 1600u; e += 64u) {
      let c = e % 16u; let p = e / 16u;
      let yy = i32(ty0) + i32(p / 10u) - 1; let xx = i32(tx0) + i32(p % 10u) - 1;
      var v = f32(0.0);
      if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(Wd)) {
        let sy = ${f.ups ? 'u32(yy) / 2u' : 'u32(yy)'}; let sx = ${f.ups ? 'u32(xx) / 2u' : 'u32(xx)'};
        v = X[(sy * Win + sx) * Cin + c0 + c];
      }
      T[e] = v;
    }
    for (var tap = 0u; tap < 9u; tap++) {
      workgroupBarrier();
      for (var e = li; e < 1024u; e += 64u) { Ws[e] = Wt[(tap * Cin + c0 + e / 64u) * 64u + e % 64u]; }
      workgroupBarrier();
      let ky = tap / 3u; let kx = tap % 3u;
      for (var ci = 0u; ci < 16u; ci++) {
        var a: array<f32, 8>; var b: array<f32, 8>;
        for (var x = 0u; x < 8u; x++) { a[x] = T[((ty + ky) * 10u + x + kx) * 16u + ci]; }
        for (var j = 0u; j < 8u; j++) { b[j] = Ws[ci * 64u + cb + j]; }
        for (var x = 0u; x < 8u; x++) { for (var j = 0u; j < 8u; j++) { acc[x * 8u + j] = fma(a[x], b[j], acc[x * 8u + j]); } }
      }
    }
    workgroupBarrier();
  }
  for (var x = 0u; x < 8u; x++) { for (var j = 0u; j < 8u; j++) {
    let co = cb + j; let idx = ((ty0 + ty) * Wd + tx0 + x) * 64u + co;
    var v = acc[x * 8u + j];
    ${f.bias ? 'v += Bv[co];' : ''}
    ${f.resid ? 'v += Y[idx];' : ''}
    ${f.relu ? 'v = max(v, 0.0);' : ''}
    ${f.clamp ? 'v = clamp(v, 0.0, 1.0);' : ''}
    Y[idx] = v;
  } }
}`;
}

// conv3gCode with vec4 everywhere: tile and weights in shared memory as vec4, 16 vec4 loads per 64 vec4 FMAs.
function conv3vCode(f) {
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4f>;  // [tap, ci, 64/4]
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
@group(0) @binding(3) var<uniform> d: vec4u;                // output H, W, Cin
${f.bias ? '@group(0) @binding(4) var<storage, read> Bv: array<vec4f>;' : ''}
var<workgroup> T: array<wv4, 400>;                        // 10 x 10 pixels x 4 vec4 (16 channels)
var<workgroup> Ws: array<wv4, 256>;                       // 16 ci x 16 vec4 (64 co)
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let H = d.x; let Wd = d.y; let Cin = d.z; let C4 = Cin / 4u;
  let tilesX = Wd / 8u; let ty0 = (wg.x / tilesX) * 8u; let tx0 = (wg.x % tilesX) * 8u;
  let Win = ${f.ups ? 'Wd / 2u' : 'Wd'};
  let ty = l.y; let cb4 = l.x * 2u;
${U(8, (x) => `  var A${x} = vec4f(0.0); var B${x} = vec4f(0.0);`)}
  for (var c0 = 0u; c0 < Cin; c0 += 16u) {
    for (var e = li; e < 400u; e += 64u) {
      let q = e % 4u; let p = e / 4u;
      let yy = i32(ty0) + i32(p / 10u) - 1; let xx = i32(tx0) + i32(p % 10u) - 1;
      var v = vec4f(0.0);
      if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(Wd)) {
        let sy = ${f.ups ? 'u32(yy) / 2u' : 'u32(yy)'}; let sx = ${f.ups ? 'u32(xx) / 2u' : 'u32(xx)'};
        v = X[(sy * Win + sx) * C4 + c0 / 4u + q];
      }
      T[e] = wput(v);
    }
    for (var tap = 0u; tap < 9u; tap++) {
      workgroupBarrier();
      for (var e = li; e < 256u; e += 64u) { Ws[e] = wput(Wt[(tap * Cin + c0 + e / 16u) * 16u + e % 16u]); }
      workgroupBarrier();
      let ky = tap / 3u; let kx = tap % 3u;
      for (var q = 0u; q < 4u; q++) {
        let t0 = ((ty + ky) * 10u + kx) * 4u + q;
${U(8, (x) => `        let a${x} = wget(T[t0 + ${4 * x}u]);`)}
${U(4, (cc) => `        {
          let b0 = wget(Ws[(q * 4u + ${cc}u) * 16u + cb4]); let b1 = wget(Ws[(q * 4u + ${cc}u) * 16u + cb4 + 1u]);
${U(8, (x) => `          A${x} = fma(vec4f(a${x}.${XYZW[cc]}), b0, A${x}); B${x} = fma(vec4f(a${x}.${XYZW[cc]}), b1, B${x});`)}
        }`)}
      }
    }
    workgroupBarrier();
  }
${U(16, (i) => { const x = i >> 1, hh = i & 1; return `  {
    let o = ((ty0 + ty) * Wd + tx0 + ${x}u) * 16u + cb4 + ${hh}u;
    var v = ${hh ? 'B' : 'A'}${x};
    ${f.bias ? `v += Bv[cb4 + ${hh}u];` : ''}
    ${f.resid ? 'v += Y[o];' : ''}
    ${f.relu ? 'v = max(v, vec4f(0.0));' : ''}
    ${f.clamp ? 'v = clamp(v, vec4f(0.0), vec4f(1.0));' : ''}
    Y[o] = v;
  }`; })}
}`;
}

// FP16-only TAEF2 conv (Cout = 64), larger tile: 16x8 pixels x 64 co per workgroup of 128 threads, thread = 8 pixels
// x 8 co. Per 8-channel chunk, the input tile and all 9 taps of weights are staged at once (12 KB as vec4h).
function conv3wCode(f) {
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;
@group(0) @binding(1) var<storage, read> Wt: array<vec4f>;  // [tap, ci, 64/4]
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
@group(0) @binding(3) var<uniform> d: vec4u;                // output H, W, Cin
${f.bias ? '@group(0) @binding(4) var<storage, read> Bv: array<vec4f>;' : ''}
var<workgroup> T: array<wv4, 360>;                        // 10 rows x 18 cols x 2 vec4 (8 channels)
var<workgroup> Ws: array<wv4, 1152>;                      // 9 taps x 8 ci x 16 vec4 (64 co)
@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let H = d.x; let Wd = d.y; let Cin = d.z; let C4 = Cin / 4u;
  let tilesX = Wd / 16u; let ty0 = (wg.x / tilesX) * 8u; let tx0 = (wg.x % tilesX) * 16u;
  let Win = ${f.ups ? 'Wd / 2u' : 'Wd'};
  let row = l.y / 2u; let xs = (l.y % 2u) * 8u; let cb4 = l.x * 2u;
${U(8, (x) => `  var A${x} = vec4f(0.0); var B${x} = vec4f(0.0);`)}
  for (var c0 = 0u; c0 < Cin; c0 += 8u) {
    for (var e = li; e < 360u; e += 128u) {
      let q = e % 2u; let p = e / 2u;
      let yy = i32(ty0) + i32(p / 18u) - 1; let xx = i32(tx0) + i32(p % 18u) - 1;
      var v = vec4f(0.0);
      if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(Wd)) {
        let sy = ${f.ups ? 'u32(yy) / 2u' : 'u32(yy)'}; let sx = ${f.ups ? 'u32(xx) / 2u' : 'u32(xx)'};
        v = X[(sy * Win + sx) * C4 + c0 / 4u + q];
      }
      T[e] = wput(v);
    }
    for (var e = li; e < 1152u; e += 128u) {
      let tap = e / 128u; let r = e % 128u;
      Ws[e] = wput(Wt[(tap * Cin + c0 + r / 16u) * 16u + r % 16u]);
    }
    workgroupBarrier();
    for (var tap = 0u; tap < 9u; tap++) {
      let ky = tap / 3u; let kx = tap % 3u;
      for (var q = 0u; q < 2u; q++) {
        let t0 = ((row + ky) * 18u + xs + kx) * 2u + q;
${U(8, (x) => `        let a${x} = wget(T[t0 + ${2 * x}u]);`)}
${U(4, (cc) => `        {
          let wi = (tap * 8u + q * 4u + ${cc}u) * 16u + cb4;
          let b0 = wget(Ws[wi]); let b1 = wget(Ws[wi + 1u]);
${U(8, (x) => `          A${x} = fma(vec4f(a${x}.${XYZW[cc]}), b0, A${x}); B${x} = fma(vec4f(a${x}.${XYZW[cc]}), b1, B${x});`)}
        }`)}
      }
    }
    workgroupBarrier();
  }
${U(16, (i) => { const x = i >> 1, hh = i & 1; return `  {
    let o = ((ty0 + row) * Wd + tx0 + xs + ${x}u) * 16u + cb4 + ${hh}u;
    var v = ${hh ? 'B' : 'A'}${x};
    ${f.bias ? `v += Bv[cb4 + ${hh}u];` : ''}
    ${f.resid ? 'v += Y[o];' : ''}
    ${f.relu ? 'v = max(v, vec4f(0.0));' : ''}
    Y[o] = v;
  }`; })}
}`;
}

// TAEF2 upsample conv (nearest 2x, then 3x3, Cout = 64, no epilogue) as four 2x2 convs on the input, one per output
// phase (py, px), with phase-summed weights (convUpWeights): 4 taps instead of 9, exact. One phase per workgroup
// (wg.y), else conv3wCode's structure: 8 x 16 input pixels x 64 co, thread = 8 pixels x 8 co, an 8-channel chunk of
// the input tile and this phase's 4 taps of weights in shared memory. Sizes are constants. On an AMD 880M (FP16):
// 1.96x / 1.53x / 1.25x faster than conv3w(ups) at 64² / 128² / 256² out, RMS error vs FP64 0.0008 (conv3w 0.0013).
// NVIDIA TITAN Xp (Pascal, FP32 decoder): TAEF2 6.46 -> 6.04 ms GPU, generator page 12.2 -> 11.3 ms per frame.
function convUpCode(Hi, Wi) {
  const Wo = 2 * Wi;
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;   // [Hi*Wi][16]
@group(0) @binding(1) var<storage, read> Wp: array<vec4f>;  // [phase 4][tap 4][ci 64][16]
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
var<workgroup> T: array<wv4, 360>;    // 10 rows x 18 cols x 2 vec4 (8 channels)
var<workgroup> Ws: array<wv4, 512>;   // 4 taps x 8 ci x 16 vec4 (64 co)
@compute @workgroup_size(8, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(local_invocation_index) li: u32) {
  let ix0 = (wg.x % ${Wi / 16}u) * 16u; let iy0 = (wg.x / ${Wi / 16}u) * 8u;
  let ph = wg.y; let py = ph / 2u; let px = ph % 2u;
  let row = l.y / 2u; let xs = (l.y % 2u) * 8u; let cb4 = l.x * 2u;
${U(8, (x) => `  var A${x} = vec4f(0.0); var B${x} = vec4f(0.0);`)}
  for (var c0 = 0u; c0 < 64u; c0 += 8u) {
    for (var e = li; e < 360u; e += 128u) {
      let q = e % 2u; let p = e / 2u;
      let yy = i32(iy0) + i32(p / 18u) - 1; let xx = i32(ix0) + i32(p % 18u) - 1;
      var v = vec4f(0.0);
      if (yy >= 0 && yy < ${Hi} && xx >= 0 && xx < ${Wi}) { v = X[(u32(yy) * ${Wi}u + u32(xx)) * 16u + c0 / 4u + q]; }
      T[e] = wput(v);
    }
    for (var e = li; e < 512u; e += 128u) { Ws[e] = wput(Wp[((ph * 4u + e / 128u) * 64u + c0 + (e / 16u) % 8u) * 16u + e % 16u]); }
    workgroupBarrier();
    for (var tap = 0u; tap < 4u; tap++) {
      let a = tap / 2u; let b = tap % 2u;
      for (var q = 0u; q < 2u; q++) {
        let t0 = ((row + py + a) * 18u + xs + px + b) * 2u + q;
${U(8, (x) => `        let a${x} = wget(T[t0 + ${2 * x}u]);`)}
${U(4, (cc) => `        {
          let wi = (tap * 8u + q * 4u + ${cc}u) * 16u + cb4;
          let b0 = wget(Ws[wi]); let b1 = wget(Ws[wi + 1u]);
${U(8, (x) => `          A${x} = fma(vec4f(a${x}.${XYZW[cc]}), b0, A${x}); B${x} = fma(vec4f(a${x}.${XYZW[cc]}), b1, B${x});`)}
        }`)}
      }
    }
    workgroupBarrier();
  }
${U(8, (x) => `  { let o = ((2u * (iy0 + row) + py) * ${Wo}u + 2u * (ix0 + xs + ${x}u) + px) * 16u + cb4; Y[o] = A${x}; Y[o + 1u] = B${x}; }`)}
}`;
}
// [tap][ci][co] 3x3 weights -> [phase][tap][ci][co]: rows {0} | {1, 2} for py = 0, {0, 1} | {2} for py = 1; columns alike.
function convUpWeights(w, Cin = 64, Cout = 64) {
  const sets = [[[0], [1, 2]], [[0, 1], [2]]], out = new Float32Array(16 * Cin * Cout);
  for (let py = 0; py < 2; py++) for (let px = 0; px < 2; px++) for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
    const o = ((py * 2 + px) * 4 + a * 2 + b) * Cin * Cout;
    for (const ky of sets[py][a]) for (const kx of sets[px][b]) {
      const s = (ky * 3 + kx) * Cin * Cout;
      for (let i = 0; i < Cin * Cout; i++) out[o + i] += w[s + i];
    }
  }
  return out;
}

// Small-image TAEF2 3x3 conv (Cin = Cout = 64, stride 1) as Winograd F(2x2, 3x3): 16 multiplies per 2x2 outputs
// instead of 36. Workgroup: 8 x 4 output pixels (4 x 2 tiles) x 64 co, 128 threads: 4x the workgroups of conv3w,
// which leaves most of the GPU idle on a 32x32 image (8 workgroups). Per 16-channel chunk: input patch -> shared,
// input transform (thread = tile x ci), then the 16 GEMMs (thread = component x 8 co over the 8 tiles); at the end
// the sums meet in shared memory (two halves of co, to fit 16 KB in FP16) for the output transform (thread = tile x
// 4 co). FP16 only: in FP32 the shared memory would exceed 16 KB. On an AMD 880M: 2.5x faster than conv3w at 32²,
// RMS error vs FP64 0.0006 (conv3w 0.0009); at 64² only 1.09x and at 128² slower, so it is used at 32² only.
function convWinoCode(H, W, f) {
  return `
@group(0) @binding(0) var<storage, read> X: array<vec4f>;   // [H*W][16]
@group(0) @binding(1) var<storage, read> Uw: array<vec4f>;  // [16 components][64 ci][16]
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
${f.bias ? '@group(0) @binding(3) var<storage, read> Bv: array<vec4f>;' : ''}
var<workgroup> Pt: array<vec4f, 240>;   // 6 rows x 10 cols x 4 vec4 (16 channels)
var<workgroup> Vs: array<f32, 2048>;    // [component][ci 16][tile 8]
var<workgroup> Ms: array<wv4, 1024>;    // [component][tile 8][8 vec4]: half of co at a time
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let ox0 = (wg.x % ${W / 8}u) * 8u; let oy0 = (wg.x / ${W / 8}u) * 4u;
  let xi = li / 8u; let cob = li % 8u;
${U(8, (t) => `  var A${t} = vec4f(0.0); var B${t} = vec4f(0.0);`)}
  for (var c0 = 0u; c0 < 64u; c0 += 16u) {
    for (var e = li; e < 240u; e += 128u) {
      let q = e % 4u; let p = e / 4u;
      let yy = i32(oy0) + i32(p / 10u) - 1; let xx = i32(ox0) + i32(p % 10u) - 1;
      var v = vec4f(0.0);
      if (yy >= 0 && yy < ${H} && xx >= 0 && xx < ${W}) { v = X[(u32(yy) * ${W}u + u32(xx)) * 16u + c0 / 4u + q]; }
      Pt[e] = v;
    }
    workgroupBarrier();
    {  // V = B^T d B, B^T = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1]
      let t = li / 16u; let ci = li % 16u; let tx = t % 4u; let ty = t / 4u; let q = ci / 4u; let cc = ci % 4u;
${U(4, (i) => U(4, (j) => `      let d${i}${j} = Pt[((ty * 2u + ${i}u) * 10u + tx * 2u + ${j}u) * 4u + q][cc];`))}
${U(4, (j) => `      let t0${j} = d0${j} - d2${j}; let t1${j} = d1${j} + d2${j}; let t2${j} = d2${j} - d1${j}; let t3${j} = d1${j} - d3${j};`)}
${U(4, (i) => `      Vs[(${i * 4}u * 16u + ci) * 8u + t] = t${i}0 - t${i}2; Vs[(${i * 4 + 1}u * 16u + ci) * 8u + t] = t${i}1 + t${i}2;
      Vs[(${i * 4 + 2}u * 16u + ci) * 8u + t] = t${i}2 - t${i}1; Vs[(${i * 4 + 3}u * 16u + ci) * 8u + t] = t${i}1 - t${i}3;`)}
    }
    workgroupBarrier();
    for (var ci = 0u; ci < 16u; ci++) {
      let base = (xi * 16u + ci) * 8u;
      let wi = (xi * 64u + c0 + ci) * 16u + cob * 2u;
      let b0 = Uw[wi]; let b1 = Uw[wi + 1u];
${U(8, (t) => `      { let a = vec4f(Vs[base + ${t}u]); A${t} = fma(a, b0, A${t}); B${t} = fma(a, b1, B${t}); }`)}
    }
    workgroupBarrier();
  }
  for (var h = 0u; h < 2u; h++) {
    if (cob / 4u == h) {
      let c = (cob % 4u) * 2u;
${U(8, (t) => `      Ms[(xi * 8u + ${t}u) * 8u + c] = wput(A${t}); Ms[(xi * 8u + ${t}u) * 8u + c + 1u] = wput(B${t});`)}
    }
    workgroupBarrier();
    if (li < 64u) {  // Y = A^T M A, A^T = [1 1 1 0; 0 1 -1 -1]
      let t = li / 8u; let cl = li % 8u; let c4 = h * 8u + cl; let tx = t % 4u; let ty = t / 4u;
${U(4, (i) => U(4, (j) => `      let m${i}${j} = wget(Ms[(${i * 4 + j}u * 8u + t) * 8u + cl]);`))}
${U(4, (j) => `      let s0${j} = m0${j} + m1${j} + m2${j}; let s1${j} = m1${j} - m2${j} - m3${j};`)}
${U(4, (k) => `      {
        let o = ((oy0 + ty * 2u + ${k >> 1}u) * ${W}u + ox0 + tx * 2u + ${k & 1}u) * 16u + c4;
        var v = ${['s00 + s01 + s02', 's01 - s02 - s03', 's10 + s11 + s12', 's11 - s12 - s13'][k]};
        ${f.bias ? 'v += Bv[c4];' : ''}
        ${f.resid ? 'v += Y[o];' : ''}
        ${f.relu ? 'v = max(v, vec4f(0.0));' : ''}
        Y[o] = v;
      }`)}
    }
    workgroupBarrier();
  }
}`;
}
// [tap][ci][co] 3x3 weights -> U = G g G^T per (ci, co), stored [component][ci][co].
function convWinoWeights(w, Cin = 64, Cout = 64) {
  const G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], u = new Float32Array(16 * Cin * Cout);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
    const k = G[i][a] * G[j][b];
    if (!k) continue;
    const s = (a * 3 + b) * Cin * Cout, o = (i * 4 + j) * Cin * Cout;
    for (let e = 0; e < Cin * Cout; e++) u[o + e] += k * w[s + e];
  }
  return u;
}

const castCode = (from, to) => `${from === 'f16' || to === 'f16' ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> X: array<${from}>;
@group(0) @binding(2) var<storage, read_write> Y: array<${to}>;
@group(0) @binding(3) var<uniform> d: vec4u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) { let i = gid.x; if (i < d.x) { Y[i] = ${to}(X[i]); } }`;

// R3GAN2 InterpolativeUpsampler: transposed conv, stride 2, separable taps [1,3,3,1]/4, optional per-channel gain.
function upCode(gain) {
  return `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;               // input h, w, C, output count
${gain ? '@group(0) @binding(4) var<storage, read> Gv: array<f32>;' : ''}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= d.w) { return; }
  var K = array<f32, 4>(0.25, 0.75, 0.75, 0.25);
  let C = d.z; let w2 = d.y * 2u; let c = i % C; let p = i / C;
  let ox = i32(p % w2); let oy = i32(p / w2);
  var acc = f32(0.0);
  for (var ky = 0; ky < 4; ky++) {
    let t = oy + 1 - ky; if (t < 0 || (t & 1) != 0) { continue; }
    let iy = t / 2; if (iy >= i32(d.x)) { continue; }
    for (var kx = 0; kx < 4; kx++) {
      let u = ox + 1 - kx; if (u < 0 || (u & 1) != 0) { continue; }
      let ix = u / 2; if (ix >= i32(d.y)) { continue; }
      acc += K[ky] * K[kx] * X[(u32(iy) * d.y + u32(ix)) * C + c];
    }
  }
  Y[i] = acc${gain ? ' * Gv[c]' : ''};
}`;
}

// Head basis: out[t, g*32+o] = lrelu(sum_c h[g*32+c] * wb[g, t, c, o]), 16 positions x `hidden` channels.
const basisCode = (hidden) => `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= ${16 * hidden}u) { return; }
  let o = i % 32u; let g = (i / 32u) % ${hidden / 32}u; let t = i / ${hidden}u;
  var v = 0.0;
  for (var c = 0u; c < 32u; c++) { v = fma(X[g * 32u + c], Wt[((g * 16u + t) * 32u + c) * 32u + o], v); }
  Y[i] = ${lrelu};
}`;

// R3GAN2 PolyphaseUpsampler, channel ratio 2 (Cin -> Cin/2): pre-filter U[c](Y, X) = gain * x[Y/2, X/2, 2c + s] with
// s = X%2 for even c, Y%2 for odd c (phase gather + pixel shuffle), then the 3x3 filter [1,2,1]^2/16, zero padded.
const upP2Code = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;               // input h, w, Cin, output count
@group(0) @binding(4) var<storage, read> Gv: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= d.w) { return; }
  var K = array<f32, 3>(0.25, 0.5, 0.25);
  let Co = d.z / 2u; let H2 = i32(d.x * 2u); let W2 = i32(d.y * 2u);
  let c = i % Co; let p = i / Co; let ox = i32(p % u32(W2)); let oy = i32(p / u32(W2));
  var acc = 0.0;
  for (var ky = 0; ky < 3; ky++) {
    let yy = oy + ky - 1; if (yy < 0 || yy >= H2) { continue; }
    for (var kx = 0; kx < 3; kx++) {
      let xx = ox + kx - 1; if (xx < 0 || xx >= W2) { continue; }
      let s = select(u32(yy & 1), u32(xx & 1), c % 2u == 0u); let src = 2u * c + s;
      acc += K[ky] * K[kx] * X[(u32(yy / 2) * d.y + u32(xx / 2)) * d.z + src] * Gv[src];
    }
  }
  Y[i] = acc;
}`;

// TAEF2 input: latent -> FLUX.2 BN-normalized latent (A, B) -> TAESD soft clamp tanh(x/3)*3.
const preCode = `
@group(0) @binding(0) var<storage, read> X: array<fp32>;
@group(0) @binding(1) var<storage, read> Av: array<fp32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<storage, read> Bv: array<fp32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= 32768u) { return; }
  Y[i] = f32(tanh(clamp((X[i] * Av[i] + Bv[i]) / 3.0, -15.0, 15.0)) * 3.0);
}`;

// GroupNorm(4 groups, 256 channels) statistics over all pixels: one workgroup per group.
const gnStatsCode = `
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<fp32>;
@group(0) @binding(3) var<uniform> d: vec4u;               // pixels
var<workgroup> s1: array<fp32, 256>;
var<workgroup> s2: array<fp32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let g = wg.x; let n = d.x * 64u;
  var a: fp32 = 0.0; var b: fp32 = 0.0;
  for (var e = li; e < n; e += 256u) { let v = fp32(X[(e / 64u) * 256u + g * 64u + e % 64u]); a += v; b += v * v; }
  s1[li] = a; s2[li] = b;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { s1[li] += s1[li + s]; s2[li] += s2[li + s]; } workgroupBarrier(); }
  if (li == 0u) { let m = s1[0] / fp32(n); Y[2u * g] = m; Y[2u * g + 1u] = inverseSqrt(max(s2[0] / fp32(n) - m * m, 0.0) + 1e-5); }
}`;
const gnNormCode = `
@group(0) @binding(0) var<storage, read> St: array<fp32>;
@group(0) @binding(1) var<storage, read> Gw: array<fp32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4u;               // element count
@group(0) @binding(4) var<storage, read> Gb: array<fp32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; if (i >= d.x) { return; }
  let c = i % 256u; let g = c / 64u;
  Y[i] = f32(max((fp32(Y[i]) - St[2u * g]) * St[2u * g + 1u] * Gw[c] + Gb[c], 0.0));
}`;
// TAEF2 kernels are written for f32 with 'fp32' where FP32 must stay; th() turns them into FP16 when HALF.
// Shared-memory vec4 arrays are declared array<wv4, N>, written as wput(v) and read as wget(x). In FP16 they hold
// vec2u: DXIL (Chrome on D3D12) has no vector types, so a vec4h shared-memory read became four 16-bit LDS loads, while
// a vec2u read stays one 64-bit load and the bitcast costs nothing (880M, Chrome: conv3w 9-15% faster). FP32: vec4f.
// That bitcast is valid WGSL, but naga (Firefox, wgpu 25-30) rejects bitcasts that change the component count, so
// createFastGAN tests it once (bitcastOK) and otherwise keeps vec4h in shared memory.
const BITCAST = 'alias wv4 = vec2u;\nfn wput(v: vec4h) -> vec2u { return bitcast<vec2u>(v); }\nfn wget(v: vec2u) -> vec4h { return bitcast<vec4h>(v); }\n';
const PLAIN = (v) => `alias wv4 = ${v};\nfn wput(v: ${v}) -> ${v} { return v; }\nfn wget(v: ${v}) -> ${v} { return v; }\n`;
let WV2U = false;  // set by createFastGAN
const th = (code, half) => half
  ? 'enable f16;\nalias fp32 = f32;\n' + (WV2U ? BITCAST : PLAIN('vec4h')) + code.replaceAll('f32', 'f16').replaceAll('vec4f', 'vec4h')
  : 'alias fp32 = f32;\n' + PLAIN('vec4f') + code;
async function bitcastOK(device) {
  device.pushErrorScope('validation');
  const m = device.createShaderModule({ code: `enable f16;\n${BITCAST}@compute @workgroup_size(1) fn main() { _ = wget(wput(vec4h(1.0))); }` });
  const info = await m.getCompilationInfo();
  const err = await device.popErrorScope();
  return !err && !info.messages.some((x) => x.type === 'error');
}

const VEC4 = !globalThis.location?.search?.includes('novec4');
const POLY = !globalThis.location?.search?.includes('nopoly');  // TAEF2 upsample convs as 4 phase convs
const GEMMT = !globalThis.location?.search?.includes('nogemmt');  // FP32 generator 1x1 convs: gemmTCode, not gemmVCode
const WINO = !globalThis.location?.search?.includes('nowino');  // TAEF2 32² convs as Winograd F(2x2, 3x3) (FP16)
export async function createFastGAN(device, man, wbin, { half = false, gHalf = false } = {}) {
  WV2U = (half || gHalf) && await bitcastOK(device);
  const T = man.tensors;
  const USE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const owned = [];  // every buffer this network creates, so destroy() can free them all
  const own = (b) => { owned.push(b); return b; };
  const buf = (n) => own(device.createBuffer({ size: Math.max(16, Math.ceil(n * 4 / 16) * 16), usage: USE }));
  const wcache = {};
  // Weight buffer for a manifest tensor, optionally transformed (convUpWeights, convWinoWeights); h: upload as FP16
  // (Float16Array: Chrome 135+).
  const W = (name, h = false, transform = null) => {
    const key = name + (transform ? ':' + transform.name : '') + (h ? ':h' : '');
    if (!wcache[key]) {
      if (!T[name]) throw new Error('missing weight ' + name);
      const [o, n] = T[name];
      const src = transform ? transform(new Float32Array(wbin, o * 4, n)) : new Float32Array(wbin, o * 4, n);
      const b = buf(src.length);
      if (h) { const a = new Float16Array(src.length + (src.length & 1)); a.set(src); device.queue.writeBuffer(b, 0, a); }  // writes are 4-byte multiples
      else device.queue.writeBuffer(b, 0, src);
      wcache[key] = b;
    }
    return wcache[key];
  };
  const pipes = {};
  // Pipelines compile asynchronously and in parallel (Chrome uses several threads): a cold build (no shader cache) took
  // about half as long as with createComputePipeline one at a time. Bind groups are made once each pipeline is ready.
  const pipe = (code) => pipes[code] ??= device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pending = [];
  const uni = (a) => { const b = own(device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })); device.queue.writeBuffer(b, 0, new Uint32Array([...a, 0, 0, 0, 0].slice(0, 4))); return b; };
  const ops = [];
  const op = (kind, code, binds, wx, wy = 1, wz = 1) => {
    const o = { kind, wx, wy, wz };
    ops.push(o);
    pending.push(pipe(code).then((p) => { o.p = p; o.bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: Object.entries(binds).map(([k, buffer]) => ({ binding: +k, resource: { buffer } })) }); }));
  };
  const PART = 1 << 18;  // FP32 partial sums for split-K GEMMs (floats)
  let part = null;
  const gemm = (kind, X, w, Y, P, K, N, f = {}, extra = {}) => {
    const b = { 0: X, 1: W(w, f.half), 2: Y, 3: uni([P, N, K, extra.off || 0]) };
    if (f.bias) b[4] = W(extra.bias, f.half);
    if (f.noise) { b[5] = W(extra.s, f.half); b[6] = noise; }
    if (GEMMT && !f.half && kind.startsWith('G 1x1') && P % 32 === 0 && N % 64 === 0 && K % 8 === 0) {
      // Split K until there are ~64 workgroups (8x8 and 16x16 images have only 16-64 tiles)
      let S = 1; while (S < 8 && (P / 32) * (N / 64) * S < 64 && K % (16 * S) === 0 && P * N * 2 * S <= PART) S *= 2;
      const e = { ...(f.bias && { 4: b[4] }), ...(f.noise && { 3: b[3], 5: b[5], 6: b[6] }) };
      if (S === 1) { op(kind, th(gemmTCode(P, N, K, 1, f), false), { 0: X, 1: b[1], 2: Y, ...e }, N / 64, P / 32); return; }
      part ??= buf(PART);
      op(kind, th(gemmTCode(P, N, K, S, f), false), { 0: X, 1: b[1], 7: part }, N / 64, P / 32, S);
      op(kind, th(gemmTReduce(P, N, S, f), false), { 2: Y, 7: part, ...e }, P * N / 256);
      return;
    }
    const vec = VEC4 && P % 64 === 0 && N % 64 === 0 && K % 16 === 0;
    op(kind, th(vec ? gemmVCode(f) : gemmCode(f), f.half), b, Math.ceil(N / 64), Math.ceil(P / 64));
  };
  const conv3 = (kind, X, w, Y, H, Wd, Cin, Cout, f = {}, bias) => {
    if (f.ups && POLY && Cin === 64 && Cout === 64 && !f.bias && !f.relu && !f.resid && (Wd / 2) % 16 === 0 && (H / 2) % 8 === 0) {
      op(kind, th(convUpCode(H / 2, Wd / 2), f.half), { 0: X, 1: W(w, f.half, convUpWeights), 2: Y }, (H / 16) * (Wd / 32), 4);
      return;
    }
    if (f.half && WINO && !f.ups && Cin === 64 && Cout === 64 && H * Wd <= 1024 && Wd % 8 === 0 && H % 4 === 0) {
      const b = { 0: X, 1: W(w, true, convWinoWeights), 2: Y };
      if (f.bias) b[3] = W(bias, true);
      op(kind, th(convWinoCode(H, Wd, f), true), b, (Wd / 8) * (H / 4));
      return;
    }
    const b = { 0: X, 1: W(w, f.half), 2: Y, 3: uni([H, Wd, Cin, Cout]) };
    if (f.bias) b[4] = W(bias, f.half);
    const wide = f.half && Cout === 64 && Wd % 16 === 0 && !globalThis.location?.search?.includes('nowide');
    // Cout is 64 everywhere except the final RGB layer (bias + clamp, no upsample or residual).
    op(kind, th(Cout === 3 ? convRGBCode() : wide ? conv3wCode(f) : VEC4 ? conv3vCode(f) : conv3gCode(f), f.half), b, (H / 8) * (Wd / (wide ? 16 : 8)));
  };
  const up = (X, Y, h, w, C, gain, hf = false) => {
    const b = { 0: X, 2: Y, 3: uni([h, w, C, 4 * h * w * C]) };
    if (gain) b[4] = W(gain, hf);
    op('G upsample', th(upCode(!!gain), hf), b, Math.ceil(4 * h * w * C / 256));
  };

  // ---- generator ----
  // Sizes come from the manifest; ImageNet-S manifests predate these fields, hence the defaults.
  const widths = man.widths, nb = man.blocks_per_stage, trans = man.transitions || widths.slice(1).map(() => 'i');
  const hidden = man.head_hidden || 1024, inDim = man.in_dim || 129, rgb = man.output === 'rgb';
  const inp = buf(inDim), noise = buf(man.noise_total || 5376);
  const h1 = buf(hidden), hb = buf(16 * hidden), h3 = buf(16 * widths[0]);
  gemm('G head', inp, 'head.w1', h1, 1, inDim, hidden, { lrelu: true });
  op('G head', basisCode(hidden), { 0: h1, 1: W('head.wb'), 2: hb }, Math.ceil(16 * hidden / 256));
  gemm('G head', hb, 'head.w3', h3, 16, hidden, widths[0]);
  let res = 8, x = buf(64 * widths[0]), off = 0;
  up(h3, x, 4, 4, widths[0]);
  const gh = gHalf && !rgb;  // generator trunk in FP16 (head and latent stay FP32); ImageNet only
  if (gh) { const x16 = buf(64 * widths[0]); op('G head', castCode('f32', 'f16'), { 0: x, 2: x16, 3: uni([64 * widths[0]]) }, Math.ceil(64 * widths[0] / 256)); x = x16; }
  let hid = 0; { let r = 8; for (const C of widths) { hid = Math.max(hid, r * r * 2 * C); r *= 2; } }
  const y1 = buf(hid), y2 = buf(hid);
  for (let s = 0; s < widths.length; s++) {
    const C = widths[s], P = res * res;
    for (let b = 0; b < nb[s]; b++) {
      const p = `s${s}b${b}`;
      gemm('G 1x1 in', x, p + '.w1', y1, P, C, 2 * C, { bias: true, noise: true, lrelu: true, half: gh }, { bias: p + '.b1', s: p + '.s1', off });
      off += P;
      const gc = gconvCfg(gh, P);
      op('G grouped 3x3', th(VEC4 ? gconvVCode(gc, gh) : gconvCode, gh), { 0: y1, 1: W(p + '.w2', gh), 2: y2, 3: uni([res, res, 2 * C]) }, VEC4 ? P / (gc.TW * gc.TH) : P / 64, 2 * C / 32);
      gemm('G 1x1 out', y2, p + '.w3', x, P, 2 * C, C, { resid: true, half: gh });
    }
    if (s < widths.length - 1) {
      const xn = buf(4 * P * widths[s + 1]);
      if (trans[s] === 'p2') op('G upsample', upP2Code, { 0: x, 2: xn, 3: uni([res, res, C, 4 * P * widths[s + 1]]), 4: W(`s${s}.tgain`) }, Math.ceil(4 * P * widths[s + 1] / 256));
      else up(x, xn, res, res, C, `s${s}.tgain`, gh);
      x = xn; res *= 2;
    }
  }
  const inBuf = new Float32Array(inDim); inBuf[inDim - 1] = 1;
  // The noise is uploaded only when it is a different array (the page makes a new one per noise seed or truncation);
  // for FFHQ it is ~350 KB, so re-sending it every frame was wasted work.
  let lastNoise = null;
  const setInputs = (z, c, n) => {
    inBuf.set(z, 0); if (c) inBuf.set(c, z.length); device.queue.writeBuffer(inp, 0, inBuf);
    if (n !== lastNoise) { device.queue.writeBuffer(noise, 0, n); lastNoise = n; }
  };
  if (rgb) {  // pixel-space model (FFHQ): the aggregation layer writes display RGB in [0,1] straight into the image
    const img = buf(res * res * 3);
    gemm('G head', x, 'agg.w', img, res * res, widths[widths.length - 1], 3, { bias: true, clamp: true }, { bias: 'agg.b' });
    await Promise.all(pending);
    return finish({ nG: ops.length, lat: null, img, setInputs });
  }
  const lat = buf(1024 * 32), K = widths[widths.length - 1];
  const head32 = (Y, h) => op('G head', th(head32Code, h), { 0: x, 1: W('agg.w', h), 2: Y, 3: uni([1024, 32, K]) }, 128);
  if (gh) { const l16 = buf(1024 * 32); head32(l16, true); op('G head', castCode('f16', 'f32'), { 0: l16, 2: lat, 3: uni([32768]) }, 128); }
  else head32(lat, false);
  const nG = ops.length;

  // ---- TAEF2 ----
  const t0 = buf(1024 * 32);
  const h = half;  // TAEF2 activations and weights in FP16; the latent and the image stay FP32
  op('T pre', th(preCode, h), { 0: lat, 1: W('t.A'), 2: t0, 4: W('t.B') }, 128);
  const big = () => buf(256 * 256 * 64);
  let cur = big(), t1 = big(), t2 = big(); res = 32;
  conv3('T conv 32²', t0, 't.c0.w', cur, 32, 32, 32, 64, { bias: true, relu: true, half: h }, 't.c0.b');
  const pbuf = buf(1024 * 256), st = buf(8);
  for (const i of man.taef2_blocks) {
    const k = `T conv ${res}²`;
    if (T[`t.b${i}.p1`]) {
      const P = res * res;
      gemm('T groupnorm pool', cur, `t.b${i}.p1`, pbuf, P, 64, 256, { half: h });
      op('T groupnorm pool', th(gnStatsCode, h), { 0: pbuf, 2: st, 3: uni([P]) }, 4);
      op('T groupnorm pool', th(gnNormCode, h), { 0: st, 1: W(`t.b${i}.gnw`), 2: pbuf, 3: uni([P * 256]), 4: W(`t.b${i}.gnb`) }, Math.ceil(P * 256 / 256));
      gemm('T groupnorm pool', pbuf, `t.b${i}.p2`, cur, P, 256, 64, { resid: true, half: h });
    }
    conv3(k, cur, `t.b${i}.c0.w`, t1, res, res, 64, 64, { bias: true, relu: true, half: h }, `t.b${i}.c0.b`);
    conv3(k, t1, `t.b${i}.c1.w`, t2, res, res, 64, 64, { bias: true, relu: true, half: h }, `t.b${i}.c1.b`);
    conv3(k, t2, `t.b${i}.c2.w`, cur, res, res, 64, 64, { bias: true, resid: true, relu: true, half: h }, `t.b${i}.c2.b`);
    const u = man.taef2_ups[String(i)];
    if (u) { conv3(`T conv ${2 * res}²`, cur, `t.${u}.w`, t1, 2 * res, 2 * res, 64, 64, { ups: true, half: h }); [cur, t1] = [t1, cur]; res *= 2; }
  }
  const img = buf(256 * 256 * 3);
  conv3('T conv 256²', cur, 't.out.w', img, 256, 256, 64, 3, { bias: true, clamp: true, half: h }, 't.out.b');

  await Promise.all(pending);
  return finish({ nG, lat, img, setInputs });

  function encode(enc, upTo, querySet, from = 0) {
    if (!querySet) {
      const pass = enc.beginComputePass();
      for (const o of ops.slice(from, upTo)) { pass.setPipeline(o.p); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(o.wx, o.wy, o.wz); }
      pass.end();
    } else {  // one pass per op, timestamped, for profiling
      ops.slice(from, upTo).forEach((o, k) => {
        const pass = enc.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
        pass.setPipeline(o.p); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(o.wx, o.wy, o.wz); pass.end();
      });
    }
  }
  // Shared tail for both model types (function declarations are hoisted, so the FFHQ early return can use it).
  function finish({ nG, lat, img, setInputs }) {
  const drawCode = `
  @group(0) @binding(0) var<storage, read> I: array<f32>;
  @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  }
  @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let k = (u32(pos.y) * 256u + u32(pos.x)) * 3u;
    return vec4f(I[k], I[k + 1u], I[k + 2u], 1.0);
  }`;
    return {
    ops, nG, lat, img, setInputs,
    // Draw the image buffer straight into a 256x256 WebGPU canvas: returns draw(encoder).
    attach(canvas) {
      const ctx = canvas.getContext('webgpu'), format = navigator.gpu.getPreferredCanvasFormat();
      // Configure once per canvas: reconfiguring clears it, which would flash black when switching models.
      if (canvas.__r3device !== device) { ctx.configure({ device, format, alphaMode: 'opaque' }); canvas.__r3device = device; }
      const m = device.createShaderModule({ code: drawCode });
      const rp = device.createRenderPipeline({ layout: 'auto', vertex: { module: m, entryPoint: 'vs' }, fragment: { module: m, entryPoint: 'fs', targets: [{ format }] } });
      const bg = device.createBindGroup({ layout: rp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: img } }] });
      return (enc) => {
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
        pass.setPipeline(rp); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
      };
    },
    // ops [from, upTo): run(nG) is the generator alone; run(ops.length, null, nG) decodes whatever is in lat.
    run(upTo = ops.length, querySet = null, from = 0) { const enc = device.createCommandEncoder(); encode(enc, upTo, querySet, from); return enc; },
    encodeOps(enc, from, upTo) { encode(enc, upTo, null, from); },  // ops [from, upTo) into an existing encoder
    destroy() { for (const b of owned) b.destroy(); owned.length = 0; },
    async read(b, n) {
      const s = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(b, 0, s, 0, n * 4); device.queue.submit([enc.finish()]);
      await s.mapAsync(GPUMapMode.READ); const out = new Float32Array(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return out;
    },
  };
  }
}

