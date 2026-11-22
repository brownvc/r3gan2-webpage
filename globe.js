// The class globe's geometry and maths, DOM-free (tests/globe_test.mjs):
//  - geodesic(levels): the fixed tiling, an icosahedron subdivided `levels` times; one tile per vertex, its cell the
//    polygon of the circumcentres of the triangles around it (its Voronoi cell) (hexagons, and 12 pentagons at the icosahedron's corners).
//  - classMix(points, p, soft, blocked): the class mix at a point of the globe, from the class layout (class_layout.py).
//  - directions(mix, axes, R0): the mix's three strongest noise directions (noise_axes.py), in a fixed orientation.
//  - tileZ(V, p, psi): a tile's z, the noise minimap's point p along those directions, at length psi * 8.

export function geodesic(levels) {
  const t = (1 + Math.sqrt(5)) / 2;
  const P = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const n = 10 * 4 ** levels + 2, V = new Float64Array(3 * n);
  let nv = 0;
  const add = (x, y, z) => { const r = Math.hypot(x, y, z); V[3 * nv] = x / r; V[3 * nv + 1] = y / r; V[3 * nv + 2] = z / r; return nv++; };
  for (const p of P) add(...p);
  for (let l = 0; l < levels; l++) {
    const mid = new Map(), nf = [];
    const m = (a, b) => {
      const k = a < b ? a * n + b : b * n + a; let i = mid.get(k);
      if (i === undefined) { i = add(V[3 * a] + V[3 * b], V[3 * a + 1] + V[3 * b + 1], V[3 * a + 2] + V[3 * b + 2]); mid.set(k, i); }
      return i;
    };
    for (const [a, b, c] of F) { const ab = m(a, b), bc = m(b, c), ca = m(c, a); nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    F = nf;
  }
  // Triangles around each vertex, then ordered counter-clockwise seen from outside.
  const deg = new Uint32Array(n);
  for (const f of F) for (const v of f) deg[v]++;
  const cellStart = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) cellStart[i + 1] = cellStart[i] + deg[i];
  const fill = new Uint32Array(n), around = new Uint32Array(cellStart[n]), cen = new Float64Array(3 * F.length);
  F.forEach(([a, b, c], f) => {  // the triangle's circumcentre on the sphere, (b - a) x (c - a) normalised outwards
    const e = [0, 1, 2].map((k) => V[3 * b + k] - V[3 * a + k]), h = [0, 1, 2].map((k) => V[3 * c + k] - V[3 * a + k]);
    let x = e[1] * h[2] - e[2] * h[1], y = e[2] * h[0] - e[0] * h[2], z = e[0] * h[1] - e[1] * h[0];
    const r = Math.hypot(x, y, z) * Math.sign(x * V[3 * a] + y * V[3 * a + 1] + z * V[3 * a + 2]);
    cen[3 * f] = x / r; cen[3 * f + 1] = y / r; cen[3 * f + 2] = z / r;
    for (const v of [a, b, c]) around[cellStart[v] + fill[v]++] = f;
  });
  const corners = new Float32Array(3 * cellStart[n]);
  for (let i = 0; i < n; i++) {
    const cx = V[3 * i], cy = V[3 * i + 1], cz = V[3 * i + 2];
    // a tangent frame at the vertex: u any perpendicular, w = c x u
    let ux = -cy, uy = cx, uz = 0; if (Math.abs(cz) > 0.9) { ux = 0; uy = -cz; uz = cy; }
    const ur = Math.hypot(ux, uy, uz); ux /= ur; uy /= ur; uz /= ur;
    const wx = cy * uz - cz * uy, wy = cz * ux - cx * uz, wz = cx * uy - cy * ux;
    const fs = Array.from(around.subarray(cellStart[i], cellStart[i + 1]));
    const ang = (f) => Math.atan2(cen[3 * f] * wx + cen[3 * f + 1] * wy + cen[3 * f + 2] * wz, cen[3 * f] * ux + cen[3 * f + 1] * uy + cen[3 * f + 2] * uz);
    fs.sort((a, b) => ang(a) - ang(b));
    fs.forEach((f, j) => corners.set(cen.subarray(3 * f, 3 * f + 3), 3 * (cellStart[i] + j)));
  }
  return { n, centres: Float32Array.from(V), cellStart, corners, faces: Uint32Array.from(F.flat()), around };
}

// The noise field over the globe: where the map blends random anchors in 64 dimensions (nearly orthogonal there, so
// a blend never comes near zero), here the noise point is 3-D, where two random points can be nearly opposite and a
// blend between them would flip. So each tile's noise point is e3 turned by a rotation that varies smoothly over the
// surface: rotation vector w(p) = sum of M random waves b_m sin(k_m . p + phi_m), |k_m| = K (wavelength 2 pi / K on
// the unit sphere: K = 26 turns the point ~45 degrees over 4 tiles at the median, steps of at most ~60 degrees per tile).
// Always unit length, continuous everywhere. Seeded, so a link reproduces it.
export function noiseField(seed, K = 26, M = 12) {
  let s = seed >>> 0;
  const rand = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const waves = [], amp = 2 / Math.sqrt(M / 2);  // the rotation angle's spread ~2 radians
  for (let m = 0; m < M; m++) {
    const k = [gauss(), gauss(), gauss()], r = Math.hypot(...k);
    waves.push({ k: k.map((x) => (K * x) / r), phi: 2 * Math.PI * rand(), b: [gauss(), gauss(), gauss()].map((x) => amp * x) });
  }
  return { waves };
}

export function noisePoint(F, p) {
  const w = [0, 0, 0];
  for (const { k, phi, b } of F.waves) { const s = Math.sin(k[0] * p[0] + k[1] * p[1] + k[2] * p[2] + phi); w[0] += b[0] * s; w[1] += b[1] * s; w[2] += b[2] * s; }
  const th = Math.hypot(...w); if (th < 1e-12) return [0, 0, 1];
  const a = w.map((x) => x / th), c = Math.cos(th), sn = Math.sin(th);  // Rodrigues: rotate e3 about a by th
  return [a[0] * a[2] * (1 - c) + a[1] * sn, a[1] * a[2] * (1 - c) - a[0] * sn, c + a[2] * a[2] * (1 - c)];
}

// ponytail: brute force over all centres (~0.5 ms); a spatial index if it is ever called per frame
export function cellAt(g, p) {
  let best = -1, bd = -2;
  for (let i = 0; i < g.n; i++) { const d = g.centres[3 * i] * p[0] + g.centres[3 * i + 1] * p[1] + g.centres[3 * i + 2] * p[2]; if (d > bd) { bd = d; best = i; } }
  return best;
}

// The class mix at unit vector p. Each class owns the part of the globe nearest its point (angles); near a border its
// neighbours blend in: weight exp(-(a^2 - amin^2) / (soft * S^2)) over the 6 nearest classes, S the typical spacing of
// 1000 points on the sphere (soft 0: hard borders). Weights under 1% are dropped; the rest sum to 1. Blocked classes
// own nothing: their neighbours take their place.
const SPACING = Math.sqrt(4 * Math.PI / 1000);
export function classMix(points, p, soft, blocked) {
  const near = [];  // [angle^2, class], the 6 nearest, sorted
  for (let c = 0; c < points.length / 3; c++) {
    if (blocked.has(c)) continue;
    const a = Math.acos(Math.max(-1, Math.min(1, points[3 * c] * p[0] + points[3 * c + 1] * p[1] + points[3 * c + 2] * p[2])));
    const a2 = a * a;
    if (near.length < 6 || a2 < near[near.length - 1][0]) { near.push([a2, c]); near.sort((x, y) => x[0] - y[0]); if (near.length > 6) near.pop(); }
  }
  const mix = {}; let s = 0;
  for (const [a2, c] of near) {
    const w = soft > 0 ? Math.exp(-(a2 - near[0][0]) / (soft * SPACING * SPACING)) : (c === near[0][1] ? 1 : 0);
    if (w >= 0.01) { mix[c] = w; s += w; }
  }
  for (const c in mix) mix[c] /= s;
  return mix;
}

// Symmetric eigendecomposition (cyclic Jacobi) of an n x n matrix A (row-major, overwritten): eigenvalues on its
// diagonal, eigenvectors in the columns of the returned matrix.
function jacobi(A, n) {
  const Q = new Float64Array(n * n); for (let i = 0; i < n; i++) Q[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i * n + j] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {  // A <- J^T A J, Q <- Q J
        const akp = A[k * n + p], akq = A[k * n + q]; A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p * n + k], aqk = A[q * n + k]; A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) { const qkp = Q[k * n + p], qkq = Q[k * n + q]; Q[k * n + p] = c * qkp - s * qkq; Q[k * n + q] = s * qkp + c * qkq; }
    }
  }
  return Q;
}

// The three strongest noise directions of a class mix, as 3 rows of 64 (Float32Array(192)), and the share of the
// output's variation they carry. The mix's metric is M = sum_k w_k U_k diag(lambda_k) U_k^T over the stored top-8
// factors, so M = B^T B with B's rows sqrt(w_k lambda_kj) u_kj; its top eigenvectors come from the small Gram matrix
// B B^T. They are then turned by the orthogonal map that brings them closest to the reference basis R0 (the polar
// factor; mirroring allowed: the orientation of the subspace means nothing, and allowing it makes the result
// independent of the eigenvectors' arbitrary signs), so the orientation depends on the mix alone, never on how the
// view got there or on signs, and neighbouring blends do not jump.
export function directions(mix, axes, R0) {
  const Z = 64, rows = [];
  let tr = 0;
  for (const k in mix) {
    const w = mix[k]; tr += w * axes.trace[k];
    for (let j = 0; j < 8; j++) {
      const s = Math.sqrt(Math.max(0, w * axes.lambda[k * 8 + j])), o = (k * 8 + j) * Z;
      rows.push(Float64Array.from(axes.U.subarray(o, o + Z), (x) => x * s));
    }
  }
  const r = rows.length, G = new Float64Array(r * r);
  for (let i = 0; i < r; i++) for (let j = i; j < r; j++) { let d = 0; for (let t = 0; t < Z; t++) d += rows[i][t] * rows[j][t]; G[i * r + j] = G[j * r + i] = d; }
  const Q = jacobi(G, r), ev = Array.from({ length: r }, (_, i) => [G[i * r + i], i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const V = new Float64Array(3 * Z);
  ev.forEach(([l, i], a) => {  // eigenvector of M: B^T q / sqrt(l)
    const f = 1 / Math.sqrt(Math.max(l, 1e-30));
    for (let j = 0; j < r; j++) { const q = Q[j * r + i] * f; for (let t = 0; t < Z; t++) V[a * Z + t] += q * rows[j][t]; }
  });
  // Orthogonal Procrustes: Q3 = argmin |Q3 V - R0| over orthogonal maps; with C = R0 V^T = X S Y^T, Q3 = X Y^T.
  const C = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { let d = 0; for (let t = 0; t < Z; t++) d += R0[a * Z + t] * V[b * Z + t]; C[a * 3 + b] = d; }
  const CtC = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let k = 0; k < 3; k++) CtC[a * 3 + b] += C[k * 3 + a] * C[k * 3 + b];
  const Y = jacobi(CtC, 3), X = new Float64Array(9);
  for (let j = 0; j < 3; j++) {  // X's column j = C y_j / s_j (an orthonormal completion if s_j is ~0)
    const s = Math.sqrt(Math.max(CtC[j * 3 + j], 0));
    for (let a = 0; a < 3; a++) { let d = 0; for (let k = 0; k < 3; k++) d += C[a * 3 + k] * Y[k * 3 + j]; X[a * 3 + j] = s > 1e-9 ? d / s : 0; }
  }
  for (let j = 0; j < 3; j++) if (Math.hypot(X[j], X[3 + j], X[6 + j]) < 0.5) {  // degenerate: complete with a cross product
    const o1 = (j + 1) % 3, o2 = (j + 2) % 3;
    X[j] = X[3 + o1] * X[6 + o2] - X[6 + o1] * X[3 + o2]; X[3 + j] = X[6 + o1] * X[o2] - X[o1] * X[6 + o2]; X[6 + j] = X[o1] * X[3 + o2] - X[3 + o1] * X[o2];
  }
  const Q3 = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) Q3[a * 3 + b] = X[a * 3] * Y[b * 3] + X[a * 3 + 1] * Y[b * 3 + 1] + X[a * 3 + 2] * Y[b * 3 + 2];
  const out = new Float32Array(3 * Z);
  for (let a = 0; a < 3; a++) for (let t = 0; t < Z; t++) out[a * Z + t] = Q3[a * 3] * V[t] + Q3[a * 3 + 1] * V[Z + t] + Q3[a * 3 + 2] * V[2 * Z + t];
  return { V: out, share: (ev[0][0] + ev[1][0] + ev[2][0]) / tr };
}

// The image size a cell draws at: 0 (16 px), 1 (64 px) or 2 (256 px) by its size on screen, but only a tier whose
// store holds the cells in view (with room to spare): a smaller one would evict tiles in view and remake them forever.
export function tierFor(cellPx, inView, caps) {
  let t = cellPx > 100 ? 2 : cellPx > 24 ? 1 : 0;
  while (t > 0 && inView > 0.8 * caps[t]) t--;
  return t;
}

export function tileZ(V, p, psi, z0) {
  // A typical sample: the seed's z0 (N(0, I)) with its part along the 3 directions replaced by the noise point at
  // length sqrt 3, a typical length for 3 of z's dimensions. (All of z's length 8 in 3 directions is 8 sigma out.)
  const d = [0, 1, 2].map((a) => { let s = 0; for (let t = 0; t < 64; t++) s += V[a * 64 + t] * z0[t]; return Math.sqrt(3) * p[a] - s; });
  const z = new Float32Array(64);
  for (let t = 0; t < 64; t++) z[t] = psi * (z0[t] + d[0] * V[t] + d[1] * V[64 + t] + d[2] * V[128 + t]);
  return z;
}

// Each cell's neighbours (the other ends of its edges), 6 per cell, -1 padding at the 12 pentagons.
export function neighbours(g) {
  const nb = new Int32Array(6 * g.n).fill(-1), add = (a, b) => {
    for (let k = 0; k < 6; k++) { const v = nb[6 * a + k]; if (v === b) return; if (v < 0) { nb[6 * a + k] = b; return; } }
  };
  for (let f = 0; f < g.faces.length; f += 3) {
    const [a, b, c] = [g.faces[f], g.faces[f + 1], g.faces[f + 2]];
    add(a, b); add(b, a); add(b, c); add(c, b); add(c, a); add(a, c);
  }
  return nb;
}

// Close in, the cells in view reshape into a rectangular grid (gridTargets gives each its place; the shader moves it).
// Locally the cells form a hex lattice: each gets lattice coordinates (a, b) by walking out from cells already labelled
// (st.lab keeps them from frame to frame, so the grid holds still while the view pans), along steps e1 (the lattice
// direction nearest the screen's x, picked once) and e2 (e1 turned 60 degrees). A hex lattice's rows are offset by
// half a cell from row to row; shifting even rows a quarter cell one way and odd rows the other lines the columns up.
// view: cell ids; pos: Map id -> [x, y] on screen (y up); anchor: the cell at the view's centre, placed where it is
// (plus its row's shift). Returns Map id -> { T: [x, y], w } (w 0: stays a hexagon, as two cells claiming one place
// near a pentagon do, or one that would move more than a cell). st.e1 is the lattice step on screen, for the shader.
const SQ3 = Math.sqrt(3) / 2;
export function gridTargets(view, pos, nb, st, anchor) {
  const inView = new Set(view);
  st.lab ??= new Map();
  for (const c of [...st.lab.keys()]) if (!inView.has(c)) st.lab.delete(c);
  const P = pos.get(anchor);
  if (!st.lab.size || !st.e1) {
    st.lab.clear(); st.lab.set(anchor, [0, 0]); let best = -2;
    for (let k = 0; k < 6; k++) {
      const v = nb[6 * anchor + k]; if (v < 0 || !pos.has(v)) continue;
      const d = [pos.get(v)[0] - P[0], pos.get(v)[1] - P[1]], c = d[0] / Math.hypot(...d);
      if (c > best) { best = c; st.e1 = d; }
    }
  }
  const e1 = st.e1, e2 = [0.5 * e1[0] - SQ3 * e1[1], SQ3 * e1[0] + 0.5 * e1[1]];
  const steps = [[1, 0, e1], [0, 1, e2], [-1, 1, [e2[0] - e1[0], e2[1] - e1[1]]], [-1, 0, [-e1[0], -e1[1]]], [0, -1, [-e2[0], -e2[1]]], [1, -1, [e1[0] - e2[0], e1[1] - e2[1]]]];
  const queue = view.filter((c) => st.lab.has(c));
  for (let qi = 0; qi < queue.length; qi++) {
    const u = queue[qi], lu = st.lab.get(u), pu = pos.get(u);
    for (let k = 0; k < 6; k++) {
      const v = nb[6 * u + k]; if (v < 0 || !inView.has(v) || st.lab.has(v)) continue;
      const d = [pos.get(v)[0] - pu[0], pos.get(v)[1] - pu[1]]; let bs = -Infinity, s = steps[0];
      for (const t of steps) { const q = d[0] * t[2][0] + d[1] * t[2][1]; if (q > bs) { bs = q; s = t; } }
      st.lab.set(v, [lu[0] + s[0], lu[1] + s[1]]); queue.push(v);
    }
  }
  // the lattice step as it is on screen now, from labelled neighbour pairs (it turns slowly as the view pans)
  let sx = 0, sy = 0, n = 0;
  for (const c of view) {
    const l = st.lab.get(c); if (!l) continue;
    for (let k = 0; k < 6; k++) {
      const v = nb[6 * c + k], lv = v >= 0 && st.lab.get(v);
      if (lv && lv[0] === l[0] + 1 && lv[1] === l[1]) { sx += pos.get(v)[0] - pos.get(c)[0]; sy += pos.get(v)[1] - pos.get(c)[1]; n++; }
    }
  }
  if (n) st.e1 = [sx / n, sy / n];
  const L = Math.hypot(...st.e1), u = [st.e1[0] / L, st.e1[1] / L], w = [-u[1], u[0]];
  const sh = (b) => ((b % 2) + 2) % 2 ? -0.25 : 0.25, la = st.lab.get(anchor) || [0, 0], out = new Map(), claim = new Map();
  for (const c of view) {
    const l = st.lab.get(c); if (!l) { out.set(c, { T: pos.get(c), w: 0 }); continue; }
    const da = l[0] - la[0], db = l[1] - la[1], x = L * (da + 0.5 * db + sh(l[1])), y = L * SQ3 * db;
    const T = [P[0] + x * u[0] + y * w[0], P[1] + x * u[1] + y * w[1]], key = `${l[0]},${l[1]}`;
    const far = Math.hypot(T[0] - pos.get(c)[0], T[1] - pos.get(c)[1]) / L;
    out.set(c, { T, w: Math.max(0, Math.min(1, (1 - far) / 0.25)) });
    claim.set(key, (claim.get(key) || 0) + 1);
  }
  for (const c of view) { const l = st.lab.get(c); if (l && claim.get(`${l[0]},${l[1]}`) > 1) out.get(c).w = 0; }
  return out;
}
