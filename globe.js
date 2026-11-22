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
  return { n, centres: Float32Array.from(V), cellStart, corners };
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
// B B^T. They are then turned by the rotation (det +1) that brings them closest to the reference basis R0, so the
// orientation depends on the mix alone, never on how the view got there.
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
  // Orthogonal Procrustes: Q3 = argmin |Q3 V - R0| over rotations; with C = R0 V^T = X S Y^T, Q3 = X D Y^T.
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
  const det = (M) => M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
  const d = Math.sign(det(X) * det(Y)) || 1, Q3 = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) Q3[a * 3 + b] = X[a * 3] * Y[b * 3] + X[a * 3 + 1] * Y[b * 3 + 1] + d * X[a * 3 + 2] * Y[b * 3 + 2];
  const out = new Float32Array(3 * Z);
  for (let a = 0; a < 3; a++) for (let t = 0; t < Z; t++) out[a * Z + t] = Q3[a * 3] * V[t] + Q3[a * 3 + 1] * V[Z + t] + Q3[a * 3 + 2] * V[2 * Z + t];
  return { V: out, share: (ev[0][0] + ev[1][0] + ev[2][0]) / tr };
}

export function tileZ(V, p, psi) {
  const z = new Float32Array(64), f = psi * 8;
  for (let t = 0; t < 64; t++) z[t] = f * (p[0] * V[t] + p[1] * V[64 + t] + p[2] * V[128 + t]);
  return z;
}
