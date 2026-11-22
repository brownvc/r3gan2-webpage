// Tests for globe.js (the class globe: tiling, class field, noise directions). Run: node tests/globe_test.mjs
import assert from 'node:assert/strict';
import { geodesic, cellAt, classMix, directions, tileZ, noiseField, noisePoint, tierFor } from '../globe.js';

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => { const n = norm(a); return a.map((x) => x / n); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const triArea = (a, b, c) => 2 * Math.atan2(Math.abs(dot(a, cross(b, c))), 1 + dot(a, b) + dot(b, c) + dot(c, a));
const vec = (arr, i) => [arr[3 * i], arr[3 * i + 1], arr[3 * i + 2]];
const row = (V, a) => Array.from(V.subarray(a * 64, a * 64 + 64));

// Synthetic noise directions for C classes: class c's 8 directions are e_{c..c+7} (mod 64), strengths 8..1.
function synthAxes(C) {
  const trace = new Float32Array(C).fill(40), lambda = new Float32Array(C * 8), U = new Float32Array(C * 8 * 64);
  for (let c = 0; c < C; c++) for (let j = 0; j < 8; j++) { lambda[c * 8 + j] = 8 - j; U[(c * 8 + j) * 64 + ((c + j) % 64)] = 1; }
  return { trace, lambda, U };
}
const E3 = new Float32Array(192); E3[0] = 1; E3[65] = 1; E3[130] = 1;  // reference basis e1, e2, e3

const g = geodesic(7);
const tests = {
  tiling() {
    assert.equal(g.n, 163842);
    let five = 0, six = 0, amin = Infinity, amax = 0;
    for (let i = 0; i < g.n; i++) {
      const k = g.cellStart[i + 1] - g.cellStart[i]; if (k === 5) five++; else if (k === 6) six++;
      const c = vec(g.centres, i); let A = 0;
      for (let j = 0; j < k; j++) A += triArea(c, vec(g.corners, g.cellStart[i] + j), vec(g.corners, g.cellStart[i] + (j + 1) % k));
      amin = Math.min(amin, A); amax = Math.max(amax, A);
    }
    assert.equal(five, 12); assert.equal(six, g.n - 12);
    assert.ok(amax / amin <= 1.4, `area ratio ${amax / amin}`);
  },
  cornersCounterClockwise() {  // seen from outside: (corner j - centre) x (corner j+1 - centre) points outwards
    for (let i = 0; i < g.n; i += 997) {
      const c = vec(g.centres, i), k = g.cellStart[i + 1] - g.cellStart[i];
      for (let j = 0; j < k; j++) {
        const a = vec(g.corners, g.cellStart[i] + j).map((x, t) => x - c[t]), b = vec(g.corners, g.cellStart[i] + (j + 1) % k).map((x, t) => x - c[t]);
        assert.ok(dot(cross(a, b), c) > 0, `cell ${i} corner ${j}`);
      }
    }
  },
  cellAt() {
    for (let i = 0; i < g.n; i += 163) assert.equal(cellAt(g, vec(g.centres, i)), i);
  },
  classMix() {
    const pts = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0]);  // four classes
    const deep = classMix(pts, unit([1, 0.01, 0.02]), 0.1, new Set());
    assert.deepEqual(Object.keys(deep), ['0']); assert.equal(deep[0], 1);
    const mid = classMix(pts, unit([1, 1, 0]), 0.1, new Set());
    assert.deepEqual(Object.keys(mid).sort(), ['0', '1']);
    assert.ok(Math.abs(mid[0] - 0.5) < 0.05 && Math.abs(mid[1] - 0.5) < 0.05, JSON.stringify(mid));
    const blocked = classMix(pts, unit([1, 0.01, 0.02]), 0.1, new Set([0]));
    assert.ok(!('0' in blocked), JSON.stringify(blocked));
    const hard = classMix(pts, unit([1, 1, 0.001]), 0, new Set());
    assert.equal(Object.keys(hard).length, 1);
  },
  directionsPure() {
    const { V, share } = directions({ 0: 1 }, synthAxes(4), E3);
    for (let a = 0; a < 3; a++) {
      const r = row(V, a); assert.ok(Math.hypot(r[0], r[1], r[2]) > 0.999, `row ${a} leaves e1..e3`);
      for (let b = 0; b < 3; b++) assert.ok(Math.abs(dot(r, row(V, b)) - (a === b)) < 1e-5, 'orthonormal');
    }
    assert.ok(Math.abs(share - 21 / 40) < 1e-6, `share ${share}`);
  },
  directionsAligned() {  // rotated to the reference: here the reference spans the same space, so V = R0
    const R0 = new Float32Array(192); const s = Math.SQRT1_2;
    R0[0] = s; R0[1] = s; R0[64] = -s; R0[65] = s; R0[130] = 1;  // a rotation of e1, e2, e3 (det +1)
    const { V } = directions({ 0: 1 }, synthAxes(4), R0);
    for (let a = 0; a < 3; a++) assert.ok(dot(row(V, a), row(R0, a)) > 0.999, `row ${a} not aligned`);
  },
  directionsIgnoreSigns() {  // an eigenvector's sign is arbitrary: flipping a stored one must not change V
    let s = 7; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) - 0.5;
    for (let trial = 0; trial < 50; trial++) {
      const ax = { trace: new Float32Array(2).fill(40), lambda: Float32Array.from({ length: 16 }, (_, j) => 8 - (j % 8)), U: new Float32Array(2 * 512) };
      for (let r = 0; r < 16; r++) { const v = unit(Array.from({ length: 64 }, rnd)); ax.U.set(v, r * 64); }
      const R0 = new Float32Array(192); for (let a = 0; a < 3; a++) R0.set(unit(Array.from({ length: 64 }, rnd)), a * 64);
      const mix = { 0: 0.6, 1: 0.4 }, A = directions(mix, ax, R0).V;
      const flipped = { ...ax, U: Float32Array.from(ax.U) }; for (let t = 0; t < 64; t++) flipped.U[trial % 16 * 64 + t] *= -1;
      const B = directions(mix, flipped, R0).V;
      const d = Math.max(...A.map((x, i) => Math.abs(x - B[i])));
      assert.ok(d < 1e-4, `trial ${trial}: V changed by ${d} when one stored eigenvector's sign flipped`);
    }
  },
  tierFor() {  // a resolution tier is used only while the cells in view fit in it, or tiles would be remade forever
    assert.equal(tierFor(150, 300, [Infinity, 8192, 512]), 2);
    assert.equal(tierFor(150, 1100, [Infinity, 8192, 512]), 1);
    assert.equal(tierFor(150, 9000, [Infinity, 8192, 512]), 0);
    assert.equal(tierFor(50, 300, [Infinity, 8192, 512]), 1);
    assert.equal(tierFor(10, 300, [Infinity, 8192, 512]), 0);
  },
  directionsThreeClasses() {
    const { V, share } = directions({ 0: 0.5, 1: 0.3, 2: 0.2 }, synthAxes(4), E3);
    assert.ok(V.every(Number.isFinite) && share > 0 && share <= 1, `share ${share}`);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) assert.ok(Math.abs(dot(row(V, a), row(V, b)) - (a === b)) < 1e-4, 'orthonormal');
  },
  noiseField() {  // the noise point varies smoothly over the globe, every few tiles, and covers the noise sphere
    const F = noiseField(123), p0 = vec(g.centres, 5);
    assert.deepEqual(noisePoint(noiseField(123), p0), noisePoint(F, p0), 'same seed, same field');
    assert.notDeepEqual(noisePoint(noiseField(124), p0), noisePoint(F, p0), 'another seed, another field');
    const sp = Math.sqrt(4 * Math.PI / g.n), a1 = [], a4 = [], mean = [0, 0, 0];  // sp: the tile spacing
    for (let i = 0; i < g.n; i += 37) {
      const c = vec(g.centres, i), d = unit(vec(g.corners, g.cellStart[i]).map((x, t) => x - c[t]));
      const at = (s) => unit(c.map((x, t) => x + s * sp * d[t])), q = noisePoint(F, c);
      assert.ok(Math.abs(norm(q) - 1) < 1e-6, 'unit');
      a1.push(Math.acos(Math.min(1, dot(q, noisePoint(F, at(1)))))); a4.push(Math.acos(Math.min(1, dot(q, noisePoint(F, at(4))))));
      for (let t = 0; t < 3; t++) mean[t] += q[t] / Math.ceil(g.n / 37);
    }
    a1.sort((x, y) => x - y); a4.sort((x, y) => x - y); const deg = (x) => (x * 180) / Math.PI;
    assert.ok(deg(a1[a1.length - 1]) < 65, `a one-tile step turns the noise point by up to ${deg(a1[a1.length - 1])} degrees`);
    assert.ok(deg(a4[a4.length >> 1]) > 35, `over 4 tiles the median turn is only ${deg(a4[a4.length >> 1])} degrees`);
    assert.ok(norm(mean) < 0.3, `noise points bunched: mean resultant ${norm(mean)}`);
  },
  tileZ() {  // a typical sample: the seed's z, except along the 3 directions, where it is the noise point at length sqrt 3
    const { V } = directions({ 0: 1 }, synthAxes(4), E3), p = unit([0.3, -0.5, 0.8]);
    const z0 = Float32Array.from({ length: 64 }, (_, t) => Math.sin(7 * t + 1) * 1.4);
    assert.ok(tileZ(V, p, 0, z0).every((x) => x === 0));
    const z = Array.from(tileZ(V, p, 1, z0));
    for (let a = 0; a < 3; a++) assert.ok(Math.abs(dot(z, row(V, a)) - Math.sqrt(3) * p[a]) < 1e-5, `along direction ${a}: ${dot(z, row(V, a))}`);
    const rest = (v) => { const o = v.slice(); for (let a = 0; a < 3; a++) { const d = dot(v, row(V, a)); for (let t = 0; t < 64; t++) o[t] -= d * V[a * 64 + t]; } return o; };
    assert.ok(Math.max(...rest(z).map((x, t) => Math.abs(x - rest(Array.from(z0))[t]))) < 1e-5, 'elsewhere z is the seed’s');
  },
  parents() {  // every finer vertex has a parent in a coarser level, about one of its own spacings away
    const levelOf = (i) => { let k = 0; while (i >= 10 * 4 ** k + 2) k++; return k; };
    for (let i = 12; i < g.n; i += 13) {
      const k = levelOf(i), j = g.parent[i];
      assert.ok(levelOf(j) < k, `vertex ${i} (level ${k}): parent ${j} is not coarser`);
      const ang = Math.acos(Math.min(1, dot(vec(g.centres, i), vec(g.centres, j))));
      assert.ok(ang < 1.5 * Math.sqrt(4 * Math.PI / (10 * 4 ** k + 2)), `vertex ${i}: parent ${ang} rad away`);
    }
  },
};
let failed = 0;
for (const [name, t] of Object.entries(tests)) {
  try { t(); console.log('ok  ', name); } catch (e) { failed++; console.log('FAIL', name, '-', e.message); }
}
console.log(failed ? `${failed} failed` : 'all passed'); process.exit(failed ? 1 : 0);
