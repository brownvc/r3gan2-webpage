// Tests for globe.js (the class globe: tiling, class field, noise directions). Run: node tests/globe_test.mjs
import assert from 'node:assert/strict';
import { geodesic, cellAt, classMix, directions, tileZ, noiseField, noisePoint } from '../globe.js';

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
    const M = [0, 1, 2].map((a) => [0, 1, 2].map((b) => dot(row(V, a), row(R0, b))));
    const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    assert.ok(det > 0, 'mirrored');
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
  tileZ() {
    const { V } = directions({ 0: 1 }, synthAxes(4), E3), p = unit([0.3, -0.5, 0.8]);
    assert.ok(tileZ(V, p, 0).every((x) => x === 0));
    assert.ok(Math.abs(norm(Array.from(tileZ(V, p, 1))) - 8) < 1e-4);
  },
};
let failed = 0;
for (const [name, t] of Object.entries(tests)) {
  try { t(); console.log('ok  ', name); } catch (e) { failed++; console.log('FAIL', name, '-', e.message); }
}
console.log(failed ? `${failed} failed` : 'all passed'); process.exit(failed ? 1 : 0);
