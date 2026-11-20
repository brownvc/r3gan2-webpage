// Tests for path.js (the map's path tool: pen, path, playhead). Run: node tests/path_test.mjs
import assert from 'node:assert/strict';
import { Path, Playhead } from '../path.js';

const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);
const draw = (path, pts, step = 0.1) => { path.penDown(pts[0][0], pts[0][1], step); for (const [x, y] of pts.slice(1)) path.penMove(x, y); };
const sweep = (x0, y0, x1, y1, n) => Array.from({ length: n + 1 }, (_, k) => [x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n]);
const tests = {
  spacing() {
    const p = new Path({ step: 0.1 }); draw(p, sweep(0, 0, 10, 0, 200)); p.penUp();
    assert.ok(p.points.length > 80, `only ${p.points.length} points`);
    for (let k = 1; k < p.points.length; k++) assert.ok(Math.abs(dist(p.points[k], p.points[k - 1]) - 0.1) <= 0.01, `gap ${k}`);
  },
  bridge() {
    const p = new Path({ step: 0.1 }); draw(p, sweep(0, 0, 1, 0, 40)); for (let k = 0; k < 30; k++) p.penMove(1, 0); p.penUp();
    const n = p.points.length, tip = p.points[n - 1];
    assert.ok(Math.abs(tip.x - 1) <= 0.1 + 1e-9, `tip at ${tip.x}`);  // within a step of the pen
    p.penDown(5, 0, 0.1);
    const added = p.points.slice(n);
    assert.ok(added.length >= 35, `${added.length} bridge points`);
    assert.ok(added.every((q) => q.bridge), 'all bridge');
    assert.ok(Math.abs(added[added.length - 1].x - 5) < 1e-9, 'ends at the new stroke');
  },
  closed() {
    const p = new Path({ step: 0.1 }), N = 300;
    draw(p, Array.from({ length: N + 1 }, (_, k) => [2 * Math.cos(2 * Math.PI * k / N), 2 * Math.sin(2 * Math.PI * k / N)]));
    p.penUp();
    assert.ok(p.loop, 'closed');
    assert.ok(p.loop.b - p.loop.a >= 0.9 * (p.points.length - 1), `loop ${JSON.stringify(p.loop)} of ${p.points.length}`);
  },
  crossing() {
    const p = new Path({ step: 0.05 }), pts = [];
    for (let t = -0.5; t <= 2 * Math.PI - 0.5; t += 0.01) pts.push([2 * Math.sin(t), Math.sin(t) * Math.cos(t)]);
    draw(p, pts, 0.05);
    assert.ok(p.loop, 'a crossing makes a loop');
    const A = p.points[p.loop.a - p.base], B = p.points[p.loop.b - p.base];
    assert.ok(Math.hypot(A.x, A.y) < 0.3 && Math.hypot(B.x, B.y) < 0.3, 'both ends at the crossing');
    assert.ok(p.loop.b - p.loop.a > 0.3 * p.points.length, 'the loop is one lobe');
  },
  trim() {
    const p = new Path({ step: 0.1, maxPoints: 100 }); draw(p, sweep(0, 0, 30, 0, 600));
    assert.ok(p.base > 0);
    assert.equal(p.points.length, 100);
    const h = new Playhead(p); h.s = p.base - 30; h.advance(0);
    assert.equal(h.s, p.base);
  },
  trimExact() {
    const p = new Path({ step: 0.1, maxPoints: 100 }); p.penDown(0, 0, 0.1);
    let k = 0; while (p.base + p.points.length < 150) p.penMove(++k * 0.02, 0);
    assert.equal(p.points.length, 100); assert.equal(p.base, 50);
    const h = new Playhead(p); h.s = 20; h.advance(0); assert.equal(h.s, 50);
  },
  chase() {
    const p = new Path({ step: 0.1 }); draw(p, sweep(0, 0, 40, 0, 2000));
    const tip = p.base + p.points.length - 1; assert.ok(tip >= 200);
    const h = new Playhead(p); h.s = tip - 200; h.advance(1);
    assert.equal(h.s - (tip - 200), 1, 'no chase: far behind, it still moves at the slider speed');
    h.s = tip; h.advance(1); assert.equal(h.s, tip, 'waits at the tip while drawing');
  },
  boomerang() {
    const p = new Path({ step: 0.1 }); p.penDown(0, 0, 0.1);
    let x = 0; while (p.points.length < 10) p.penMove((x += 0.05), 0);
    p.penUp(); assert.equal(p.points.length, 10); assert.equal(p.loop, null);
    const h = new Playhead(p), seen = [];
    for (let k = 0; k < 40; k++) { h.advance(1); seen.push([h.s, h.dir]); }
    const top = seen.findIndex(([s]) => s === 9); assert.ok(top >= 0, 'reaches the tip');
    const back = seen.findIndex(([s], i) => i > top && s === 0); assert.ok(back > top, 'comes back to the start');
    assert.ok(seen.slice(top + 1, back).every(([, d]) => d === -1), 'going back');
    assert.equal(seen[back + 1][1], 1, 'and turns again');
  },
  loopwrap() {
    const p = new Path({ step: 0.1 }), N = 300;
    draw(p, Array.from({ length: N + 1 }, (_, k) => [2 * Math.cos(2 * Math.PI * k / N), 2 * Math.sin(2 * Math.PI * k / N)]));
    p.penUp(); const { a, b } = p.loop;
    const h = new Playhead(p); h.s = b; h.advance(1); h.advance(1); h.advance(1);
    assert.ok(h.s >= a && h.s < a + 4, `continued from the loop's start: s=${h.s}, a=${a}, b=${b}`);
  },
  farStroke() {  // C1: a stroke far away (a teleport) starts a new path at once, without a bridge of thousands of points
    const p = new Path({ step: 0.05 }); draw(p, sweep(0, 0, 3, 0, 60), 0.05); p.penUp();
    const h = new Playhead(p); h.s = p.tip;
    const t0 = performance.now(); p.penDown(1e5, 1e5, 0.05); const ms = performance.now() - t0;
    assert.ok(ms < 30, `penDown took ${ms.toFixed(0)} ms`);
    assert.equal(p.points.length, 1, 'a new path');
    h.advance(1); assert.ok(Math.abs(h.at().x - 1e5) < 1e-6, 'the playhead starts on the new path');
  },
  bigBridgeOneTrim() {  // a bridge longer than the cap allows still costs one trim, not one per point
    const p = new Path({ step: 0.05, maxPoints: 3000 }); draw(p, sweep(0, 0, 100, 0, 4000), 0.05); p.penUp();
    const t0 = performance.now(); p.penDown(100, 30, 0.05); const ms = performance.now() - t0;  // 600 steps away: a bridge
    assert.ok(ms < 30, `bridge took ${ms.toFixed(0)} ms`); assert.ok(p.points.length <= 3000);
  },
  jumpMove() {  // I4: a pen that jumps (the view zoomed under it) does not flood the path
    const p = new Path({ step: 0.01 }); draw(p, sweep(0, 0, 1, 0, 50), 0.01);
    const n0 = p.points.length, t0 = performance.now(); p.penMove(5000, 0); const ms = performance.now() - t0;
    assert.ok(ms < 30, `penMove took ${ms.toFixed(0)} ms`); assert.ok(p.points.length - n0 < 100, `${p.points.length - n0} points added`);
  },
  fastMove() {  // a fast drag (a big step between two mouse events) stays one solid stroke, not a dashed bridge
    const p = new Path({ step: 0.1 }); draw(p, sweep(0, 0, 1, 0, 10)); p.penMove(1 + 300 * 0.1 / 4, 0);  // 300 CSS px at 4 px a step
    for (let k = 0; k < 40; k++) p.penMove(9, 0);
    assert.ok(p.points.every((q) => !q.bridge), 'no bridge points'); assert.equal(p.start, 0, 'the same stroke');
  },
  newStrokeDropsLoop() {  // I7: a loop from an earlier stroke does not capture the playhead after a new stroke
    const p = new Path({ step: 0.1 }), N = 300;
    draw(p, Array.from({ length: N + 1 }, (_, k) => [2 * Math.cos(2 * Math.PI * k / N), 2 * Math.sin(2 * Math.PI * k / N)])); p.penUp();
    assert.ok(p.loop); p.penDown(2.5, 0, 0.1); assert.equal(p.loop, null);
  },
  position() {
    const p = new Path({ step: 0.1 }); draw(p, sweep(0, 0, 5, 0, 100)); p.penUp();
    const h = new Playhead(p); h.s = 2.5; const q = h.at();
    assert.ok(Math.abs(q.x - (p.points[2].x + p.points[3].x) / 2) < 1e-9, 'interpolates');
  },
};
let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
  try { fn(); console.log('ok  ', name); } catch (e) { failed++; console.log('FAIL', name, '-', e.message); }
}
if (failed) { console.log(`${failed} failed`); process.exit(1); }
console.log('all path tests passed');
