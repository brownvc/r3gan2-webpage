// The map's path tool, without the page: a drawn path (smoothed, evenly spaced points in world tile units, strokes
// joined by straight bridges, closed or self-crossing parts made loops, the oldest points dropped past a cap) and the
// playhead that travels it (chasing the pen, then boomeranging along an open path or circling a loop).

const lerp = (p, q, u) => ({ x: p.x + (q.x - p.x) * u, y: p.y + (q.y - p.y) * u });
const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);
// Do segments pq and rs cross (strictly, so segments that only touch at an end do not count)?
function crosses(p, q, r, s) {
  const d = (q.x - p.x) * (s.y - r.y) - (q.y - p.y) * (s.x - r.x);
  if (!d) return false;
  const u = ((r.x - p.x) * (s.y - r.y) - (r.y - p.y) * (s.x - r.x)) / d, v = ((r.x - p.x) * (q.y - p.y) - (r.y - p.y) * (q.x - p.x)) / d;
  return u > 0 && u < 1 && v > 0 && v < 1;
}

export class Path {
  // step: spacing of the points (world units; each stroke sets its own, from the zoom it is drawn at).
  // closeDist: a stroke ending this near its own start is closed (default 6 steps).
  constructor({ step = 0.05, maxPoints = 3000, closeDist } = {}) {
    this.step = step; this.maxPoints = maxPoints; this.closeDist = closeDist;
    this.clear();
  }
  clear() { this.points = []; this.base = 0; this.loop = null; this.drawing = false; this.f = null; this.start = 0; this.epoch = (this.epoch || 0) + 1; }
  get tip() { return this.base + this.points.length - 1; }  // absolute index of the newest point (-1: none)
  point(abs) { return this.points[abs - this.base]; }
  #add(x, y, bridge, trim = true) {
    this.points.push({ x, y, bridge });
    const n = this.points.length;
    if (n >= 5 && !bridge) {  // the new segment against all but the last few: a crossing closes a loop
      const p = this.points[n - 2], q = this.points[n - 1];
      for (let k = n - 5; k >= 0; k--) {
        if (crosses(p, q, this.points[k], this.points[k + 1])) { this.loop = { a: this.base + k + 1, b: this.base + n - 2 }; break; }
      }
    }
    if (trim) this.#trim();
  }
  #trim() {  // endless drawing: the oldest points go (one splice)
    const n = this.points.length;
    if (n > this.maxPoints) {
      const d = n - this.maxPoints; this.points.splice(0, d); this.base += d;
      if (this.loop && this.loop.a < this.base) this.loop = null;
      if (this.start < this.base) this.start = this.base;
    }
  }
  penDown(x, y, step = this.step) {
    this.loop = null;  // a new stroke: the playhead follows it, not an old loop
    if (this.points.length && dist(this.point(this.tip), { x, y }) > step * this.maxPoints / 4) this.clear();  // far away (a teleport): a new path
    this.step = step; this.drawing = true; this.f = { x, y };
    if (!this.points.length) this.#add(x, y, false);
    else {
      const t = this.point(this.tip), d = dist(t, this.f);
      if (d > 2 * step) {  // elsewhere: a straight bridge from the tip
        const n = Math.ceil(d / step);
        for (let k = 1; k <= n; k++) { const m = lerp(t, this.f, k / n); this.#add(m.x, m.y, true, false); }
        this.#trim();
      }
    }
    this.start = this.tip;
  }
  penMove(x, y) {
    if (!this.drawing) return;
    // A jump far beyond any hand movement (the view moved under the pen; zooms restart the stroke in the map): a new
    // stroke. A fast drag (hundreds of px between two mouse events) must stay one solid stroke.
    if (dist(this.point(this.tip), { x, y }) > (this.maxPoints / 4) * this.step) { this.penUp(); this.penDown(x, y, this.step); return; }
    const f = this.f; f.x += (x - f.x) * 0.35; f.y += (y - f.y) * 0.35;  // the hand's wobble smoothed out
    for (let t = this.point(this.tip), d = dist(t, f); d >= this.step; t = this.point(this.tip), d = dist(t, f)) {
      const m = lerp(t, f, this.step / d); this.#add(m.x, m.y, false);
    }
  }
  penUp() {
    if (!this.drawing) return;
    this.drawing = false;
    const s = this.point(this.start), t = this.point(this.tip);
    if (s && this.tip - this.start >= 10 && dist(s, t) <= (this.closeDist ?? 6 * this.step)) this.loop = { a: this.start, b: this.tip };
  }
}

export class Playhead {
  constructor(path) { this.path = path; this.epoch = path.epoch; this.s = path.base; this.dir = 1; this.arrived = false; }
  // The point at s, between the two path points around it.
  at() {
    const P = this.path, k = Math.max(0, Math.min(P.points.length - 1, this.s - P.base)), i = Math.floor(k);
    const p = P.points[i], q = P.points[Math.min(i + 1, P.points.length - 1)];
    return p ? lerp(p, q, k - i) : null;
  }
  // One frame's move, speed in points per frame, always the same speed (no chase). It follows the pen to the newest
  // point; once there and the pen is up, it circles the loop if there is one, else goes back and forth along the path.
  advance(speed) {
    const P = this.path, tip = P.tip;
    if (tip < P.base) return null;
    if (this.epoch !== P.epoch) { this.epoch = P.epoch; this.s = P.base; this.dir = 1; this.arrived = false; }  // a new path
    if (this.s < P.base) this.s = P.base;
    if (P.drawing) { this.arrived = false; this.dir = 1; }
    if (!this.arrived) {
      const lag = tip - this.s;
      if (lag > 0) this.s += Math.min(lag, speed);
      if (!P.drawing && this.s >= tip) { this.s = tip; this.arrived = true; this.dir = -1; }
    } else if (P.loop) {
      const { a, b } = P.loop;
      this.s = Math.max(this.s, a) + speed;
      if (this.s > b) this.s = a + (this.s - b) % Math.max(1e-9, b - a);
    } else {
      this.s += this.dir * speed;
      if (this.s >= tip) { this.s = Math.max(P.base, 2 * tip - this.s); this.dir = -1; }
      else if (this.s <= P.base) { this.s = Math.min(tip, 2 * P.base - this.s); this.dir = 1; }
    }
    return this.at();
  }
}
