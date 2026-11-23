"""Page test for mapv2.html's path tool, on a flat map and on the globe (Playwright, Chrome with WebGPU). Serve the
repo with the globe's data files, then:
    python tests/mapv2_path.py [url] [screenshot folder]
default url: http://localhost:8768/mapv2.html"""
import sys, math, io
from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8768/mapv2.html'
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None
V, G, PT = 'window.__v2', 'window.__globe', 'window.__path'

def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond: check.failed += 1
check.failed = 0

with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=False, args=['--enable-unsafe-webgpu'])
    c = b.new_context(viewport={'width': 1200, 'height': 800}, has_touch=True)
    pg = c.new_page(); errors = []
    pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    pg.on('pageerror', lambda e: errors.append(str(e)))
    cdp = c.new_cdp_session(pg)
    path = lambda: (pg.wait_for_timeout(350), pg.evaluate(PT))[1]   # the hook refreshes four times a second
    g = lambda: pg.evaluate(G)
    shown = lambda i: pg.evaluate(f"document.getElementById('{i}').offsetWidth > 0")
    cx, cy = 430, 420   # clear of the readout (top left), the minimap (bottom left) and the panel (right)

    def stroke(pts, pause=12, hold=0):
        pg.mouse.move(*pts[0]); pg.mouse.down()
        for q in pts[1:]: pg.mouse.move(*q); pg.wait_for_timeout(pause)
        if hold: pg.wait_for_timeout(hold)
        pg.mouse.up()
    line = lambda x0, y0, x1, y1, n=16: [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1)]
    ring = lambda rx, ry, n=40, over=3: [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n + over + 1)]
    def colours(clip):  # in a screenshot of the map area: white-ish line pixels, and pixels of the dot's / arrow's red
        im = Image.open(io.BytesIO(pg.screenshot(clip=clip))).convert('RGB'); px = list(im.get_flattened_data()) if hasattr(im, 'get_flattened_data') else list(im.getdata())
        return sum(1 for r, gg, bb in px if r > 250 and gg > 250 and bb > 250), sum(1 for r, gg, bb in px if r > 170 and gg < 30 and bb < 30)
    AREA = {'x': 240, 'y': 60, 'width': 600, 'height': 560}   # the map between the overlays
    # is the arrow (its place in the page's test hook, CSS px in the map area) clear of every overlay?
    CLEAR = """() => { const a = window.__path.arrow; if (!a) return 'no arrow'; const w = document.getElementById('wrap').getBoundingClientRect(), x = w.left + a[0], y = w.top + a[1];
      for (const id of ['stats', 'pathwin', 'mini', 'gmini', 'minicap', 'rail', 'railtab', 'credit']) { const e = document.getElementById(id); if (!e || !e.offsetWidth) continue;
        const r = e.getBoundingClientRect(); if (x > r.left - 4 && x < r.right + 4 && y > r.top - 4 && y < r.bottom + 4) return 'under ' + id; }
      return x >= w.left && x <= w.right && y >= w.top && y <= w.bottom ? 'clear' : 'off the map'; }"""
    def touch(kind, pts):
        cdp.send('Input.dispatchTouchEvent', {'type': kind, 'touchPoints': [{'x': x, 'y': y, 'id': i} for i, (x, y) in enumerate(pts)]})

    def run(name, canvas, zoom_of, turned):
        """The same checks in either layout. zoom_of(): the zoom; turned(a, b): did the view move between two states."""
        state = lambda: (pg.evaluate(PT)['view'] if canvas == 'map' else g()['rot'])
        w0, r0 = colours(AREA)
        # an open stroke: drawn, the view stays put, the window plays frames along it
        v0 = state(); pg.keyboard.press('d'); stroke(line(cx - 150, cy - 40, cx + 150, cy + 60)); s = path()
        check(s['n'] > 20 and not s['loop'] and not s['drawing'] and not turned(v0, state()) and shown('pathwin'), f"{name}: an open stroke is drawn ({s['n']} points), the view stays put, the window shows")
        w1, r1 = colours(AREA); check(w1 > w0 + 400 and r1 > r0 + 60, f'{name}: the line and its dot show ({w1 - w0} white px, {r1 - r0} red px)')
        ink = pg.evaluate("() => { const d = document.getElementById('pathcv').getContext('2d').getImageData(0, 0, 256, 256).data; let n = 0; for (let i = 0; i < d.length; i += 4) n += d[i] + d[i + 1] + d[i + 2] > 30; return n; }")
        check(ink > 20000, f'{name}: the window plays frames ({ink} px drawn)')
        # the playhead travels: its place along the path changes
        s1 = path()['s']; pg.wait_for_timeout(700); s2 = path()['s']; check(s1 != s2, f'{name}: the playhead moves along the path ({s1:.1f} -> {s2:.1f})')
        # right-click clears (Draw mode stays on)
        pg.mouse.click(cx, cy + 150, button='right'); s = path(); check(s['n'] == 0 and not shown('pathwin'), f'{name}: right-click clears the path')
        # a closed stroke becomes a loop
        stroke(ring(150, 100)); s = path(); check(s['loop'] and s['n'] > 40, f"{name}: a closed stroke is a loop ({s['n']} points)")
        if SHOTS: pg.screenshot(path=f'{SHOTS}/path_{name}_loop.png')
        # the dot scrubs: out of Draw mode, a drag on it moves the playhead and not the view
        pg.keyboard.press('d'); pg.click('#pathplay'); pg.wait_for_timeout(500)   # paused, so the dot holds still
        bb0 = pg.locator('#' + canvas).bounding_box(); d = path()['dot']; dot = [bb0['x'] + d[0], bb0['y'] + d[1]] if d else None
        check(dot is not None, f'{name}: the dot is found on screen: {dot}')
        if dot:
            v0 = state(); s0 = path()['s']; pg.mouse.move(dot[0], dot[1]); pg.mouse.down(); pg.wait_for_timeout(100)
            for q in ring(150, 100, n=40, over=0)[10:22]: pg.mouse.move(*q); pg.wait_for_timeout(20)
            pg.wait_for_timeout(300); mid = pg.evaluate(PT); pg.mouse.up(); s1 = path()['s']
            check(mid['scrub'] and abs(s1 - s0) > 3 and not turned(v0, state()), f'{name}: dragging the dot scrubs ({s0:.1f} -> {s1:.1f}) and the view stays put')
        pg.click('#pathplay'); pg.click('#pathclear'); pg.keyboard.press('d')
        # edge scrolling: the pen held near the right edge moves the view, and the stroke goes on
        bb = pg.locator('#' + canvas).bounding_box(); ex = bb['x'] + bb['width'] - 12
        pg.mouse.move(cx, cy); pg.mouse.down()
        for q in line(cx, cy, cx + 60, cy, 6): pg.mouse.move(*q); pg.wait_for_timeout(15)
        v0 = state(); n0 = path()['n']
        for q in line(cx + 60, cy, ex, cy, 10): pg.mouse.move(*q); pg.wait_for_timeout(15)
        n1 = path()['n']; v1 = state(); pg.wait_for_timeout(700); n2 = path()['n']; v2 = state(); pg.mouse.up()
        check(turned(v1, v2) and n2 > n1 > n0, f'{name}: the pen held at an edge moves the view and the stroke goes on ({n0} -> {n1} -> {n2} points)')
        pg.mouse.click(cx, cy + 150, button='right')
        # arrow keys carry the stroke
        pg.mouse.move(cx, cy); pg.mouse.down()
        for q in line(cx, cy, cx + 60, cy + 10, 6): pg.mouse.move(*q); pg.wait_for_timeout(15)
        n0 = path()['n']; v0 = state(); pg.keyboard.down('ArrowRight'); pg.wait_for_timeout(600); pg.keyboard.up('ArrowRight'); n1 = path()['n']; v1 = state(); pg.mouse.up()
        check(turned(v0, v1) and n1 > n0, f'{name}: an arrow key held while drawing moves the view and carries the stroke ({n0} -> {n1} points)')
        pg.mouse.click(cx, cy + 150, button='right')
        # a wheel zoom mid-stroke: the stroke goes on at the new zoom, no flood of points
        pg.mouse.move(cx, cy); pg.mouse.down()
        for q in line(cx, cy, cx + 80, cy, 8): pg.mouse.move(*q); pg.wait_for_timeout(15)
        z0 = zoom_of(); n0 = path()['n']; pg.mouse.wheel(0, -200); pg.wait_for_timeout(1200)
        for q in line(cx + 80, cy, cx + 160, cy + 40, 8): pg.mouse.move(*q); pg.wait_for_timeout(15)
        n1 = path()['n']; pg.mouse.up(); z1 = zoom_of()
        check(z1 > 1.2 * z0 and n0 < n1 < n0 + 400, f'{name}: a zoom mid-stroke ({z0:.0f}% -> {z1:.0f}%) and the stroke goes on ({n0} -> {n1} points)')
        pg.keyboard.press('-'); pg.keyboard.press('-'); pg.wait_for_timeout(1200); pg.mouse.click(cx, cy + 150, button='right')
        # a second finger: the stroke ends and the two fingers pinch
        touch('touchStart', [(cx, cy)])
        for q in line(cx, cy, cx + 100, cy + 20, 8): touch('touchMove', [q]); pg.wait_for_timeout(20)
        s0 = path(); z0 = zoom_of(); touch('touchStart', [(cx + 100, cy + 20), (cx + 100, cy + 140)])
        for k in range(1, 9): touch('touchMove', [(cx + 100, cy + 20 - 10 * k), (cx + 100, cy + 140 + 10 * k)]); pg.wait_for_timeout(20)
        s1 = path(); z1 = zoom_of(); touch('touchEnd', []); pg.wait_for_timeout(300)
        check(s0['drawing'] and s0['n'] > 5 and not s1['drawing'] and not s1['pen'] and z1 > 1.3 * z0, f"{name}: a second finger ends the stroke ({s0['n']} points) and the two pinch ({z0:.0f}% -> {z1:.0f}%)")
        pg.click('#zpct'); pg.wait_for_timeout(1200); pg.mouse.click(cx, cy + 150, button='right'); pg.keyboard.press('d')

    # ---- a flat map ----
    pg.goto(URL, wait_until='load'); pg.wait_for_function(f'{V} && {V}.viewTotal > 20 && {V}.left === 0', timeout=180000)
    run('flat', 'map', lambda: pg.evaluate(V + '.zoomPct'), lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) > 0.05)
    # the dot off screen: an arrow at the edge points to it
    pg.keyboard.press('d'); stroke(line(cx - 60, cy, cx + 60, cy + 20)); pg.keyboard.press('d'); pg.click('#pathplay'); pg.wait_for_timeout(400)
    stroke(line(cx + 200, cy + 150, cx - 350, cy - 250, 20), hold=200); stroke(line(cx + 200, cy + 150, cx - 350, cy - 250, 20), hold=200); pg.wait_for_timeout(500)
    s = path(); where = pg.evaluate(CLEAR); check(s['dotShown'] == 'arrow' and s['dot'] is not None and not (0 <= s['dot'][0] <= 1200 and 0 <= s['dot'][1] <= 800) and where == 'clear', f"flat: with the path off screen its dot is drawn as an arrow, clear of the overlays (arrow at {s['arrow']}: {where})")
    if SHOTS: pg.screenshot(path=f'{SHOTS}/path_flat_arrow.png')
    flat_n = path()['n']

    # ---- the globe: its own path; the flat one waits ----
    pg.click('.lbtn[data-l="globe"]'); pg.wait_for_function(f'{G} && {G}.on && {G}.drawn >= 50', timeout=180000); pg.wait_for_timeout(800)
    check(path()['n'] == 0 and not shown('pathwin'), "globe: starts with no path (the flat map's waits)")
    run('globe', 'globe', lambda: pg.evaluate(V + '.zoomPct'), lambda a, b: max(abs(x - y) for x, y in zip(a, b)) > 1e-4)
    # the dot round the back of the globe: an arrow at the rim points to it; nothing of the line shows
    pg.goto(URL + '?layout=globe&zoom=6', wait_until='load'); pg.wait_for_function(f'{G} && {G}.on && {G}.drawn >= 50', timeout=180000); pg.wait_for_timeout(1500)
    w0, r0 = colours(AREA); pg.keyboard.press('d'); stroke(line(cx - 60, cy, cx + 60, cy + 20)); pg.keyboard.press('d'); pg.click('#pathplay'); pg.wait_for_timeout(500)
    w1, r1 = colours(AREA); check(w1 > w0 + 100 and path()['dotShown'] == 'dot', f'globe, far out: the path and its dot show ({w1 - w0} white px)')
    rad = pg.evaluate(f'{G}.zoom'); need = math.pi * rad   # px of drag for half a turn
    for _ in range(math.ceil(need / 500)): stroke(line(cx - 250, cy + 250, cx + 250, cy + 250, 20), hold=200)
    pg.wait_for_timeout(600); w2, r2 = colours(AREA)
    s = path(); where = pg.evaluate(CLEAR); check(w2 < w0 + 80 and s['dotShown'] == 'arrow' and s['dot'] is None and where == 'clear', f"globe: with the path round the back, the line is hidden and its dot is an arrow, clear of the overlays (arrow at {s['arrow']}: {where}; {w2 - w0} white px)")
    if SHOTS: pg.screenshot(path=f'{SHOTS}/path_globe_arrow.png')
    globe_n = path()['n']
    # each layout keeps its own path
    pg.click('.lbtn[data-l="all"]'); pg.wait_for_timeout(800); check(path()['n'] == 0, 'a reloaded page has no flat path')
    pg.keyboard.press('d'); stroke(line(cx - 100, cy, cx + 100, cy + 30)); pg.keyboard.press('d'); fn = path()['n']
    pg.click('.lbtn[data-l="globe"]'); pg.wait_for_timeout(800); gn = path()['n']
    pg.click('.lbtn[data-l="all"]'); pg.wait_for_timeout(800); fn2 = path()['n']
    check(gn == globe_n and fn2 == fn and fn > 10 and shown('pathwin'), f'each layout keeps its own path across switches: globe {globe_n} -> {gn}, flat {fn} -> {fn2}')
    check(not [e for e in errors if 'favicon' not in e], f'no console errors: {errors[:3]}')
    c.close(); b.close()
print('all passed' if not check.failed else f'{check.failed} failed'); sys.exit(1 if check.failed else 0)
