"""Page test for globe.html (Playwright, Chrome with WebGPU). Serve the repo with the globe's data files, then:
    python tests/globe_page.py [url] [screenshot.png]
default url: http://localhost:8768/globe.html?data=./"""
import sys, math
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8768/globe.html?data=./'
SHOT = sys.argv[2] if len(sys.argv) > 2 else None
G = 'window.__globe'

def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond: check.failed += 1
check.failed = 0

with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=False, args=['--enable-unsafe-webgpu'])
    c = b.new_context(viewport={'width': 1000, 'height': 700})
    pg = c.new_page(); errors = []
    pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    pg.on('pageerror', lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until='load')
    pg.wait_for_function(f'{G} && {G}.inView > 0 && {G}.drawn >= {G}.inView', timeout=180000)
    s = pg.evaluate(G); check(s['drawn'] >= s['inView'] > 20, f"tiles drawn: {s['drawn']} of {s['inView']} in view")
    box = pg.locator('#globe').bounding_box(); cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    r0, v0 = pg.evaluate(f'{G}.rot'), pg.evaluate(f'{G}.miniView')
    pg.mouse.move(cx, cy); pg.mouse.down(); pg.mouse.move(cx + 80, cy + 30, steps=8); pg.wait_for_timeout(150); pg.mouse.up(); pg.wait_for_timeout(200)  # a pause first: no flick
    check(pg.evaluate(f'{G}.rot') != r0, 'a drag turns the globe')
    r1 = pg.evaluate(f'{G}.rot'); pg.keyboard.press('ArrowLeft'); pg.wait_for_timeout(300)
    check(pg.evaluate(f'{G}.rot') != r1, 'an arrow key turns the globe')
    f1, nc = pg.evaluate(f'[{G}.miniFront, {G}.noiseCentre]')
    v1 = pg.evaluate(f'{G}.miniView')
    check(v0 is not None and v0 == v1, f'turning the class globe leaves the noise globe still: {v0} -> {v1}')
    check(nc is not None and max(abs(a - b) for a, b in zip(f1, nc)) < 1e-3, f'the noise globe faces the noise at the view centre: {f1} vs {nc}')
    txt = pg.inner_text('#stats'); check('noise directions carry' in txt and '%' in txt, f'readout: {txt!r}')
    # the noise minimap: spinning it turns the noise of the class it names only; the tiles holding that class are made again
    s0 = pg.evaluate(G); mb = pg.locator('#mini').bounding_box(); mx, my = mb['x'] + mb['width'] / 2, mb['y'] + mb['height'] / 2
    def spin(dx=6, dy=-3):
        pg.mouse.move(mx, my); pg.mouse.down()
        for k in range(10): pg.mouse.move(mx + dx * k, my + dy * k)
        pg.wait_for_timeout(150)  # a pause before the release: a spin, not a flick
        pg.mouse.up()
    spin()
    s1 = pg.evaluate(G); spun = list(s1['spun'])
    check(len(spun) >= 1 and s1['Mm'] == s0['Mm'] and s1['epoch'] == s0['epoch'], f"a spin turns the noise of the named class only: {spun}; all-classes rotation and epoch unchanged")
    check(0 < s1['stale'] < s1['madeCells'], f"a spin remakes only the tiles holding that class: {s1['stale']} of {s1['madeCells']} made")
    check(s1['noiseCentre'] != s0['noiseCentre'], 'the view centre gets new noise')
    check(s1['miniView'] != s0['miniView'], 'spinning the noise globe turns it')
    check('Shift' in pg.inner_text('#minicap'), f"the caption says what Shift does: {pg.inner_text('#minicap')!r}")
    pg.wait_for_timeout(8000); s2 = pg.evaluate(G)
    check(s2['made'] > s1['made'] and s2['epoch'] == s1['epoch'], f"after the spin, {s2['made'] - s1['made']} tiles made for the last point only")
    # Shift: every class's noise turns, and every tile is made again
    pg.keyboard.down('Shift'); spin(-5, 4); pg.keyboard.up('Shift'); s3 = pg.evaluate(G)
    check(s3['Mm'] != s2['Mm'] and s3['epoch'] > s2['epoch'] and s3['spun'].keys() == s2['spun'].keys() and all(s3['spun'][k] != s2['spun'][k] for k in spun),
          'a Shift-spin turns every class’s noise, the spun classes’ too')
    pg.wait_for_timeout(8000); s2 = pg.evaluate(G)
    pg.wait_for_timeout(800)
    pg.wait_for_function(f'{G}.mini >= 5', timeout=60000); check(pg.evaluate(f'{G}.mini') >= 5, f"minimap tiles made once the view is done: {pg.evaluate(f'{G}.mini')}")
    # the minimap's tiles wrap its sphere as the main globe's do: once all are made, little of the bare sphere shows
    pg.wait_for_function(f'{G}.mini >= {G}.miniN', timeout=180000); pg.wait_for_timeout(200)
    bare = pg.evaluate('''() => { const m = document.getElementById('mini'), s = m.width, r = s / 2 - 2 * devicePixelRatio;
      const a = m.getContext('2d').getImageData(0, 0, s, s).data;
      const o = new OffscreenCanvas(s, s).getContext('2d'), gr = o.createRadialGradient(s / 2 - r / 3, s / 2 - r / 3, r / 10, s / 2, s / 2, r);
      gr.addColorStop(0, '#3a4a66'); gr.addColorStop(1, '#0e1420'); o.fillStyle = gr; o.fillRect(0, 0, s, s); const b = o.getImageData(0, 0, s, s).data;
      let n = 0, k = 0; for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) { if (Math.hypot(x - s / 2, y - s / 2) > 0.8 * r) continue; n++; const i = 4 * (y * s + x);
        if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) < 12) k++; } return k / n; }''')
    check(bare < 0.03, f'the minimap tiles cover its sphere: {bare:.1%} bare inside 0.8 r')
    url = pg.url; check('p=' in url, 'the link carries p')
    if SHOT: pg.screenshot(path=SHOT)
    pg2 = c.new_page(); pg2.on('pageerror', lambda e: errors.append(str(e))); pg2.goto(url, wait_until='load')
    pg2.wait_for_function(f'{G} && {G}.drawn >= 50', timeout=180000)
    check(max(abs(a - b) for a, b in zip(pg2.evaluate(f'{G}.p'), s2['p'])) < 1e-3, 'a reload of the link restores p')
    m2 = pg.evaluate(f'{G}.Mm'); check(m2 is not None and max(abs(a - b) for a, b in zip(pg2.evaluate(f'{G}.Mm') or [9] * 9, m2)) < 1e-3, 'a reload restores the whole minimap rotation (every tile’s noise)')
    sp1, sp2 = s2['spun'], pg2.evaluate(f'{G}.spun') or {}
    check(sp1.keys() == sp2.keys() and all(max(abs(a - b) for a, b in zip(sp1[k], sp2[k])) < 1e-3 for k in sp1), f'a reload restores each spun class’s noise: {list(sp2)}')
    pg2.close()
    # psi = 0: every tile is its class centre, z = 0
    pg.goto(URL + '&psi=0', wait_until='load'); pg.wait_for_function(f'{G} && {G}.drawn >= 50', timeout=180000)
    check('measured at' in pg.inner_text('#stats'), 'away from psi = 1 the readout says what the share is measured at')
    # model switch, when M is available
    if pg.locator('.mbtn[data-m="m"]').is_enabled():
        pg.click('.mbtn[data-m="m"]')
        pg.wait_for_function(f"{G}.model === 'm' && {G}.drawn >= 50", timeout=300000)
        check(pg.evaluate(f'{G}.model') == 'm', 'model switch to M: tiles made again')
    else: print('skip  model switch (M not available)')
    # far out, as on the map: each cell's latent preview first, nearest the centre first, then its decode; a still
    # view never makes a cell twice. It carries class labels, and P stops and restarts the making of tiles.
    pg.goto(URL + '&zoom=900', wait_until='load'); pg.wait_for_function(f'{G} && {G}.inView > 5000', timeout=180000); pg.wait_for_timeout(20000)
    s = pg.evaluate(G); check(s['previews'] > s['made'] > 0, f"far out, previews come first: {s['previews']} previews, {s['made']} decoded, {s['drawn']} of {s['inView']} shown")
    check(s['remade'] == 0, f"far out, a still view makes no cell twice: {s['remade']} remade")
    try: pg.wait_for_function(f'{G}.mini >= {G}.miniN', timeout=20000); ok = True
    except Exception: ok = False
    check(ok, f"the noise globe fills all the way while the class globe is busy: {pg.evaluate(f'{G}.mini')} of {s['miniN']} (20 s more allowed)")
    check(s['labels'] >= 10, f"far out, labels: {s['labels']}")
    pg.keyboard.press('p'); pg.wait_for_timeout(1500); m0 = pg.evaluate(f'{G}.made'); pg.wait_for_timeout(2500); m1 = pg.evaluate(f'{G}.made')
    check(m1 == m0, f'P pauses: {m0} -> {m1} made')
    pg.keyboard.press('p'); pg.wait_for_timeout(2500); check(pg.evaluate(f'{G}.made') > m1, 'P again resumes')
    pg.goto(URL, wait_until='load'); pg.wait_for_function(f'{G} && {G}.drawn >= 50', timeout=180000); pg.wait_for_timeout(500)
    cap = pg.inner_text('#minicap'); check('golden retriever' in cap, f'the noise globe names its class: {cap!r}')
    # close in, the cells reshape into a rectangular grid; at the start zoom they are hexagons
    check(pg.evaluate(f'{G}.morph') == 0, f"at the start zoom the cells are hexagons: morph {pg.evaluate(f'{G}.morph')}")
    # midway through the morph the tiles stay whole to the screen's edges: no empty space between or above them (a tall
    # window: the far rows drift most)
    tall = b.new_context(viewport={'width': 1100, 'height': 1300}).new_page()
    for zm in (12730, 14100):
        tall.goto(URL + f'&zoom={zm}', wait_until='load'); tall.wait_for_function(f'{G} && {G}.inView > 0 && {G}.drawn >= {G}.inView', timeout=180000)
        tall.wait_for_timeout(2500); bb = tall.locator('#globe').bounding_box()
        shot = tall.screenshot(clip={'x': bb['x'] + 320, 'y': bb['y'] + 8, 'width': 320, 'height': bb['height'] - 16})
        from PIL import Image; import io
        im = Image.open(io.BytesIO(shot)).convert('RGB'); px = list(im.getdata()); empty = sum(1 for p in px if p == (0, 0, 0)) / len(px)
        check(empty < 0.003, f'midway through the morph (zoom {zm}) no empty space: {empty:.2%} of the strip is bare')
    tall.context.close()
    pg.goto(URL + '&zoom=25000', wait_until='load'); pg.wait_for_function(f'{G} && {G}.drawn >= 4', timeout=180000); pg.wait_for_timeout(1500)
    s = pg.evaluate(G); check(s['morph'] == 1 and s['gridded'] >= 0.9 * s['drawn'], f"close in, the tiles are a grid: morph {s['morph']}, {s['gridded']} of {s['drawn']} placed")
    check(abs(s['gridAngle']) < 2, f"close in, the grid's rows are level: {s['gridAngle']:.1f} degrees")
    # spinning the class globe across classes: the noise globe holds still (no class switches, no fades) until it settles
    pg.goto(URL + '&zoom=2500', wait_until='load'); pg.wait_for_function(f'{G} && {G}.drawn >= 50', timeout=180000); pg.wait_for_timeout(1500)
    caps = [pg.inner_text('#minicap')]; bb = pg.locator('#globe').bounding_box(); x, y = bb['x'] + 100, bb['y'] + bb['height'] / 2
    pg.mouse.move(x, y); pg.mouse.down()
    for k in range(1, 31): pg.mouse.move(x + 25 * k, y); pg.wait_for_timeout(50); caps.append(pg.inner_text('#minicap')) if k % 3 == 0 else None
    pg.wait_for_timeout(150); pg.mouse.up(); pg.wait_for_timeout(1500); after = pg.inner_text('#minicap')
    check(len(set(caps)) == 1, f'while the class globe turns, the noise globe keeps its class: {sorted(set(caps))}')
    check(after != caps[0], f'once it settles, the noise globe takes the new class: {caps[0]!r} -> {after!r}')
    # a flick keeps either globe turning after the release
    def flick(sel, key):
        bb = pg.locator(sel).bounding_box(); x, y = bb['x'] + bb['width'] / 2, bb['y'] + bb['height'] / 2
        pg.mouse.move(x, y); pg.mouse.down()
        for k in range(1, 6): pg.mouse.move(x + 12 * k, y + 4 * k)
        pg.mouse.up(); pg.wait_for_timeout(60); a = pg.evaluate(f'{G}.{key}'); pg.wait_for_timeout(300); b = pg.evaluate(f'{G}.{key}')
        pg.wait_for_timeout(2500); return a != b
    check(flick('#globe', 'rot'), 'a flick keeps the class globe turning')
    check(flick('#mini', 'miniView'), 'a flick keeps the noise globe turning')
    # a wheel notch glides to its zoom (about the centre: the globe does not turn) instead of jumping
    bb = pg.locator('#globe').bounding_box(); pg.mouse.move(bb['x'] + 0.3 * bb['width'], bb['y'] + 0.3 * bb['height'])
    z0, r0 = pg.evaluate(f'[{G}.zoom, {G}.rot]'); pg.mouse.wheel(0, -100); pg.wait_for_timeout(40); za = pg.evaluate(f'{G}.zoom')
    pg.wait_for_timeout(1000); zb, rb = pg.evaluate(f'[{G}.zoom, {G}.rot]'); pg.wait_for_timeout(300); zc = pg.evaluate(f'{G}.zoom')
    fa = math.log(za / z0) / math.log(zb / z0)
    check(zb > 1.1 * z0 and fa < 0.8 and abs(zc / zb - 1) < 1e-3 and rb == r0, f'a wheel notch glides to its zoom about the centre: x{zb / z0:.2f}, {fa:.0%} of the way after 40 ms, settled after 1 s')
    # the noise globe's size button: larger, then back
    w0 = pg.evaluate("document.getElementById('mini').clientWidth"); pg.click('#minisize'); pg.wait_for_timeout(300)
    w1, cw = pg.evaluate("document.getElementById('mini').clientWidth"), pg.evaluate("document.getElementById('mini').width")
    pg.click('#minisize'); pg.wait_for_timeout(300); w2 = pg.evaluate("document.getElementById('mini').clientWidth")
    check(w1 > 1.5 * w0 and cw >= w1 and w2 == w0, f'the size button grows the noise globe and shrinks it back: {w0} -> {w1} (canvas {cw}) -> {w2}')
    # loading: a data folder given without its trailing slash, and no data parameter (the files next to the page)
    base = URL.split('globe.html')[0]
    for u in (base + 'globe.html?data=.', base + 'globe.html'):
        pg.goto(u, wait_until='load')
        try: pg.wait_for_function(f'{G} && {G}.drawn >= 50', timeout=120000); ok = True
        except Exception: ok = False
        check(ok, f'loads: {u} ({pg.inner_text("#status")!r})')
    check(not [e for e in errors if 'favicon' not in e], f'no console errors: {errors[:3]}')
    c.close(); b.close()
print('all passed' if not check.failed else f'{check.failed} failed'); sys.exit(1 if check.failed else 0)
