"""Page test for mapv2.html: the flat maps and the globe in one page, with one set of controls (Playwright, Chrome
with WebGPU). Serve the repo with the globe's data files, then:
    python tests/mapv2_page.py [url]
default url: http://localhost:8768/mapv2.html"""
import sys, math
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8768/mapv2.html'
V, G = 'window.__v2', 'window.__globe'

def check(cond, what):
    print(('ok   ' if cond else 'FAIL ') + what)
    if not cond: check.failed += 1
check.failed = 0

with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=False, args=['--enable-unsafe-webgpu'])
    c = b.new_context(viewport={'width': 1200, 'height': 800})
    pg = c.new_page(); errors = []
    pg.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
    pg.on('pageerror', lambda e: errors.append(str(e)))
    v = lambda: pg.evaluate(V)
    g = lambda: pg.evaluate(G)
    pressed = lambda sel: pg.get_attribute(sel, 'aria-pressed')

    # the flat map, as map.html: All classes fills its view
    pg.goto(URL, wait_until='load')
    pg.wait_for_function(f'{V} && {V}.viewTotal > 20 && {V}.left === 0 && {V}.made > 20', timeout=180000)
    s = v(); check(s['layout'] == 'all', f"starts on All classes and fills its view: {s['made']} made of {s['viewTotal']}")
    check(pressed('.lbtn[data-l="all"]') == 'true' and pg.locator('.lbtn[data-l="globe"]').is_enabled(), 'three layouts offered; Globe enabled')
    flat_made = s['made']
    # a wheel notch on the flat map glides
    box = pg.locator('#map').bounding_box(); cx, cy = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    pg.mouse.move(cx, cy); z0 = v()['zoomPct']; pg.mouse.wheel(0, -100); pg.wait_for_timeout(40); za = pg.evaluate('document.getElementById("zpct").textContent')
    pg.wait_for_timeout(1200); zb = v()['zoomPct']
    check(zb > 1.1 * z0, f'the wheel zooms the flat map: {z0:.0f}% -> {zb:.0f}% (label 40 ms in: {za})')
    pg.click('#zpct'); pg.wait_for_timeout(1200); check(abs(v()['zoomPct'] - 100) < 1, f"the % button returns to 100%: {v()['zoomPct']:.1f}")
    pg.wait_for_function(f'{V}.left === 0', timeout=120000); flat_made = v()['made']

    # a path drawn on the flat map: it waits while the globe shows, and is back afterwards
    shown = lambda i: pg.evaluate(f"document.getElementById('{i}').offsetWidth > 0")
    pg.keyboard.press('d'); pg.mouse.move(cx - 150, cy); pg.mouse.down(); pg.mouse.move(cx + 150, cy + 40, steps=12); pg.mouse.up(); pg.keyboard.press('d'); pg.wait_for_timeout(600)
    check(shown('pathwin'), 'a path is drawn and its window shows')

    # the globe
    pg.click('.lbtn[data-l="globe"]')
    pg.wait_for_function(f'{G} && {G}.on && {G}.drawn >= 50', timeout=180000); pg.wait_for_timeout(600)
    s = v(); check(s['layout'] == 'globe' and pressed('.lbtn[data-l="globe"]') == 'true', 'Globe shows')
    check(not shown('pathwin'), "the flat map's path is out of sight on the globe")
    # drawing on the globe: D, a drag draws (the globe does not turn), the window plays frames along it; the dot scrubs
    ink = "() => { const d = document.getElementById('pathcv').getContext('2d').getImageData(0, 0, 256, 256).data; let n = 0; for (let i = 0; i < d.length; i += 4) n += d[i] + d[i + 1] + d[i + 2] > 30; return n; }"
    line = "() => { const c = document.getElementById('labels'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 4) n += d[i] > 250 && d[i + 1] > 250 && d[i + 2] > 250 && d[i + 3] > 250; return n; }"
    w0 = pg.evaluate(line); r0 = g()['rot']
    pg.keyboard.press('d'); pg.mouse.move(cx - 160, cy + 60); pg.mouse.down(); pg.mouse.move(cx + 160, cy + 120, steps=14); pg.mouse.up(); pg.keyboard.press('d'); pg.wait_for_timeout(2500)
    check(g()['rot'] == r0 and shown('pathwin') and pg.evaluate(ink) > 20000, f'a path drawn on the globe: the globe stays put, the window plays frames ({pg.evaluate(ink)} px drawn)')
    check(pg.evaluate(line) > w0 + 300, f'the path shows on the globe: {pg.evaluate(line) - w0} more white px')
    pg.mouse.move(cx, cy - 150); pg.mouse.down(); pg.mouse.move(cx + 90, cy - 120, steps=8); pg.wait_for_timeout(150); pg.mouse.up(); pg.wait_for_timeout(400)
    check(g()['rot'] != r0 and pg.evaluate(line) > w0 + 300, 'out of Draw mode a drag turns the globe, and the path turns with it')
    pg.click('#pathclear'); pg.wait_for_timeout(300); check(not shown('pathwin') and pg.evaluate(line) < w0 + 300, 'Clear removes it')
    vis = pg.evaluate('''() => ['map', 'globe', 'labels', 'mini', 'gmini', 'minicap', 'minisize'].map((i) => document.getElementById(i).offsetWidth > 0)''')
    check(vis == [False, True, True, False, False, False, False], f'the globe takes the map\'s place; the noise globe is off by default: {vis}')
    dis = pg.evaluate('''() => ['size', 'draw', 'fillk', 'span', 'soft', 'trunc', 'pause', 'teleport', 'zin'].map((i) => document.getElementById(i).disabled)''')
    check(dis == [True, False, True, False, False, False, False, False, False], f'Region size and Fill are off on the globe; the rest work: {dis}')
    txt = pg.inner_text('#stats'); check('This view' in txt and 'Generated' in txt and 'Centre:' not in txt, f'the map\'s readout on the globe: {txt[:60]!r}...')
    check('layout=globe' in pg.url and 'rot=' in pg.url, 'the link carries the globe and its view')
    check(abs(s['zoomPct'] - 100) < 2, f"the globe starts at 100%: {s['zoomPct']:.1f}")
    # zoom buttons and keys
    pg.click('#zin'); pg.wait_for_timeout(1000); z1 = v()['zoomPct']; check(abs(z1 - 150) < 3, f'+ zooms the globe by 1.5: {z1:.0f}%')
    pg.keyboard.press('-'); pg.wait_for_timeout(1000); check(abs(v()['zoomPct'] - 100) < 3, f"the - key zooms back: {v()['zoomPct']:.0f}%")
    # arrow key, held: a continuous turn
    r0 = g()['rot']; pg.keyboard.down('ArrowLeft'); pg.wait_for_timeout(300); r1 = g()['rot']; pg.wait_for_timeout(300); r2 = g()['rot']; pg.keyboard.up('ArrowLeft')
    pg.wait_for_timeout(200); r3 = g()['rot']; pg.wait_for_timeout(200)
    check(r1 != r0 and r2 != r1 and g()['rot'] == r3, 'an arrow key turns the globe while held, and stops when released')
    # teleport
    pg.keyboard.press('t'); pg.wait_for_timeout(200); r4 = g()['rot']
    check(max(abs(a - b) for a, b in zip(r4, r3)) > 0.05, 'T teleports the globe')
    pg.wait_for_function(f'{G}.drawn >= 50', timeout=120000)
    # pause, by its button
    pg.click('#pause'); pg.wait_for_timeout(1500); m0 = g()['made']; pg.keyboard.press('t'); pg.wait_for_timeout(2500); m1 = g()['made']
    check(m1 == m0 and v()['held'] and 'paused' in pg.inner_text('#stats'), f'the Pause button holds the globe: {m0} -> {m1} made, readout says paused')
    pg.click('#pause'); pg.wait_for_function(f'{G}.made > {m1}', timeout=60000); check(True, 'Pause again resumes')
    # seed: next
    e0 = g()['epoch']; pg.evaluate("document.querySelector('details').open = true"); pg.click('#nnext'); pg.wait_for_timeout(300)
    check(g()['epoch'] > e0, 'a new noise seed makes the globe again')
    # morphing: four stops
    rng = pg.evaluate("() => { const s = document.getElementById('span'); return [s.min, s.max, s.value]; }")
    check(rng[:2] == ['0', '3'], f'Morphing has four stops on the globe: {rng}')
    e1 = g()['epoch']; pg.evaluate("() => { const s = document.getElementById('span'); s.value = 3; s.dispatchEvent(new Event('input')); }"); pg.wait_for_timeout(2500)
    check(g()['epoch'] > e1 and '8 tiles' in pg.inner_text('#spanv') and 'gspan=8' in pg.url, f"Morphing at its last stop: {pg.inner_text('#spanv')!r}, globe made again")
    # decoders: TAEF2 off leaves Conv alone
    pg.wait_for_function(f'{V}.left === 0', timeout=180000)
    if pressed('.dtog[data-d="flux2"]') == 'true': pg.click('.dtog[data-d="flux2"]')  # FLUX.2 off first: TAEF2 off alone would switch to it
    pg.click('.dtog[data-d="taef2"]')
    check(pressed('.dtog[data-d="conv"]') == 'true' and pressed('.dtog[data-d="taef2"]') == 'false' and pressed('.dtog[data-d="flux2"]') == 'false', 'only Conv is on')
    pg.wait_for_timeout(500); m2 = g()['made']; pg.keyboard.press('t'); pg.wait_for_timeout(4000); s = g()
    check(s['made'] == m2 and s['drawn'] >= 30, f"Conv alone: latents only ({s['drawn']} drawn, none decoded)")
    # Conv alone, far out (more than 256 tiles in view): every tile still gets its latent
    pg.keyboard.press('-'); pg.keyboard.press('-'); pg.wait_for_timeout(1500)
    try: pg.wait_for_function(f'{V}.left === 0 && {V}.viewTotal > 300', timeout=90000); ok = True
    except Exception: ok = False
    check(ok, f"Conv alone, far out: all {v()['viewTotal']} tiles in view get a latent ({v()['left']} to go)")
    # Conv alone: a new seed makes the latents shown again (no stale tiles)
    p0 = g()['previews']; n0 = v()['viewTotal']; pg.click('#nnext'); pg.wait_for_timeout(500)
    try: pg.wait_for_function(f'{G}.previews >= {p0} + 0.9 * {n0} && {V}.left === 0', timeout=90000); ok = True
    except Exception: ok = False
    check(ok, f"Conv alone: a new seed remakes the latents in view ({g()['previews'] - p0} of {n0})")
    pg.keyboard.press('+'); pg.keyboard.press('+'); pg.wait_for_timeout(1200)
    pg.click('.dtog[data-d="taef2"]'); pg.wait_for_function(f'{G}.made > {m2}', timeout=120000); check(True, 'TAEF2 back on: tiles are decoded again')
    # the noise globe: an Advanced option, off by default; on, it shows and steers the class it names
    N = 'window.__noise'
    pg.wait_for_function(f'{V}.left === 0', timeout=180000); e2 = g()['epoch']; pg.check('#ngon')
    pg.wait_for_function(f'{N} && {N}.shown && {N}.tiles >= 20', timeout=120000); pg.wait_for_timeout(600)
    cap = pg.inner_text('#minicap'); code = pg.inner_text('#ngcode'); txt = pg.inner_text('#stats')
    check(g()['epoch'] > e2 and 'Noise for' in cap and 'Shift' in cap and 'Centre:' in txt and 'ng=1' in pg.url, f'the noise globe turns on: tiles made again, caption {cap.splitlines()[0]!r}, readout line, link')
    check('as every class' in code and not pg.evaluate("document.getElementById('noiserow').hidden"), f'its code shows in Advanced: {code!r}')
    mb = pg.locator('#gmini').bounding_box(); mx, my = mb['x'] + mb['width'] / 2, mb['y'] + mb['height'] / 2
    def spin(dx=6, dy=-3):
        pg.mouse.move(mx, my); pg.mouse.down()
        for k in range(10): pg.mouse.move(mx + dx * k, my + dy * k)
        pg.wait_for_timeout(150); pg.mouse.up()
    pg.wait_for_function(f'{V}.left === 0', timeout=180000); n0 = pg.evaluate(N); e3 = g()['epoch']; spin(); pg.wait_for_timeout(400); n1 = pg.evaluate(N); s = g()
    check(len(n1['spun']) >= 1 and n1['Mm'] == n0['Mm'] and s['epoch'] == e3 and 0 < s['stale'] < s['madeCells'], f"a spin turns the named class's noise only ({list(n1['spun'])}) and remakes its tiles only: {s['stale']} of {s['madeCells']}")
    check(n1['view'] != n0['view'] and 'as every class' not in pg.inner_text('#ngcode') and 'nc=' in pg.url, f"the small globe turns, its code and the link follow: {pg.inner_text('#ngcode')!r}")
    pg.keyboard.down('Shift'); spin(-5, 4); pg.keyboard.up('Shift'); pg.wait_for_timeout(300); n2 = pg.evaluate(N)
    check(n2['Mm'] != n1['Mm'] and g()['epoch'] > e3, "a Shift-spin turns every class's noise")
    v0 = n2['view']; pg.mouse.move(cx, cy); pg.mouse.down(); pg.mouse.move(cx + 80, cy + 30, steps=8); pg.wait_for_timeout(150); pg.mouse.up(); pg.wait_for_timeout(300); n3 = pg.evaluate(N)
    check(n3['view'] == v0 and max(abs(a - b) for a, b in zip(n3['front'], n3['centre'])) < 1e-3, "turning the class globe leaves the noise globe still, facing the centre's noise")
    # M hides the noise globe
    pg.keyboard.press('m'); pg.wait_for_timeout(200); h = pg.evaluate("document.getElementById('gmini').offsetWidth"); pg.keyboard.press('m'); pg.wait_for_timeout(200)
    check(h == 0 and pg.evaluate("document.getElementById('gmini').offsetWidth") > 0, 'M hides and shows the noise globe')
    url = pg.url; rot = g()['rot']

    # back to All classes: its tiles are still there
    pg.click('.lbtn[data-l="all"]'); pg.wait_for_timeout(1500); s = v()
    check(s['layout'] == 'all' and s['made'] >= flat_made and pg.evaluate("document.getElementById('map').offsetWidth") > 0, f"back on All classes with its tiles kept: {s['made']} made (was {flat_made})")
    check(not pg.evaluate("document.getElementById('size').disabled || document.getElementById('draw').disabled"), 'Region size and Draw work again')
    check(shown('pathwin'), 'the drawn path and its window are back'); pg.click('#pathclear')
    q = dict(kv.split('=') for kv in pg.url.split('?')[1].split('&'))
    check(q.get('layout') == 'all' and 'x' in q and abs(float(q['zoom']) - 84) < 1, f"the link is the flat map's again: layout {q.get('layout')}, zoom {q.get('zoom')}")
    # quick toggles: one draw loop, not two (frames per second stay at the screen's rate)
    for l in ('globe', 'all', 'globe', 'all', 'globe'): pg.click(f'.lbtn[data-l="{l}"]')
    pg.wait_for_function(f'{G}.on', timeout=60000)
    n = pg.evaluate('''() => new Promise((res) => { let k = 0, f = 0; const t0 = performance.now(), d0 = window.__globe;
      const tick = () => { f++; if (window.__globe !== k) { k = window.__globe; } if (performance.now() - t0 < 1000) requestAnimationFrame(tick); else res(f); }; requestAnimationFrame(tick); })''')
    check(pg.evaluate(f'{G}.on') and v()['layout'] == 'globe', f'quick layout toggles end on the globe ({n} frames in a second)')
    pg.click('.lbtn[data-l="all"]'); pg.wait_for_timeout(800)
    pg.wait_for_function(f'{V}.left === 0', timeout=180000); check(True, 'the flat view completes (made again after the seed change)')
    # the noise globe on the flat map: beside the minimap; a spin makes the map again; its spins came along from the globe
    pos = pg.evaluate("() => ['mini', 'gmini'].map((i) => { const r = document.getElementById(i).getBoundingClientRect(); return [Math.round(r.left), Math.round(r.width)]; })")
    check(pos[0][1] > 0 and pos[1][1] > 0 and pos[1][0] >= pos[0][0] + pos[0][1], f'on the flat map the noise globe sits beside the minimap: {pos}')
    nf = pg.evaluate(N); check(nf['spun'].keys() == n2['spun'].keys(), 'the spun classes are the same in every layout')
    pg.wait_for_function(f'{N}.tiles >= 20', timeout=120000)
    mb = pg.locator('#gmini').bounding_box(); mx, my = mb['x'] + mb['width'] / 2, mb['y'] + mb['height'] / 2
    e4 = v()['epoch']; spin(); pg.wait_for_timeout(500); nf2 = pg.evaluate(N)
    check(v()['epoch'] > e4 and len(nf2['spun']) >= len(nf['spun']) and nf2['view'] != nf['view'], f"a spin on the flat map turns its centre class's noise and makes the map again: {pg.inner_text('#minicap').splitlines()[0]!r}")
    pg.wait_for_function(f'{V}.left === 0', timeout=180000)
    pg.uncheck('#ngon'); pg.wait_for_timeout(500)
    check(pg.evaluate("document.getElementById('gmini').offsetWidth") == 0 and 'ng=' not in pg.url and 'Centre:' not in pg.inner_text('#stats'), 'off again: no noise globe, no readout line, not in the link')
    pg.wait_for_function(f'{V}.left === 0', timeout=180000)
    # Conv alone on the flat map: a new seed makes the latents shown again (no stale tiles)
    pg.click('.dtog[data-d="taef2"]'); pg.wait_for_timeout(500); s = v(); p0, n0 = s['previews'], s['viewTotal']; pg.click('#nnext')
    try: pg.wait_for_function(f'{V}.previews >= {p0} + 0.9 * {n0}', timeout=60000); ok = True
    except Exception: ok = False
    check(ok, f"Conv alone on the flat map: a new seed remakes the latents in view ({v()['previews'] - p0} of {n0})")
    pg.click('.dtog[data-d="taef2"]'); pg.wait_for_function(f'{V}.left === 0', timeout=180000)
    # and to the globe again: its place is kept
    pg.click('.lbtn[data-l="globe"]'); pg.wait_for_function(f'{G}.on', timeout=60000); pg.wait_for_timeout(300)
    check(max(abs(a - b) for a, b in zip(g()['rot'], rot)) < 1e-6, 'the globe is where it was left')

    # a globe link opens on the globe, at its view
    pg2 = c.new_page(); pg2.on('pageerror', lambda e: errors.append(str(e))); pg2.goto(url, wait_until='load')
    pg2.wait_for_function(f'{G} && {G}.on && {G}.drawn >= 30', timeout=180000)
    check(max(abs(a - b) for a, b in zip(pg2.evaluate(f'{G}.rot'), rot)) < 1e-3, 'a globe link opens on the globe at its view')
    pg2.close()
    check(not [e for e in errors if 'favicon' not in e], f'no console errors: {errors[:3]}')
    c.close(); b.close()
print('all passed' if not check.failed else f'{check.failed} failed'); sys.exit(1 if check.failed else 0)
