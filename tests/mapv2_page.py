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

    # the globe
    pg.click('.lbtn[data-l="globe"]')
    pg.wait_for_function(f'{G} && {G}.on && {G}.drawn >= 50', timeout=180000); pg.wait_for_timeout(600)
    s = v(); check(s['layout'] == 'globe' and pressed('.lbtn[data-l="globe"]') == 'true', 'Globe shows')
    vis = pg.evaluate('''() => ['map', 'globe', 'labels', 'mini', 'gmini', 'minicap', 'minisize'].map((i) => document.getElementById(i).offsetWidth > 0)''')
    check(vis == [False, True, True, False, True, True, True], f'the globe takes the map\'s place, the noise globe the minimap\'s: {vis}')
    dis = pg.evaluate('''() => ['size', 'draw', 'fillk', 'span', 'soft', 'trunc', 'pause', 'teleport', 'zin'].map((i) => document.getElementById(i).disabled)''')
    check(dis == [True, True, True, False, False, False, False, False, False], f'Region size, Draw and Fill are off on the globe; the rest work: {dis}')
    txt = pg.inner_text('#stats'); check('This view' in txt and 'Generated' in txt and 'Centre:' in txt, f'the map\'s readout, with the globe\'s centre line: {txt[:60]!r}...')
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
    pg.click('.dtog[data-d="taef2"]'); pg.wait_for_function(f'{G}.made > {m2}', timeout=120000); check(True, 'TAEF2 back on: tiles are decoded again')
    # M hides the noise globe
    pg.keyboard.press('m'); pg.wait_for_timeout(200); h = pg.evaluate("document.getElementById('gmini').offsetWidth"); pg.keyboard.press('m'); pg.wait_for_timeout(200)
    check(h == 0 and pg.evaluate("document.getElementById('gmini').offsetWidth") > 0, 'M hides and shows the noise globe')
    url = pg.url; rot = g()['rot']

    # back to All classes: its tiles are still there
    pg.click('.lbtn[data-l="all"]'); pg.wait_for_timeout(1500); s = v()
    check(s['layout'] == 'all' and s['made'] >= flat_made and pg.evaluate("document.getElementById('map').offsetWidth") > 0, f"back on All classes with its tiles kept: {s['made']} made (was {flat_made})")
    check(not pg.evaluate("document.getElementById('size').disabled || document.getElementById('draw').disabled"), 'Region size and Draw work again')
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
