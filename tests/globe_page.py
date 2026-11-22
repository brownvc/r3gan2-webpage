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
    r0 = pg.evaluate(f'{G}.rot')
    pg.mouse.move(cx, cy); pg.mouse.down(); pg.mouse.move(cx + 80, cy + 30, steps=8); pg.mouse.up(); pg.wait_for_timeout(200)
    check(pg.evaluate(f'{G}.rot') != r0, 'a drag turns the globe')
    r1 = pg.evaluate(f'{G}.rot'); pg.keyboard.press('ArrowLeft'); pg.wait_for_timeout(100)
    check(pg.evaluate(f'{G}.rot') != r1, 'an arrow key turns the globe')
    mk, qc = pg.evaluate(f'{G}.marker'), pg.evaluate(f'{G}.qCentre')
    check(mk is not None and qc is not None and abs(mk[0] - qc[0]) < 1e-6 and abs(mk[1] - qc[1]) < 1e-6, f'the minimap marker sits at the centre tile’s noise point: {mk} vs {qc}')
    txt = pg.inner_text('#stats'); check('noise directions carry' in txt and '%' in txt, f'readout: {txt!r}')
    # the noise minimap: spinning it moves p, starts a new epoch, and the tiles in view are made again
    s0 = pg.evaluate(G); mb = pg.locator('#mini').bounding_box(); mx, my = mb['x'] + mb['width'] / 2, mb['y'] + mb['height'] / 2
    pg.mouse.move(mx, my); pg.mouse.down()
    for k in range(10): pg.mouse.move(mx + 6 * k, my - 3 * k)
    pg.mouse.up()
    s1 = pg.evaluate(G); check(s1['p'] != s0['p'] and s1['epoch'] > s0['epoch'], f"noise point moved {s0['p']} -> {s1['p']}")
    pg.wait_for_timeout(8000); s2 = pg.evaluate(G)
    check(s2['made'] > s1['made'] and s2['epoch'] == s1['epoch'], f"after the spin, {s2['made'] - s1['made']} tiles made for the last point only")
    pg.wait_for_timeout(800)
    pg.wait_for_function(f'{G}.mini >= 5', timeout=60000); check(pg.evaluate(f'{G}.mini') >= 5, f"minimap tiles made once the view is done: {pg.evaluate(f'{G}.mini')}")
    url = pg.url; check('p=' in url, 'the link carries p')
    if SHOT: pg.screenshot(path=SHOT)
    pg2 = c.new_page(); pg2.on('pageerror', lambda e: errors.append(str(e))); pg2.goto(url, wait_until='load')
    pg2.wait_for_function(f'{G} && {G}.drawn >= 50', timeout=180000)
    check(max(abs(a - b) for a, b in zip(pg2.evaluate(f'{G}.p'), s2['p'])) < 1e-3, 'a reload of the link restores p')
    m2 = pg.evaluate(f'{G}.Mm'); check(m2 is not None and max(abs(a - b) for a, b in zip(pg2.evaluate(f'{G}.Mm') or [9] * 9, m2)) < 1e-3, 'a reload restores the whole minimap rotation (every tile’s noise)')
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
    check(not [e for e in errors if 'favicon' not in e], f'no console errors: {errors[:3]}')
    c.close(); b.close()
print('all passed' if not check.failed else f'{check.failed} failed'); sys.exit(1 if check.failed else 0)
