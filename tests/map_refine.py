"""Test for map.html's FLUX.2 refining (Playwright, Chrome with WebGPU; serve the repo on 8768): zoomed in, no FLUX.2 refine starts while a button is held still or the mouse hovers. Each check starts on
a fresh area (tiles left to refine) once refining is under way there."""
import sys
from playwright.sync_api import sync_playwright
INIT = '''window.__done = []; let last = -1; (function f(n) { const s = document.getElementById('stats'), m = s && /Decoded \\(FLUX\\.2\\)\\s*([\\d,]+)/.exec(s.innerText), c = m ? +m[1].replace(/,/g, '') : 0;
  window.__busy = !!s && /refining/.test(s.innerText); if (c !== last) { if (last >= 0 && c > last) window.__done.push(n); last = c; } requestAnimationFrame(f); })(0);'''
with sync_playwright() as p:
    b = p.chromium.launch(channel='chrome', headless=False, args=['--enable-unsafe-webgpu'])
    ctx = b.new_context(viewport={'width': 1600, 'height': 1000}); ctx.add_init_script(INIT); pg = ctx.new_page()
    URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8768/map.html'
    pg.goto(URL + '?zoom=280', wait_until='load')
    pg.wait_for_function('window.__v2 && __v2.viewTotal > 0 && __v2.left === 0', timeout=300000)
    def fresh(dx):  # arrow-key pan to an unrefined area, then wait for refining to be under way there
        pg.mouse.move(5, 995); pg.keyboard.down('ArrowRight' if dx > 0 else 'ArrowLeft'); pg.wait_for_timeout(1500); pg.keyboard.up('ArrowRight' if dx > 0 else 'ArrowLeft')
        pg.wait_for_function('__v2.left === 0', timeout=120000); n = pg.evaluate('window.__done.length')
        pg.wait_for_function(f'window.__done.length >= {n + 1}', timeout=120000, polling=20)
    ok = True
    fresh(1); n0 = pg.evaluate('window.__done.length'); pg.mouse.move(800, 500); pg.mouse.down(); pg.wait_for_timeout(5000)
    k = pg.evaluate('window.__done.length') - n0; pg.mouse.up(); ok &= k <= 1
    print(f'{"ok  " if k <= 1 else "FAIL"} button held still 5 s: {k} tiles refined (at most the one under way)')
    fresh(-1); n0 = pg.evaluate('window.__done.length')
    for i in range(200): pg.mouse.move(800 + 30 * (i % 2), 500); pg.wait_for_timeout(25)
    k = pg.evaluate('window.__done.length') - n0; ok &= k <= 1
    print(f'{"ok  " if k <= 1 else "FAIL"} hovering 5 s: {k} tiles refined (at most the one under way)')
    pg.mouse.move(5, 995); n0 = pg.evaluate('window.__done.length'); pg.wait_for_timeout(5000); k = pg.evaluate('window.__done.length') - n0
    ok &= k >= 2; print(f'{"ok  " if k >= 2 else "FAIL"} then left alone 5 s: {k} tiles refined')
    b.close()
    sys.exit(0 if ok else 1)
