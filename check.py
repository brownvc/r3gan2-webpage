"""Checks the page before a push: python check.py (exit 1 on a problem).

Section order, local links and images, leftover template text, and that every heavy data file is fetched
through DATA (config.js), so pointing DATA at the weights host moves all of them at once.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
ORDER = ['samples', 'speed', 'map', 'demo', 'abstract', 'citation', 'acknowledgements', 'references']
problems = []

index = (ROOT / 'index.html').read_text(encoding='utf-8') if (ROOT / 'index.html').exists() else ''
if not index:
    problems.append('index.html missing')

ids = re.findall(r'<(?:section|figure)[^>]*\bid="([a-z]+)-section"', index)
if ids != ORDER:
    problems.append(f'section order {ids}, want {ORDER}')

for page in ['index.html', 'map.html', 'generator.html']:
    p = ROOT / page
    if not p.exists():
        problems.append(f'{page} missing')
        continue
    live = re.sub(r'<!--.*?-->', '', p.read_text(encoding='utf-8'), flags=re.S)  # commented-out markup is not live
    for ref in re.findall(r'(?:src|href)="([^"#?]+)', live):
        if not re.match(r'(https?:|mailto:|data:|//)', ref) and not (ROOT / ref).exists():
            problems.append(f'{page}: {ref} not found')

# The comparison's samples: images/samples/<method>_<class>.jpg for every method and class in page.js.
js = (ROOT / 'page.js').read_text(encoding='utf-8') if (ROOT / 'page.js').exists() else ''
fig8 = re.search(r"const FIG8 = \[(.*?)\];", js)
classes = re.search(r"const CLASSES = \[(.*?)\];", js)
if not (fig8 and classes):
    problems.append('page.js: FIG8 or CLASSES not found')
else:  # the samples figure: images/<folder>/<class>_<nn>.jpg, 5 per class
    for folder in re.findall(r"\['[^']+', '([^']+)'\]", fig8.group(1)):
        for c in re.findall(r"\[(\d+),", classes.group(1)):
            for k in range(5):
                if not (ROOT / 'images' / folder / f'{c}_{k:02d}.jpg').exists():
                    problems.append(f'images/{folder}/{c}_{k:02d}.jpg missing')
for key, n in re.findall(r"'([a-z0-9-]+)':\s*\{[^}]*\bn: (\d+)", js):  # the speed panels: images/samples/<key>/207_<nn>.jpg
    for k in range(int(n)):
        if not (ROOT / 'images' / 'samples' / key / f'207_{k:02d}.jpg').exists():
            problems.append(f'images/samples/{key}/207_{k:02d}.jpg missing')
            break

for leftover in ['PROJECT TITLE', 'XXXX', 'AUTHOR1', 'ALL-CAPS']:
    if leftover in index:
        problems.append(f'index.html still has template text {leftover!r}')

# Heavy data: every fetch of a weight, embedding, test or ONNX file goes through DATA.
DATA_NAME = re.compile(r"""['"`][^'"`]*?(weights\.(bin|json)|embed\.bin|test(_full)?\.bin|\.onnx)['"`]""")
for f in ['generator.html', 'lab.js', 'map.html']:
    p = ROOT / f
    if not p.exists():
        continue
    for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if DATA_NAME.search(line) and 'DATA' not in line and 'data(' not in line and not line.lstrip().startswith('//'):
            problems.append(f'{f}:{n}: data file not fetched through DATA: {line.strip()[:100]}')

print('\n'.join(problems) or 'page ok')
sys.exit(1 if problems else 0)
