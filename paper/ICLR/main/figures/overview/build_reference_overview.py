"""Reference A/B layout, redrawn with native editable macro-flow shapes."""
from pathlib import Path
import importlib.util

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('method_scene', OUT.parent / 'method' / 'build_method.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
g.OUT = OUT
s = g.Scene()
BLUE, PURPLE, PALE = '#298fd6', '#440154', '#d6e8f1'


def line(x, y, xx, yy, dashed=False, color='#242424'):
    s.arrow(x, y, xx, yy, head=False, dashed=dashed, color=color)


def answer(x, y, alternative=False):
    s.box(x-12, y-12, 24, 24, '#e4c85c' if alternative else BLUE, 'none', ellipse=True)


def question(y):
    s.box(35, y, 685, 44, '#f0f5f8', '#b3c9d6')
    s.text(377, y+22, 'Question', size=24)


def cells(x, y, n, color):
    for k in range(n):
        s.box(x+28*k, y-11, 22, 22, color, 'white', lw=.8)


def path(y, inherited=0):
    s.arrow(211, y, 239, y)
    cells(245, y, inherited, PURPLE)
    cells(245+28*inherited, y, 5-inherited, PALE)
    s.text(414, y-3, '…', size=25, color='#666666')
    cells(447, y, 2, PALE)
    s.arrow(510, y, 538, y)


s.box(16, 15, 868, 412, '#edf5fa', 'none')
s.text(450, 48, 'Self-consistency (SC)', size=30, bold=True)
question(78)
for i, y in enumerate((168, 224, 280, 336)):
    s.box(39, y-20, 155, 40, '#f0f5f8', 'none')
    s.text(116, y, f'Trajectory {i+1}', size=22)
    path(y)
    answer(559, y, i == 2)
    line(577, y, 610, y)
line(610, 168, 610, 336)
s.arrow(610, 252, 642, 252)
s.box(650, 228, 145, 48, '#e4eef3', 'none')
s.text(722, 252, 'Final vote', size=24)
s.text(450, 393, 'Each trajectory starts from the question', size=24)

s.box(16, 451, 868, 861, '#f4eff8', 'none')
s.text(450, 484, 'Path-consistency (PC)', size=30, bold=True)
question(515)
s.arrow(278, 563, 278, 590, color=BLUE)


def window(index, top, inherited):
    ys = [top+18+31*k for k in range(4)]
    center = (ys[0]+ys[-1])/2
    s.box(39, top, 155, 130, '#edf4f8', 'none')
    s.text(116, center-13, f'Window {index}', size=23, bold=True)
    s.text(116, center+18, 'from question' if index == 1 else 'reuse prefix', size=18)
    for k, y in enumerate(ys):
        path(y, inherited)
        answer(559, y, k == 2)
        line(577, y, 595, y)
    line(595, ys[0], 595, ys[-1])
    s.arrow(595, center, 616, center)
    s.box(624, center-30, 146, 60, '#e4eef3', 'none')
    s.text(697, center-12, 'Answer', size=22)
    s.text(697, center+13, 'agreement', size=22)
    return center


def prefix(y, count):
    s.box(211, y, 559, 42, '#f0eaf3', 'none')
    s.text(337, y+21, 'Prefix selection', size=23, color=PURPLE)
    cells(468, y+21, count, PURPLE)
    s.text(687, y+21, 'Shared prefix', size=20, color=PURPLE)


c1 = window(1, 598, 0)
s.arrow(278, 730, 278, 746, color=BLUE)
prefix(752, 2)
s.arrow(278, 797, 278, 819, color=BLUE)
s.arrow(697, c1+32, 697, 749, color=BLUE)

c2 = window(2, 825, 2)
s.box(597, 825, 199, 30, '#ffffff', '#000000')
s.text(696, 840, 'Trial and recovery', size=19)
s.arrow(278, 958, 278, 975, color=BLUE)
prefix(980, 3)
s.arrow(278, 1025, 278, 1046, color=BLUE)
s.arrow(697, c2+33, 697, 975, color=BLUE)
s.text(736, 958, 'pass', size=18, color=BLUE)
c3 = window(3, 1052, 3)
# After window 2 passes, the first proposal is committed. If window 3
# fails its longer-prefix trial, recover that earlier committed prefix.
line(773, c3, 853, c3, dashed=True, color=PURPLE)
line(853, c3, 853, 774, dashed=True, color=PURPLE)
s.arrow(853, 774, 776, 774, dashed=True, color=PURPLE)
s.text(811, 956, 'recover', size=17, color=PURPLE)
s.text(300, 1217, 'All valid answers', size=23)
answer(451, 1217)
s.arrow(471, 1217, 523, 1217)
s.box(534, 1193, 194, 48, '#e4eef3', 'none')
s.text(631, 1217, 'Final vote', size=24)
s.text(450, 1280, 'Reuse and extend one continuous prefix across windows', size=22)

s.box(137, 1340, 22, 22, PALE, 'white', lw=.8)
s.text(245, 1351, 'Reasoning', size=22)
answer(387, 1351)
s.text(455, 1351, 'Answer', size=22)
s.box(562, 1340, 22, 22, PURPLE, 'white', lw=.8)
s.text(687, 1351, 'Shared prefix', size=22)

# Compact the vertical layout while keeping every reasoning cell and answer
# exactly the same size. Functional modules receive black borders, whereas
# the two background bands and the reasoning/answer symbols remain unframed.
Y_SCALE = .86
for item in s.items:
    if item['kind'] in ('rect', 'ellipse'):
        is_cell = item['w'] == 22 and item['h'] == 22
        is_background = item['w'] == 868
        is_symbol = is_cell or item['kind'] == 'ellipse'
        if not is_symbol and not is_background:
            item['stroke'] = '#000000'
            item['lw'] = 1.3
        if is_symbol:
            item['y'] = (item['y'] + item['h']/2)*Y_SCALE-item['h']/2
        else:
            item['y'] *= Y_SCALE
            item['h'] *= Y_SCALE
    else:
        item['y'] *= Y_SCALE
        if 'y2' in item:
            item['y2'] *= Y_SCALE

if __name__ == '__main__':
    g.drawio(s, 'sc_pc_reasoning', 900, 1187)
    g.render(s, 'sc_pc_reasoning', 900, 1187)
    print('Generated reference A/B-style overview in draw.io, SVG, PDF and PNG')
