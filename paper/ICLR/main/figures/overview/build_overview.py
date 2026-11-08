"""SC/PC macro flow, using the method figure's shared vector renderer."""
from pathlib import Path
import importlib.util

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('method_scene', OUT.parent / 'method' / 'build_method.py')
graphics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graphics)
graphics.OUT = OUT
s = graphics.Scene()
BLUE = '#298fd6'
PURPLE = '#440154'
GRAY = '#e6e6e6'
INK = '#242424'


def node(x, y, w, h, label, fill, color=INK, size=22):
    s.box(x, y, w, h, fill, 'none')
    s.text(x+w/2, y+h/2, label, size=size, color=color)


def line(x, y, x2, y2):
    s.arrow(x, y, x2, y2, head=False)


def answer(x, y):
    s.box(x-13, y-13, 26, 26, BLUE, 'none', ellipse=True)


def chain(xs, y):
    for i, x in enumerate(xs):
        s.box(x, y-11, 22, 22, '#c2dce9', '#6c9cb5', lw=1)
        if i:
            s.arrow(xs[i-1]+26, y, x-4, y, color='#667780')


# Squares denote reasoning steps, circles denote answers (not correctness).
s.text(500, 40, 'SC   Independent reasoning paths', size=30, bold=True)
s.text(474, 103, 'Generate each path from the question', size=23)
node(16, 230, 122, 50, 'Question', GRAY)
line(138, 255, 177, 255)
line(177, 150, 177, 360)
for y in (150, 220, 290, 360):
    s.arrow(177, y, 222, y)
    chain((230, 318, 406, 494, 582, 670), y)
    s.arrow(700, y, 735, y)
    answer(755, y)
    line(773, y, 803, y)
line(803, 150, 803, 360)
s.arrow(803, 255, 842, 255)
node(850, 228, 134, 54, 'Final vote', '#d9e9ef')
s.text(755, 403, 'Answers', size=22, color=BLUE)

s.text(500, 525, 'PC   Reuse a prefix across reasoning paths', size=30, bold=True)
node(16, 707, 122, 50, 'Question', GRAY)
s.arrow(138, 732, 163, 732)
s.box(171, 622, 174, 220, '#f5f7f8', '#b6c2c7')
s.text(258, 591, 'Initial sampling', size=22)
for y in (651, 705, 759, 813):
    chain((187, 226, 265), y)
    s.arrow(291, y, 297, y)
    answer(315, y)

# These labels match the method figure, without exposing its internal tests.
s.arrow(345, 705, 375, 683)
s.box(383, 650, 174, 65, '#e9f1f5', '#a2b7c3')
s.text(470, 671, 'Answer', size=23)
s.text(470, 695, 'agreement', size=23)
s.arrow(470, 719, 470, 747)
s.text(470, 767, 'Prefix selection', size=22, color=PURPLE)
node(383, 793, 174, 44, 'Shared prefix', PURPLE, 'white', size=22)
line(557, 815, 583, 815)
line(583, 651, 583, 813)
s.box(608, 622, 214, 220, 'none', '#a2b7c3', dashed=True)
s.text(715, 576, 'Trial and recovery', size=23)
s.text(715, 602, 'Subsequent sampling', size=19, color='#666666')
for y in (651, 705, 759, 813):
    s.arrow(583, y, 616, y)
    chain((624, 670, 716), y)
    s.arrow(744, y, 756, y)
    answer(775, y)
    line(793, y, 835, y)
line(835, 651, 835, 813)
s.arrow(835, 732, 848, 732)
node(856, 705, 128, 54, 'Final vote', '#d9e9ef')
# Every initial answer remains in the final vote as well.
line(315, 842, 315, 887)
line(315, 887, 920, 887)
s.arrow(920, 887, 920, 765)
s.text(606, 913, 'All valid answers enter the final vote', size=22)

# Role legend, not a repeated-token color key.
s.box(208, 977, 22, 22, '#c2dce9', '#6c9cb5', lw=1)
s.text(325, 988, 'Reasoning step', size=22)
answer(474, 988)
s.text(536, 988, 'Answer', size=22)
s.box(636, 977, 22, 22, PURPLE, 'none')
s.text(755, 988, 'Shared prefix', size=22)

if __name__ == '__main__':
    graphics.drawio(s, 'sc_pc_reasoning', 1000, 1020)
    graphics.render(s, 'sc_pc_reasoning', 1000, 1020)
    print('Generated SC/PC right-hand overview in draw.io, SVG, PDF and PNG')
