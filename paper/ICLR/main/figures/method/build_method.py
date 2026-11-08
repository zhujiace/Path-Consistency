"""Native method diagrams, with a common scene for PDF/SVG/PNG and draw.io."""

from pathlib import Path
import copy
import html
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch

OUT = Path(__file__).resolve().parent
PURPLE = '#440154'
BLUE = '#298fd6'
TEAL = '#007d78'
YELLOW = '#ffe53b'
GRAY = '#e6e6e6'
INK = '#242424'
MUTED = '#666666'
W, H = 360, 470
TOKEN_SIZE = 22


class Scene:
    def __init__(self):
        self.items = []

    def text(self, x, y, text, size=23, color=INK, bold=False):
        self.items.append(dict(kind='text', x=x, y=y, text=text, size=size, color=color, bold=bold))

    def box(self, x, y, w, h, fill='white', stroke=INK, lw=1.5, dashed=False, ellipse=False):
        self.items.append(dict(kind='ellipse' if ellipse else 'rect', x=x, y=y, w=w, h=h,
                               fill=fill, stroke=stroke, lw=lw, dashed=dashed))

    def arrow(self, x, y, x2, y2, color=INK, dashed=False, head=True):
        self.items.append(dict(kind='arrow', x=x, y=y, x2=x2, y2=y2,
                               color=color, dashed=dashed, head=head))

    def tokens(self, x, y, colors):
        for i, color in enumerate(colors):
            self.box(x+i*TOKEN_SIZE, y, TOKEN_SIZE, TOKEN_SIZE, color, 'white', .9)


def agreement():
    s = Scene()
    s.text(180, 25, r'Admitted answers $\mathcal{V}$')
    s.box(30, 52, 175, 142, fill='none', stroke=MUTED, dashed=True, ellipse=True)
    s.box(226, 52, 98, 142, fill='none', stroke=MUTED, dashed=True, ellipse=True)
    for row in range(2):
        for col in range(3):
            s.box(59+col*42, 84+row*49, 29, 29, BLUE, BLUE, ellipse=True)
    s.box(261, 85, 29, 29, TEAL, TEAL, ellipse=True)
    s.box(261, 134, 29, 29, YELLOW, '#a89919', ellipse=True)
    s.text(117, 212, r'Top-1 $a_\star$', size=23)
    s.text(276, 212, 'All rest', size=23)
    s.text(117, 244, r'$v_\star$', size=24)
    s.text(276, 244, r'$n-v_\star$', size=24)
    s.arrow(180, 268, 180, 291)
    s.box(47, 302, 266, 53, '#f3f7fa', '#8a9aa6')
    s.text(180, 329, r'$c(n,v_\star)\geq C$', size=28)
    s.arrow(110, 355, 86, 393)
    s.arrow(250, 355, 278, 393, dashed=True)
    s.text(43, 371, 'pass', size=22)
    s.text(316, 371, 'fail', size=22)
    s.text(85, 415, 'Propose', size=24, color=BLUE)
    s.text(278, 415, 'Hold', size=24, color=MUTED)
    s.text(180, 455, '(a) Answer agreement', size=26, bold=True)
    return s


def selection():
    s = Scene()
    s.text(180, 25, 'Accepted window')
    s.text(54, 63, 'prefix', size=20, color=PURPLE)
    s.text(144, 63, 'extension', size=20, color=BLUE)
    s.text(225, 63, 'tail', size=20, color=MUTED)
    # The three blue answer chips support the modal answer. Fourth is excluded.
    local = [['#a7d8eb', '#d1eaf4', '#76b9d6', '#a7d8eb'],
             ['#3e99bd', '#2283af', '#3e99bd', '#2283af'],
             ['#76b9d6', '#e0f0f6', '#a7d8eb', '#76b9d6']]
    for row in range(4):
        y = 91+44*row
        s.tokens(27, y, [PURPLE]*3)
        s.tokens(93, y, local[row] if row < 3 else [GRAY]*4)
        s.tokens(181, y, [GRAY]*4)
        s.box(301, y-3, 31, 31, BLUE if row < 3 else YELLOW, 'none', ellipse=True)
        if row == 3:
            s.arrow(293, y-7, 338, y+31, color=MUTED, head=False)
    # Single selected source; the prefix itself remains continuous.
    s.box(21, 129, 254, 34, 'none', INK, lw=2.4)
    s.text(317, 271, r'$a_\star$', size=24, color=BLUE)
    s.text(151, 272, 'Score added tokens', size=23)
    s.text(180, 311, r'$s(z)=\min_{W\in\mathcal{W}(z)}\frac{1}{|W|}\sum_{t\in W}\ell_t$', size=24)
    s.arrow(180, 334, 180, 369)
    s.text(270, 350, r'max $s(z)$', size=23)
    s.tokens(103, 381, [PURPLE]*3 + [BLUE]*4)
    s.text(180, 425, r'One proposal $r\to r+0.2$', size=23)
    s.text(180, 455, '(b) Prefix selection', size=26, bold=True)
    return s


def recovery():
    s = Scene()
    s.text(180, 25, 'Trial in the next window')
    s.tokens(103, 56, [PURPLE]*3+[BLUE]*4)
    # One proposed prefix conditions every trajectory in the logical window.
    s.arrow(180, 84, 180, 106, head=False)
    s.arrow(42, 106, 318, 106, head=False)
    for x in (42, 134, 226, 318):
        s.arrow(x, 106, x, 129)
        s.tokens(x-22, 135, [TEAL, TEAL])
        s.box(x-9, 163, 18, 18, BLUE, BLUE, ellipse=True)
    s.text(180, 212, 'Same parent mode', size=24)
    s.text(180, 244, r'and $c(n,v_\star)\geq C$', size=24)
    s.arrow(117, 266, 77, 301)
    s.arrow(244, 266, 282, 301, dashed=True)
    s.text(45, 276, 'pass', size=22)
    s.text(320, 276, 'fail', size=22)
    s.text(78, 322, 'Commit', size=24, color=BLUE)
    s.text(281, 322, 'Roll back', size=24, color=MUTED)
    s.tokens(1, 347, [PURPLE]*3+[BLUE]*4)
    s.tokens(248, 347, [PURPLE]*3)
    s.text(79, 393, r'Admit to $\mathcal{V}$', size=23)
    s.text(267, 388, '2 failures', size=23)
    s.text(267, 414, 'Disable reuse', size=22, color=MUTED)
    s.text(180, 455, '(c) Trial and recovery', size=26, bold=True)
    return s


def combined(panels):
    result = Scene()
    for i, scene in enumerate(panels):
        for item in scene.items:
            shifted = copy.deepcopy(item)
            shifted['x'] += i*390
            if 'x2' in shifted:
                shifted['x2'] += i*390
            result.items.append(shifted)
    # Compact shared footer, separated from the mechanism sketches.
    result.box(147, 491, 22, 16, PURPLE, 'none')
    result.text(247, 499, 'Shared prefix', size=22)
    result.box(368, 491, 22, 16, BLUE, 'none')
    result.text(481, 499, 'New extension', size=22)
    result.text(830, 499, 'All valid answers enter the final vote', size=22)
    return result


def render(scene, stem, width, height):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                         'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(width/72, height/72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    text_artists = []
    for item in scene.items:
        k = item['kind']
        if k == 'text':
            text_artists.append(ax.text(item['x'], item['y'], item['text'],
                fontsize=item['size'], color=item['color'], ha='center', va='center',
                fontweight='bold' if item['bold'] else 'normal'))
        elif k in ('rect', 'ellipse'):
            kw = dict(facecolor=item['fill'], edgecolor=item['stroke'], linewidth=item['lw'],
                      linestyle=(0, (3, 3)) if item['dashed'] else '-')
            if k == 'rect':
                ax.add_patch(Rectangle((item['x'], item['y']), item['w'], item['h'], **kw))
            else:
                ax.add_patch(Ellipse((item['x']+item['w']/2, item['y']+item['h']/2), item['w'], item['h'], **kw))
        else:
            ax.add_patch(FancyArrowPatch((item['x'], item['y']), (item['x2'], item['y2']),
                arrowstyle='-|>' if item['head'] else '-', mutation_scale=13,
                color=item['color'], linewidth=1.6, shrinkA=0, shrinkB=0,
                linestyle=(0, (3, 3)) if item['dashed'] else '-'))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check text bounds against the complete canvas; visual inspection checks semantics.
    for artist in text_artists:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, artist.get_text()
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{extension}', dpi=180)
    plt.close(fig)


def drawio(scene, stem, width, height):
    doc = ET.Element('mxfile', host='app.diagrams.net', type='device')
    diagram = ET.SubElement(doc, 'diagram', id=stem, name=stem)
    model = ET.SubElement(diagram, 'mxGraphModel', grid='0', page='1', pageWidth=str(width),
                          pageHeight=str(height), background='#ffffff', math='1')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')
    for i, item in enumerate(scene.items, 2):
        k = item['kind']
        attrs = dict(id=str(i), parent='1')
        if k == 'arrow':
            attrs.update(edge='1', value='', style=f'endArrow={"block" if item["head"] else "none"};endFill=1;strokeWidth=1.6;strokeColor={item["color"]};dashed={int(item["dashed"])};html=1;')
            cell = ET.SubElement(root, 'mxCell', attrs)
            geom = ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as':'geometry'})
            for suffix, role in (('', 'sourcePoint'), ('2', 'targetPoint')):
                ET.SubElement(geom, 'mxPoint', x=str(item['x'+suffix]), y=str(item['y'+suffix]), attrib={'as':role})
            continue
        if k == 'text':
            # MathJax expressions stay editable as LaTeX in draw.io.
            import re
            value = re.sub(r'\$(.*?)\$', lambda m: r'\(' + m[1] + r'\)', item['text'])
            attrs.update(vertex='1', value=html.escape(value),
                style=f'text;html=1;align=center;verticalAlign=middle;whiteSpace=nowrap;spacing=0;fontFamily=DejaVu Sans;fontSize={item["size"]};fontColor={item["color"]};fontStyle={int(item["bold"])};fillColor=none;strokeColor=none;')
            x, y, w, h = item['x']-180, item['y']-20, 360, 40
        else:
            attrs.update(vertex='1', value='',
                style=f'{"ellipse;" if k == "ellipse" else "rounded=0;"}html=1;fillColor={item["fill"]};strokeColor={item["stroke"]};strokeWidth={item["lw"]};dashed={int(item["dashed"])};')
            x, y, w, h = item['x'], item['y'], item['w'], item['h']
        cell = ET.SubElement(root, 'mxCell', attrs)
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as':'geometry'})
    ET.indent(doc, space='  ')
    path = OUT / f'{stem}.drawio'
    ET.ElementTree(doc).write(path, encoding='utf-8', xml_declaration=True)
    ET.parse(path)


if __name__ == '__main__':
    scenes = [agreement(), selection(), recovery()]
    names = ['answer_agreement', 'prefix_selection', 'trial_recovery']
    for name, scene in zip(names, scenes):
        render(scene, name, W, H)
        drawio(scene, name, W, H)
    overview = combined(scenes)
    render(overview, 'fig1_method', 1140, 525)
    drawio(overview, 'fig1_method', 1140, 525)
    print('Generated three technical panels and Figure 1 in PDF, SVG, PNG and draw.io')
