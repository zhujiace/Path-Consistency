"""Convert the existing rectangle/text SVG into editable draw.io cells."""

from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'paper/ICLR/figures/sc_repetition_matrix'
SOURCE = DIRECTORY / 'q75_sc_pc_comparison.svg'
OUTPUT = DIRECTORY / 'q75_sc_pc_comparison.drawio'


def main():
    svg = ET.parse(SOURCE).getroot()
    width, height = map(float, svg.attrib['viewBox'].split()[2:])
    doc = ET.Element('mxfile', host='app.diagrams.net', type='device')
    diagram = ET.SubElement(doc, 'diagram', id='sc-pc-comparison', name='SC and PC')
    model = ET.SubElement(diagram, 'mxGraphModel', grid='0', page='1', pageScale='1',
                          pageWidth=str(width), pageHeight=str(height),
                          background='#ffffff', math='0', shadow='0')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0', value='Background')
    for layer, label in [('sc', 'SC'), ('pc', 'PC'), ('labels', 'Shared labels')]:
        ET.SubElement(root, 'mxCell', id=layer, parent='0', value=label)
    counts = {'rectangles': 0, 'texts': 0}
    metrics = TextToPath()

    def cell(identity, value, style, layer, x, y, w, h):
        node = ET.SubElement(root, 'mxCell', id=identity, value=value, style=style,
                             vertex='1', parent=layer)
        ET.SubElement(node, 'mxGeometry', x=f'{x:.6f}', y=f'{y:.6f}',
                      width=f'{w:.6f}', height=f'{h:.6f}', attrib={'as': 'geometry'})

    def visit(element, layer='1'):
        tag = element.tag.rsplit('}', 1)[-1]
        eid = element.get('id', '')
        if eid == 'axes_1':
            layer = 'sc'
        elif eid == 'axes_2':
            layer = 'pc'
        elif eid.startswith(('text_', 'legend_')) and layer == '1':
            layer = 'labels'
        style = dict(p.strip().split(':', 1) for p in element.get('style', '').split(';') if ':' in p)
        style = {k: v.strip() for k, v in style.items()}
        if tag == 'path':
            data = element.attrib['d']
            assert set(re.findall('[A-Za-z]', data)) <= {'M', 'L', 'z'}, data
            numbers = list(map(float, re.findall(r'-?\d+(?:\.\d+)?', data)))
            assert len(numbers) == 8
            xs, ys = numbers[::2], numbers[1::2]
            assert len(set(xs)) == len(set(ys)) == 2
            counts['rectangles'] += 1
            cell(f'rect-{counts["rectangles"]}', '',
                 f'rounded=0;html=1;fillColor={style.get("fill", "#000000")};'
                 f'strokeColor={style.get("stroke", "none")};strokeWidth={style.get("stroke-width", "1")};',
                 layer, min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
        elif tag == 'text':
            text = ''.join(element.itertext())
            font = style['font']
            size = float(re.search(r'([\d.]+)px', font)[1])
            bold = '700 ' in font or 'bold' in font
            prop = FontProperties(family='DejaVu Sans', size=size, weight='bold' if bold else 'normal')
            advance, _, _ = metrics.get_text_width_height_descent(text, prop, ismath=False)
            x, y = float(element.attrib['x']), float(element.attrib['y'])
            anchor = style.get('text-anchor', 'start')
            left = x - {'start': 0, 'middle': advance/2, 'end': advance}[anchor]
            rotation = float(re.search(r'rotate\(([-\d.]+)', element.get('transform', 'rotate(0'))[1])
            # Position the center of an editable text box around the SVG baseline.
            cx, cy = left + advance/2, y - .35*size
            angle = math.radians(rotation)
            dx, dy = cx-x, cy-y
            cx, cy = x+math.cos(angle)*dx-math.sin(angle)*dy, y+math.sin(angle)*dx+math.cos(angle)*dy
            box_width, box_height = advance+4, size*1.3
            counts['texts'] += 1
            cell(f'text-{counts["texts"]}', text,
                 f'text;html=1;align=center;verticalAlign=middle;whiteSpace=nowrap;'
                 f'fillColor=none;strokeColor=none;spacing=0;fontFamily=DejaVu Sans;'
                 f'fontSize={size};fontStyle={int(bold)};fontColor={style.get("fill", "#000000")};rotation={rotation};',
                 layer, cx-box_width/2, cy-box_height/2, box_width, box_height)
        elif tag in ('image', 'use'):
            raise ValueError(f'Non-native SVG element not supported: {tag}')
        if tag not in ('defs', 'metadata'):
            for child in element:
                visit(child, layer)

    visit(svg)
    ET.indent(doc, space='  ')
    ET.ElementTree(doc).write(OUTPUT, encoding='utf-8', xml_declaration=True)
    parsed = ET.parse(OUTPUT).getroot()
    cells = parsed.findall('.//mxCell')
    assert len({c.attrib['id'] for c in cells}) == len(cells)
    for layer in ('sc', 'pc'):
        squares = [c for c in cells if c.get('parent') == layer and c.get('id', '').startswith('rect-')
                   and abs(float(c.find('mxGeometry').get('width')) - 11.34) < 1e-5]
        assert len(squares) == 1024, (layer, len(squares))
    assert counts['texts'] == len(svg.findall('.//{http://www.w3.org/2000/svg}text'))
    print(OUTPUT)
    print(counts, 'validated 1024 editable matrix cells per panel')


if __name__ == '__main__':
    main()
