"""Geometry-only cleanup and vector export of the author's native draw.io."""
from pathlib import Path
import copy
import html
import re
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch
from matplotlib.transforms import Affine2D

DIR = Path(__file__).resolve().parent
SOURCE = DIR / 'method.drawio'
BACKUP = DIR / 'method_before_alignment.drawio'
WIDTH, HEIGHT = 360, 460


def style(cell):
    return dict(v.split('=', 1) for v in cell.get('style', '').split(';') if '=' in v)


def panel_id(cell):
    n = int(cell.get('id').rsplit('-', 1)[1])
    return 0 if n <= 301 else 1 if n <= 368 else 2


def shift(cell, dx, dy):
    geom = cell.find('mxGeometry')
    if cell.get('vertex'):
        geom.set('x', str(float(geom.get('x', 0))+dx))
        geom.set('y', str(float(geom.get('y', 0))+dy))
    for p in geom.iter('mxPoint'):
        p.set('x', str(float(p.get('x', 0))+dx))
        p.set('y', str(float(p.get('y', 0))+dy))


def cleanup(tree):
    cells = [c for c in tree.iter('mxCell') if c.get('vertex') or c.get('edge')]
    by_id = {int(c.get('id').rsplit('-', 1)[1]): c for c in cells}
    original = {c.get('id'): dict(c.attrib) for c in cells}
    # All content/style attributes stay byte-for-byte equal; only geometry moves.
    updates = {287: {'y':531}, 288:{'y':580}, 290:{'y':641},
               295:{'x':1482}, 317:{'y':533.5}, 329:{'y':577.5},
               341:{'y':621.5}, 353:{'y':665.5}, 403:{'x':2340.5},
               404:{'x':2432.5}, 405:{'x':2524.5}, 406:{'x':2386.5},
               383:{'y':643}, 384:{'y':675}, 401:{'x':2323}}
    for number, attrs in updates.items():
        geom = by_id[number].find('mxGeometry')
        for key, value in attrs.items():
            geom.set(key, str(value))
    # Keep the new extension link, centering its landing on the author's label.
    geom = by_id[408].find('mxGeometry')
    geom.find("mxPoint[@as='sourcePoint']").set('x', '2300')
    geom.find("mxPoint[@as='targetPoint']").set('x', '2363')
    old = geom.find('Array')
    if old is not None:
        geom.remove(old)
    waypoints = ET.SubElement(geom, 'Array', {'as':'points'})
    for x, y in ((2300, 838), (2363, 838)):
        ET.SubElement(waypoints, 'mxPoint', x=str(x), y=str(y))
    assert all(c.attrib == original[c.get('id')] for c in cells)
    return cells


def export_pdf(cells, name, width, height):
    plt.rcParams.update({'font.family':'Liberation Sans', 'mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42, 'svg.fonttype':'path'})
    fig = plt.figure(figsize=(width/72, height/72), facecolor='white')
    ax = fig.add_axes([0,0,1,1], xlim=(0,width), ylim=(height,0))
    ax.axis('off')
    texts = []
    for c in cells:
        st, geo = style(c), c.find('mxGeometry')
        color = st.get('strokeColor', '#000000')
        lw = float(st.get('strokeWidth',1))
        dashed = (0,(3,3)) if st.get('dashed') == '1' else '-'
        if c.get('edge'):
            start = geo.find("mxPoint[@as='sourcePoint']")
            end = geo.find("mxPoint[@as='targetPoint']")
            points = [start]+list(geo.findall('./Array/mxPoint'))+[end]
            for j, (a,b) in enumerate(zip(points,points[1:])):
                last = j == len(points)-2
                head = last and st.get('endArrow','none') != 'none'
                ax.add_patch(FancyArrowPatch((float(a.get('x')),float(a.get('y'))),
                    (float(b.get('x')),float(b.get('y'))), arrowstyle='-|>' if head else '-',
                    mutation_scale=13, linewidth=lw, color=color, linestyle=dashed,
                    shrinkA=0, shrinkB=0, zorder=2))
            continue
        x,y,w,h = (float(geo.get(k,0)) for k in ('x','y','width','height'))
        if c.get('value'):
            value = html.unescape(c.get('value'))
            value = re.sub(r'<b>(.*?)</b>', lambda m: r'$\mathbf{'+m[1]+'}$', value)
            value = re.sub(r'\\\((.*?)\\\)', lambda m: '$'+m[1]+'$', value)
            texts.append(ax.text(x+w/2,y+h/2,value,ha='center',va='center',
                fontsize=float(st.get('fontSize',23)),color=st.get('fontColor','#000000'),
                fontweight='bold' if st.get('fontStyle')=='1' else 'normal',zorder=3))
        elif st.get('shape') == 'mxgraph.flowchart.annotation_2':
            # Official stencil path, rotated exactly as in the author's XML.
            transform = Affine2D().rotate_deg_around(x+w/2,y+h/2,float(st.get('rotation',0)))+ax.transData
            for xx,yy in (([x+w,x+w/2,x+w/2,x+w],[y,y,y+h,y+h]),
                          ([x,x+w/2],[y+h/2,y+h/2])):
                ax.plot(xx,yy,color=color,linewidth=lw,transform=transform,zorder=2)
        else:
            kw = dict(facecolor=st.get('fillColor','none'),edgecolor=color,linewidth=lw,linestyle=dashed,zorder=1)
            patch = Ellipse((x+w/2,y+h/2),w,h,**kw) if c.get('style','').startswith('ellipse;') else Rectangle((x,y),w,h,**kw)
            ax.add_patch(patch)
    fig.canvas.draw()
    for txt in texts:
        b = txt.get_window_extent(fig.canvas.get_renderer())
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height, txt.get_text()
    for ext in ('pdf','svg','png'):
        fig.savefig(DIR/f'{name}.{ext}',dpi=180)
    plt.close(fig)


def xml_document(cells, width, height, name):
    root = ET.Element('mxfile',host='app.diagrams.net')
    diagram = ET.SubElement(root,'diagram',id=name,name=name)
    model = ET.SubElement(diagram,'mxGraphModel',page='1',pageWidth=str(width),pageHeight=str(height),math='1',background='#ffffff')
    graph = ET.SubElement(model,'root')
    ET.SubElement(graph,'mxCell',id='0')
    ET.SubElement(graph,'mxCell',id='1',parent='0')
    graph.extend(copy.deepcopy(cells))
    ET.indent(root,space='  ')
    return ET.tostring(root,encoding='unicode')


if __name__ == '__main__':
    if not BACKUP.exists():
        shutil.copy2(SOURCE,BACKUP)
    tree = ET.parse(BACKUP)
    cells = cleanup(tree)
    # Normalize the off-page author canvas, preserving horizontal panel order.
    for c in cells:
        shift(c,-1442,-447)
    SOURCE.write_text(xml_document(cells,1140,HEIGHT,'method'),encoding='utf-8')
    for i in range(3):
        panel = [copy.deepcopy(c) for c in cells if panel_id(c)==i]
        for c in panel:
            shift(c,-390*i,0)
        name = f'method_{i+1}'
        (DIR/f'{name}.drawio').write_text(xml_document(panel,WIDTH,HEIGHT,name),encoding='utf-8')
        export_pdf(panel,name,WIDTH,HEIGHT)
    export_pdf(cells,'method_aligned',1140,HEIGHT)
    for file in (SOURCE, *(DIR/f'method_{i}.drawio' for i in (1,2,3))):
        ET.parse(file)
    print('Preserved every cell/style/value; aligned geometry; exported three vector panels.')
