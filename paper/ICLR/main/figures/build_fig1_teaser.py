"""基于已核验汇总数据生成 Figure 1 初版，原生 draw.io + SVG/PDF/PNG。"""
from pathlib import Path
import xml.etree.ElementTree as E
import math
import ctypes as C

HERE = Path(__file__).resolve().parent
W, H = 1440, 700
INK = '#24324B'
GREEN, RED, PURPLE = '#B7DCAA', '#EEA8A6', '#BAA6D9'
mx = E.Element('mxfile', host='app.diagrams.net', type='device')
d = E.SubElement(mx, 'diagram', id='pc-teaser', name='Figure 1 capability teaser')
m = E.SubElement(d, 'mxGraphModel', page='1', pageScale='1', pageWidth=str(W), pageHeight=str(H), grid='1', gridSize='10', background='#FFFFFF', math='0')
root = E.SubElement(m, 'root')
E.SubElement(root, 'mxCell', id='0')
E.SubElement(root, 'mxCell', id='1', parent='0')
svg = E.Element('svg', xmlns='http://www.w3.org/2000/svg', width=str(W), height=str(H), viewBox=f'0 0 {W} {H}')
E.SubElement(svg, 'rect', width=str(W), height=str(H), fill='white')
defs = E.SubElement(svg, 'defs')
marker = E.SubElement(defs, 'marker', id='arrow', markerWidth='9', markerHeight='9', refX='8', refY='4.5', orient='auto', markerUnits='userSpaceOnUse')
E.SubElement(marker, 'path', d='M0 0 L9 4.5 L0 9 Z', fill=INK)
boxes = {}

def node(id, x, y, w, h, label='', fill='none', stroke='none', kind='rect', size=22, bold=False, radius=0, dash=False):
    boxes[id] = (x, y, w, h)
    shape = {'rect': 'rectangle', 'diamond': 'rhombus', 'text': 'text', 'ellipse': 'ellipse', 'star': 'star'}[kind]
    style = f'shape={shape};html=0;whiteSpace=wrap;fillColor={fill};strokeColor={stroke};strokeWidth=1.7;fontFamily=Arial;fontSize={size};fontColor={INK};fontStyle={1 if bold else 0};align=center;verticalAlign=middle;spacing=0;'
    if radius:
        style += f'rounded=1;absoluteArcSize=1;arcSize={radius*2};'
    if dash:
        style += 'dashed=1;dashPattern=5 4;'
    c = E.SubElement(root, 'mxCell', id=id, value=label, style=style, vertex='1', parent='1')
    E.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as':'geometry'})
    props = {'fill':fill, 'stroke':stroke, 'stroke-width':'1.7'}
    if dash: props['stroke-dasharray'] = '7 5'
    if kind == 'rect':
        E.SubElement(svg, 'rect', x=str(x), y=str(y), width=str(w), height=str(h), rx=str(radius), **props)
    elif kind == 'ellipse':
        E.SubElement(svg, 'ellipse', cx=str(x+w/2), cy=str(y+h/2), rx=str(w/2), ry=str(h/2), **props)
    elif kind in ('diamond', 'star'):
        if kind == 'diamond': pts = [(x+w/2,y),(x+w,y+h/2),(x+w/2,y+h),(x,y+h/2)]
        else:
            pts = [(x+w/2+math.sin(i*math.pi/5)*w/2*(1 if i%2==0 else .4), y+h/2-math.cos(i*math.pi/5)*h/2*(1 if i%2==0 else .4)) for i in range(10)]
        E.SubElement(svg, 'polygon', points=' '.join(f'{a},{b}' for a,b in pts), **props)
    if label:
        lines = label.split('\n')
        for i, text in enumerate(lines):
            t = E.SubElement(svg, 'text', x=str(x+w/2), y=str(y+h/2+(i-(len(lines)-1)/2)*size*1.2+size*.34), fill=INK, attrib={'font-family':'Arial, Liberation Sans, sans-serif','font-size':str(size),'font-weight':'bold' if bold else 'normal','text-anchor':'middle'})
            t.text = text

def edge(id, pts, source=None, target=None, color=INK, arrow=True, width=2, dash=False):
    style = f'edgeStyle=none;rounded=0;html=0;strokeColor={color};strokeWidth={width};endArrow={"block" if arrow else "none"};endFill=1;endSize=7;'
    attrs = {}
    for key, terminal, p in [('source',source,pts[0]), ('target',target,pts[-1])]:
        if terminal:
            attrs[key] = terminal
            x,y,w,h = boxes[terminal]
            port = 'exit' if key == 'source' else 'entry'
            style += f'{port}X={(p[0]-x)/w};{port}Y={(p[1]-y)/h};{port}Dx=0;{port}Dy=0;{port}Perimeter=0;'
    if dash: style += 'dashed=1;dashPattern=2 3;'
    c = E.SubElement(root,'mxCell',id=id,edge='1',parent='1',style=style,**attrs)
    g = E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    for name,p in [('sourcePoint',pts[0]),('targetPoint',pts[-1])]:
        E.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),attrib={'as':name})
    if len(pts)>2:
        a = E.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in pts[1:-1]: E.SubElement(a,'mxPoint',x=str(x),y=str(y))
    props = {'stroke':color,'stroke-width':str(width),'fill':'none','stroke-linejoin':'round'}
    if arrow: props['marker-end']='url(#arrow)'
    if dash: props['stroke-dasharray']='3 5'
    E.SubElement(svg,'path',d='M'+' L'.join(f'{x} {y}' for x,y in pts),**props)


import json
DATA=json.loads((HERE.parent/'data/results_snapshot.json').read_text())['sets']
settings=[('Llama','GSM8K'),('DeepSeek','MATH-500'),('DeepSeek','AIME 2024'),('DeepSeek','AIME 2025'),('DeepSeek','GPQA-D')]
rows=[next(r for r in DATA if (r['model'],r['dataset'])==s) for s in settings]
savings=[100*(1-r['pc']['tok']/r['sc']['tok']) for r in rows]
assert len(rows)==5
PURPLE='#7959C8'; TEAL='#219C96'; LIGHT='#E9E5F3'; GREY='#E8EBF0'
def text(id,x,y,w,h,label,size=22,bold=False,color=INK):
    node(id,x,y,w,h,label,kind='text',size=size,bold=bold)
    cell=root.find(f"mxCell[@id='{id}']")
    cell.set('style',cell.get('style').replace('fontColor='+INK,'fontColor='+color))
    for el in list(svg)[-len(label.split('\n')):]:
        if el.tag=='text':el.set('fill',color)

text('title',35,13,1370,48,'Path-consistency',35,True)
text('subtitle',35,63,1370,32,'Less generation. Flexible composition.',23,color='#637087')
node('left',25,119,851,511,fill='#FAF9FD',radius=16)
node('right',897,119,518,511,fill='#F5FAFA',radius=16)
text('left_head',50,138,800,33,'(a) Fewer tokens, similar accuracy',26,True)
text('hero',65,180,259,81,f'{sum(savings)/5:.1f}%',65,True,PURPLE)
text('hero_caption',65,263,259,24,'mean token saving',21)
text('range',380,193,415,47,f'{min(savings):.1f}% to {max(savings):.1f}%',36,True,PURPLE)
text('range_caption',380,243,415,31,'across five main-text comparisons',20)
text('cost_header',267,302,420,28,'Generated tokens relative to SC',19,color='#637087')
text('saving_header',719,302,116,28,'Saving',19,color='#637087')
text('zero',251,332,35,22,'0',16,color='#637087')
text('full',661,332,62,22,'100%',16,color='#637087')
for i,(r,saving) in enumerate(zip(rows,savings)):
    y=363+i*43
    name='Llama · GSM8K' if i==0 else 'DeepSeek · '+r['dataset']
    text(f'label{i}',43,y-1,209,28,name,19)
    node(f'sc_budget{i}',270,y,420,26,fill=GREY,radius=4)
    node(f'pc_budget{i}',270,y,420*(1-saving/100),26,fill=PURPLE if i==0 else TEAL,radius=4)
    text(f'saving{i}',724,y-2,99,30,f'{saving:.1f}%',23,True,PURPLE if i==0 else TEAL)
node('legend_sc',273,591,22,12,fill=GREY,radius=2)
text('legend_sclabel',300,582,143,28,'SC budget',17)
node('legend_pc',485,591,22,12,fill=TEAL,radius=2)
text('legend_pclabel',512,582,185,28,'PC generated tokens',17)

text('right_head',916,138,480,33,'(b) Works with early-stopping methods',24,True)
text('additional',937,182,442,83,'7%',65,True,TEAL)
text('additional_caption',934,263,447,27,'additional token saving',22)
text('samebudget',934,305,447,27,'Each parent normalized to 100%',18,color='#637087')
for i,(parent,tok,pc) in enumerate([('AC',2730,2539),('ESC',2552,2374)]):
    y=371+111*i
    text(f'parent{i}',921,y-3,111,29,parent,23,True)
    node(f'parentbar{i}',1048,y,305,23,fill=GREY,radius=4)
    text(f'parentvalue{i}',1356,y-3,53,29,'100',18)
    text(f'composed{i}',921,y+29,111,29,parent+' + PC',21,True)
    node(f'composedbar{i}',1048,y+32,305*pc/tok,23,fill=TEAL,radius=4)
    text(f'composedvalue{i}',1337,y+29,64,29,f'{100*pc/tok:.1f}',18,True,TEAL)
text('right_foot',929,585,460,27,'Llama-3.1-8B-Instruct · GSM8K',18,color='#637087')
text('foot1',28,647,1383,22,'SC = self-consistency. Bars show generated tokens, not total compute or API billing.',17,color='#637087')
text('foot2',28,671,1383,20,'Panel (a) uses the reported grading, including DeepSeek regrading. Mean savings are unweighted across settings.',16,color='#637087')


E.indent(mx)
out=HERE/'fig1_teaser_v1.drawio'
out.write_text(E.tostring(mx,encoding='unicode')+'\n')
svg.set('content',E.tostring(mx,encoding='unicode'))
E.indent(svg)
preview=HERE/'fig1_teaser_v1.drawio.svg'
preview.write_text(E.tostring(svg,encoding='unicode')+'\n')

def render():
    r=C.CDLL('librsvg-2.so.2'); c=C.CDLL('libcairo.so.2'); g=C.CDLL('libgobject-2.0.so.0')
    ptr=C.c_void_p; dbl=C.c_double; integer=C.c_int; string=C.c_char_p
    def bind(lib,name,args,ret):
        f=getattr(lib,name); f.argtypes=args; f.restype=ret; return f
    handle=bind(r,'rsvg_handle_new_from_file',[string,ptr],ptr)(str(preview).encode(),None)
    assert handle
    class Rect(C.Structure): _fields_=[('x',dbl),('y',dbl),('width',dbl),('height',dbl)]
    surface=bind(c,'cairo_image_surface_create',[integer,integer,integer],ptr)(0,W,H)
    ctx=bind(c,'cairo_create',[ptr],ptr)(surface)
    assert bind(r,'rsvg_handle_render_document',[ptr,ptr,C.POINTER(Rect),ptr],integer)(handle,ctx,C.byref(Rect(0,0,W,H)),None)
    assert bind(c,'cairo_surface_write_to_png',[ptr,string],integer)(surface,str(HERE/'fig1_teaser_v1.png').encode())==0
    bind(c,'cairo_destroy',[ptr],None)(ctx)
    bind(c,'cairo_surface_destroy',[ptr],None)(surface)
    pdf=bind(c,'cairo_pdf_surface_create',[string,dbl,dbl],ptr)(str(HERE/'fig1_teaser_v1.pdf').encode(),W*.5,H*.5)
    ctx=bind(c,'cairo_create',[ptr],ptr)(pdf)
    bind(c,'cairo_scale',[ptr,dbl,dbl],None)(ctx,.5,.5)
    assert bind(r,'rsvg_handle_render_document',[ptr,ptr,C.POINTER(Rect),ptr],integer)(handle,ctx,C.byref(Rect(0,0,W,H)),None)
    bind(c,'cairo_destroy',[ptr],None)(ctx)
    bind(c,'cairo_surface_finish',[ptr],None)(pdf)
    bind(c,'cairo_surface_destroy',[ptr],None)(pdf)
    bind(g,'g_object_unref',[ptr],None)(handle)

render()
cells=E.parse(out).findall('.//mxCell')
ids=[c.get('id') for c in cells]
assert len(ids)==len(set(ids))
for cell in cells:
    for field in ('source','target','parent'):
        assert cell.get(field) is None or cell.get(field) in ids
    if cell.get('edge')=='1': assert cell.find('mxGeometry').get('relative')=='1'
assert 'shape=image' not in out.read_text()
print(f'Validated {len(cells)} native cells. Created {out.name} and vector preview.')
