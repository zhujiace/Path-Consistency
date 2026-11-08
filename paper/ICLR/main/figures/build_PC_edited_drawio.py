"""将 PC_edited.png 的元素重建为原生 draw.io；附带同几何 SVG 检查预览。"""
from pathlib import Path
import xml.etree.ElementTree as E
import math
import ctypes as C

HERE = Path(__file__).resolve().parent
W, H = 1684, 934
INK = '#242424'
GREEN, RED, PURPLE = '#B7DCAA', '#EEA8A6', '#BAA6D9'
mx = E.Element('mxfile', host='app.diagrams.net', type='device')
d = E.SubElement(mx, 'diagram', id='pc-edited', name='PC method overview')
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

node('question',13,244,154,122,fill='#F8FBFD',stroke=INK,radius=5)
node('question_label',17,254,146,33,'Question q',kind='text',size=23,bold=True)
for i,length in enumerate([116,116,95,74]): node(f'question_line_{i}',31,296+i*15,length,6,fill='#ABB1B8',radius=3)

for j,(x,w,tx,tw,ax,cx) in enumerate([(199,296,239,188,443,345),(667,280,705,175,895,807),(1104,272,1140,171,1325,1237)]):
    tag=f'w{j+1}'
    node(tag,x,135,w,267,fill='#FFFFFF',stroke=INK,radius=6)
    node(tag+'_header',x+1,136,w-2,66,fill='#E9F2FC',radius=5)
    node(tag+'_title',x+2,143,w-4,33,f'Window {j+1}',kind='text',size=28,bold=True)
    node(tag+'_state',x+2,174,w-4,27,['root, r = 0','trial, r = 0.2','trial, r = 0.4'][j],kind='text',size=23)
    for row in range(4):
        y=218+45*row
        for k in range(7):
            fill=PURPLE if k<j*2 else (GREEN if row<3 else RED)
            node(f'{tag}_path{row}_token{k}',tx+k*tw/7,y,tw/7-1.4,31,fill=fill,radius=1)
        node(f'{tag}_answer{row}',ax,y-5,39,39,'A' if row<3 else 'B',fill=GREEN if row<3 else RED,kind='ellipse',size=25)
    node(tag+'_selected',x+8,222,23,25,fill=INK,kind='star')
    node(tag+'_validate',cx-98,418,196,100,'Validate',fill='#FFFFFF',stroke=INK,kind='diamond',size=25,bold=True)
    edge(tag+'_validation',[(cx,402),(cx,418)],tag,tag+'_validate')
    # 原生折线画勾叉，不依赖符号字体。
    checkx=[477,953,1380][j]
    edge(tag+'_pass_mark',[(checkx,447),(checkx+10,458),(checkx+30,434)],color='#29914A',arrow=False,width=5)
    for k,pts in enumerate([[(cx+22,531),(cx+46,553)],[(cx+46,531),(cx+22,553)]]):
        edge(tag+f'_fail_mark{k}',pts,color='#CD3E39',arrow=False,width=5)

for j,(x,w) in enumerate([(509,139),(960,128),(1387,116)]):
    tag=f'extract{j+1}'
    node(tag,x,248,w,104,fill='#FFFFFF',stroke=INK,radius=5)
    label=['extract prefix p\n(20%)','extract longer\nprefix p′ (40%)','extract longer\nprefix p″ (60%)'][j]
    node(tag+'_label',x+3,259,w-6,47,label,kind='text',size=[19,18,16][j])
    for k in range(7):
        tilew=(w-18)/7
        node(tag+f'_token{k}',x+9+k*tilew,313,tilew-1,27,fill=PURPLE if k<[2,3,4][j] else '#E0E4E9',radius=1)

node('vote',1556,244,117,117,fill='#FFFFFF',stroke=INK,radius=5)
node('vote_title',1558,254,113,27,'Majority vote',kind='text',size=18,bold=True)
node('final_answer',1583,291,69,59,'A',fill=GREEN,size=38,radius=8)
node('budget',1481,351,78,54,'until\nbudget B',kind='text',size=22)
node('failure',724,599,231,75,'Root: keep sampling\nOtherwise: rollback',fill='#FFFFFF',stroke=INK,size=22,radius=6)

edge('question_to_root',[(167,300),(199,300)],'question','w1')
for j,(right,ex,ew,nextx,cx) in enumerate([(495,509,139,667,345),(947,960,128,1104,807),(1376,1387,116,1556,1237)]):
    edge(f'w{j+1}_to_extract',[(right,300),(ex,300)],f'w{j+1}',f'extract{j+1}')
    edge(f'extract{j+1}_next',[(ex+ew,300),(nextx,300)],f'extract{j+1}',f'w{j+2}' if j<2 else 'vote',dash=j==2)
    excenter=ex+ew/2
    edge(f'w{j+1}_pass',[(cx+98,468),(excenter,468),(excenter,352)],f'w{j+1}_validate',f'extract{j+1}')
edge('root_failure',[(345,518),(345,635),(724,635)],'w1_validate','failure')
edge('trial2_failure',[(807,518),(807,599)],'w2_validate','failure')
edge('trial3_failure',[(1237,518),(1237,635),(955,635)],'w3_validate','failure')

node('legend',266,735,1152,73,fill='#FFFFFF',stroke='#59636C',radius=4,dash=True)
for i,(x,color,label,tw) in enumerate([(315,PURPLE,'Shared prefix (reused)',261),(685,GREEN,'Correct path / answer',265),(1048,RED,'Incorrect path / answer',277)]):
    node(f'legend_swatch{i}',x,749,67,45,fill=color,radius=4)
    node(f'legend_text{i}',x+81,749,tw,45,label,kind='text',size=24)

E.indent(mx)
out=HERE/'PC_edited.drawio'
out.write_text(E.tostring(mx,encoding='unicode')+'\n')
svg.set('content',E.tostring(mx,encoding='unicode'))
E.indent(svg)
preview=HERE/'PC_edited.drawio.svg'
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
    assert bind(c,'cairo_surface_write_to_png',[ptr,string],integer)(surface,str(HERE/'PC_edited_drawio_preview.png').encode())==0
    bind(c,'cairo_destroy',[ptr],None)(ctx)
    bind(c,'cairo_surface_destroy',[ptr],None)(surface)
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
