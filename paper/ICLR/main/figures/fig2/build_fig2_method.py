"""生成原生 draw.io，再从其可编辑几何导出 SVG；用本机 librsvg/Cairo 导出 PDF。
无需联网、GPU或draw.io桌面。仅支持本图使用的矩形、文本与折线样式。
"""
from pathlib import Path
import ctypes as C
import xml.etree.ElementTree as E
from html.parser import HTMLParser

HERE=Path(__file__).resolve().parent
W,H=1420,650
BLUE='#285C78'; INK='#202A32'; GRAY='#59636C'
mx=E.Element('mxfile',host='app.diagrams.net',type='device')
diagram=E.SubElement(mx,'diagram',id='pc-method',name='Figure 2 — Method Overview')
model=E.SubElement(diagram,'mxGraphModel',dx=str(W),dy=str(H),grid='1',gridSize='10',
                    page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),background='#FFFFFF',math='0')
root=E.SubElement(model,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')

def node(id,x,y,w,h,value='',box=False,bold=False,size=22,color=INK,fill='#FFFFFF',align='center'):
    style=f'html=1;whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={size};fontColor={color};align={align};verticalAlign=middle;spacing=0;'
    style+=f'fillColor={fill};strokeColor={INK};strokeWidth=1.5;' if box else 'shape=text;fillColor=none;strokeColor=none;'
    if bold:style+='fontStyle=1;'
    c=E.SubElement(root,'mxCell',id=id,value=value,style=style,vertex='1',parent='1')
    E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    return c

def edge(id,points,color=INK,arrow=True):
    c=E.SubElement(root,'mxCell',id=id,value='',edge='1',parent='1',style=
        f'html=1;rounded=0;strokeColor={color};strokeWidth=1.6;endArrow={"block" if arrow else "none"};endFill=1;endSize=7;')
    g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    for name,p in [('sourcePoint',points[0]),('targetPoint',points[-1])]:
        E.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),attrib={'as':name})
    if len(points)>2:
        a=E.SubElement(g,'Array',attrib={'as':'points'})
        for p in points[1:-1]:E.SubElement(a,'mxPoint',x=str(p[0]),y=str(p[1]))

# 路径演化为主，所有元素均为原生可编辑形状。
PURPLE='#7762B5'; PREFIX='#B6A3DC'; TEAL='#59AD9C'; LIGHT='#D2EAE3'
ORANGE='#F2C779'; RED='#C77576'; PALE='#F7F5FB'

def shape(id,x,y,w,h,fill,stroke='none',kind='rect',rounding=0,dash=False):
    c=node(id,x,y,w,h,box=True,fill=fill)
    s=c.get('style').replace(f'strokeColor={INK}',f'strokeColor={stroke}')
    if kind!='rect':s+='shape='+kind+';'
    if rounding:s=s.replace('rounded=0','rounded=1')+f'arcSize={rounding};'
    if dash:s+='dashed=1;'
    c.set('style',s)
    return c

def line(id,pts,color=INK,arrow=True,dash=False,width=1.8):
    edge(id,pts,color,arrow)
    c=root.find(f"mxCell[@id='{id}']")
    c.set('style',c.get('style').replace('strokeWidth=1.6',f'strokeWidth={width}')+('dashed=1;' if dash else ''))

def text(id,x,y,w,h,label,size=22,color=INK,bold=False,align='center'):
    node(id,x,y,w,h,label,bold=bold,size=size,color=color,align=align)

def trace(id,x,y,length,prefix=0,answer='A',highlight=False):
    if prefix:
        shape(id+'prefix',x,y,prefix,23,PREFIX,kind='rect',rounding=16)
    shape(id+'new',x+prefix,y,length-prefix,23,LIGHT,kind='rect',rounding=16)
    # 细分刻度表示推理步骤，而不是额外的算法模块。
    for offset in range(prefix+24,length-8,27):
        line(id+f'tick{offset}',[(x+offset,y+5),(x+offset,y+18)],'#A9CFC4',False,width=1)
    if highlight:shape(id+'selected',x-4,y-4,42,31,'none',ORANGE,rounding=10)
    shape(id+'answer',x+211,y-1,26,26,ORANGE if answer=='A' else '#B9CFEB',kind='ellipse')
    text(id+'letter',x+211,y-1,26,26,answer,size=17,bold=True)

# 灰色SC对照带，只说明完整路径重复生成；不填实测性能数字。
shape('sc_band',18,14,1384,193,'#F7F8FA',rounding=5)
text('sc_heading',34,44,164,61,'Self-<br/>consistency',26,BLUE,True)
text('sc_sub',34,113,164,49,'Full paths<br/>every window',20,GRAY)
for j,x in enumerate([231,561,891]):
    text(f'sc_window{j}',x,25,246,30,['Window 1','Window 2','Window 3 … 8'][j],21,GRAY)
    for i,n in enumerate([177,157,190,170]):
        trace(f'sc{j}_{i}',x,65+i*32,n,answer=['A','A','B','A'][i])
    if j<2:line(f'sc_next{j}',[(x+252,122),(x+308,122)],'#98A3AE')
text('sc_vote',1218,87,155,34,'Majority vote',23,BLUE,True)
shape('sc_final',1270,132,49,42,ORANGE,rounding=20)
text('sc_answer',1270,132,49,42,'A',26,INK,True)
line('sc_to_vote',[(1138,122),(1206,122)],'#98A3AE')

# PC主视图：三个窗口足以表达提取、试用和更长前缀，之后用省略符。
text('pc_heading',32,237,172,55,'Path-<br/>Consistency',27,PURPLE,True)
text('illustration_note',31,293,174,27,'Illustrative sequence',17,GRAY)
shape('prompt_card',39,330,147,94,'#EDF2F8','#AABCD0',rounding=15)
text('question',48,342,129,30,'Question',23,BLUE,True)
for i,n in enumerate([94,107,74]):line(f'promptline{i}',[(60,382+i*10),(60+n,382+i*10)],'#AABCD0',False,width=2)
line('prompt_to_root',[(188,376),(220,376)],PURPLE)

for j,x in enumerate([220,550,880]):
    shape(f'window_panel{j}',x,302,267,173,'#FFFFFF','#C9CCD5',rounding=8)
    text(f'window_title{j}',x,242,267,34,f'Window {j+1}',25,INK,True)
    text(f'window_state{j}',x,275,267,25,['Root · <i>r</i> = 0','Trial · <i>r</i> = 0.2','Trial · <i>r</i> = 0.4'][j],21,PURPLE)
    for i,n in enumerate([177,157,190,170]):
        trace(f'pc{j}_{i}',x+11,317+i*38,n,prefix=[0,35,70][j],
              answer=['A','A','B','A'][i],highlight=(i==0 and j==0))

    # 源选择或验证位于窗口下方，使用短标签替代规则段落。
    if j==0:
        shape('gate_chip',x,489,267,44,'#FFF3D9','#E4BE6D',rounding=22)
        text('gate_short',x+8,492,251,38,'Gate + BLC → extract',21,INK,True)
    else:
        shape(f'validate_chip{j}',x,489,267,44,'#EEE9F7','#B6A3DC',rounding=22)
        text(f'validate_short{j}',x+8,492,251,38,'Validate → commit',20,PURPLE,True)
    if j<2:
        line(f'continue{j}',[(x+269,388),(x+318,388)],PURPLE,width=2.5)
        text(f'reuse_label{j}',x+265,347,66,29,'reuse',18,PURPLE)

# 聚合视觉：括号覆盖全部窗口，不把通过门控的子集误当最终投票池。
text('ellipsis',1152,360,73,43,'⋯',37,GRAY)
text('later_windows',1147,400,70,26,'to 8',18,GRAY)
text('vote_title',1245,298,155,59,'Majority<br/>vote',25,BLUE,True)
shape('final_answer_card',1261,373,118,78,'#FFF0CB','#DDB566',rounding=22)
text('answer_label',1261,381,118,29,'Answer',19,GRAY)
text('final_A',1261,411,118,34,'A',32,INK,True)
line('collect_bracket',[(220,542),(220,562),(1205,562),(1205,429),(1252,429)],BLUE,width=2)
line('bracket_end',[(1147,542),(1147,562)],BLUE,False)
text('all_votes',666,567,519,30,'All answers, including rejected windows',20,BLUE)

# 回退为局部异常标注，无冗长控制流程树。
shape('rollback_tag',227,568,409,32,'#F9ECEC',rounding=14)
text('rollback_note',235,568,393,32,'Failed validation ↶ rollback',20,RED)

# 限量图例：编码语义，而非给路径贴“正确”标签。
shape('legend_prefix',28,619,25,16,PREFIX,rounding=10)
text('legend_prefix_text',61,609,135,34,'Shared prefix',19,GRAY,align='left')
shape('legend_new',215,619,25,16,LIGHT,rounding=10)
text('legend_new_text',248,609,121,34,'New tokens',19,GRAY,align='left')
shape('legend_A',389,615,23,23,ORANGE,kind='ellipse')
shape('legend_B',417,615,23,23,'#B9CFEB',kind='ellipse')
text('legend_answer_text',452,609,216,34,'Answer categories',19,GRAY,align='left')
text('budget_v2',865,609,513,34,'<i>B</i> = 32   ·   <i>m</i> = 4   ·   <i>C</i> = 0.8',21,GRAY,align='right')

path=HERE/'fig2_method.drawio'
E.indent(mx);path.write_text(E.tostring(mx,encoding='unicode')+'\n')

class Text(HTMLParser):
    def __init__(self):super().__init__();self.lines=[[]];self.tags=[]
    def handle_starttag(self,tag,attrs):
        if tag=='br':self.lines.append([])
        else:self.tags.append(tag)
    def handle_endtag(self,tag):
        if tag in self.tags:self.tags.remove(tag)
    def handle_data(self,data):self.lines[-1].append((data,list(self.tags)))

def export_svg():
    src=E.parse(path).getroot()
    svg=E.Element('svg',xmlns='http://www.w3.org/2000/svg',width=str(W),height=str(H),viewBox=f'0 0 {W} {H}')
    E.SubElement(svg,'title').text='Figure 2: Path-Consistency method overview'
    E.SubElement(svg,'metadata',id='editable-drawio-source').text=path.read_text()
    E.SubElement(svg,'rect',width=str(W),height=str(H),fill='white')
    defs=E.SubElement(svg,'defs')
    colors=sorted({INK,BLUE,*[dict(x.split('=',1) for x in c.get('style','').split(';') if '=' in x).get('strokeColor',INK) for c in src.findall('.//mxCell') if c.get('edge')=='1']})
    for color in colors:
        name='arrow'+color.lstrip('#')
        m=E.SubElement(defs,'marker',id=name,markerWidth='7',markerHeight='7',refX='6',refY='3.5',orient='auto',markerUnits='userSpaceOnUse')
        E.SubElement(m,'path',d='M0 0 L7 3.5 L0 7 Z',fill=color)
    for c in src.findall('.//mxCell'):
        if c.get('vertex')!='1' and c.get('edge')!='1':continue
        style=dict(x.split('=',1) for x in c.get('style','').split(';') if '=' in x)
        geo=c.find('mxGeometry');color=style.get('strokeColor',INK)
        if c.get('edge')=='1':
            points=[geo.find("mxPoint[@as='sourcePoint']"),*geo.findall('Array/mxPoint'),geo.find("mxPoint[@as='targetPoint']")]
            d='M'+' L'.join(f"{p.get('x')} {p.get('y')}" for p in points)
            a=dict(d=d,fill='none',stroke=color,attrib={'stroke-width':style['strokeWidth']})
            el=E.SubElement(svg,'path',**a)
            if style.get('dashed')=='1':el.set('stroke-dasharray','5 4')
            if style.get('endArrow')!='none':el.set('marker-end',f'url(#arrow{color.lstrip("#")})')
            continue
        x,y,w,h=[float(geo.get(k)) for k in ['x','y','width','height']]
        if style.get('shape')!='text':
            props=dict(fill=style['fillColor'],stroke=color,attrib={'stroke-width':style['strokeWidth']})
            if style.get('shape')=='ellipse':
                el=E.SubElement(svg,'ellipse',cx=str(x+w/2),cy=str(y+h/2),rx=str(w/2),ry=str(h/2),**props)
            else:
                el=E.SubElement(svg,'rect',x=str(x),y=str(y),width=str(w),height=str(h),**props)
                if style.get('rounded')=='1':el.set('rx',str(min(w,h)*float(style.get('arcSize','10'))/100))
            if style.get('dashed')=='1':el.set('stroke-dasharray','5 4')
        value=c.get('value','')
        if not value:continue
        parser=Text();parser.feed(value);size=float(style['fontSize']);lh=size*1.18
        anchor={'center':'middle','left':'start','right':'end'}[style['align']]
        tx=x+({'center':w/2,'left':0,'right':w}[style['align']])
        for i,line in enumerate(parser.lines):
            ty=y+h/2+(i-(len(parser.lines)-1)/2)*lh+size*.34
            text=E.SubElement(svg,'text',x=str(tx),y=str(ty),fill=style['fontColor'],attrib={
                'font-family':'Arial, Liberation Sans, sans-serif','font-size':str(size),'text-anchor':anchor,
                'font-weight':'bold' if style.get('fontStyle')=='1' else 'normal'})
            for data,tags in line:
                attrs={}
                if 'i' in tags:attrs['font-style']='italic'
                if 'sub' in tags:attrs.update({'baseline-shift':'sub','font-size':'70%'})
                E.SubElement(text,'tspan',attrs).text=data
    E.indent(svg);target=HERE/'fig2_method.svg';target.write_text(E.tostring(svg,encoding='unicode')+'\n')
    return target

def render(svg):
    r=C.CDLL('librsvg-2.so.2');c=C.CDLL('libcairo.so.2');g=C.CDLL('libgobject-2.0.so.0')
    def bind(lib,name,args,ret):f=getattr(lib,name);f.argtypes=args;f.restype=ret;return f
    ptr=C.c_void_p; dbl=C.c_double; integer=C.c_int; string=C.c_char_p
    create=bind(r,'rsvg_handle_new_from_file',[string,ptr],ptr)
    handle=create(str(svg).encode(),None);assert handle
    class Rect(C.Structure):_fields_=[('x',dbl),('y',dbl),('width',dbl),('height',dbl)]
    paint=bind(r,'rsvg_handle_render_document',[ptr,ptr,C.POINTER(Rect),ptr],integer)
    context=bind(c,'cairo_create',[ptr],ptr)
    status=bind(c,'cairo_status',[ptr],integer)
    finish=bind(c,'cairo_surface_finish',[ptr],None)
    destroy=bind(c,'cairo_surface_destroy',[ptr],None)
    dc=bind(c,'cairo_destroy',[ptr],None)
    for kind in ['pdf','png']:
        if kind=='pdf':surface=bind(c,'cairo_pdf_surface_create',[string,dbl,dbl],ptr)(str(HERE/'fig2_method.pdf').encode(),W*.75,H*.75)
        else:surface=bind(c,'cairo_image_surface_create',[integer,integer,integer],ptr)(0,W,H)
        ctx=context(surface)
        if kind=='pdf':bind(c,'cairo_scale',[ptr,dbl,dbl],None)(ctx,.75,.75)
        assert paint(handle,ctx,C.byref(Rect(0,0,W,H)),None)
        assert status(ctx)==0
        if kind=='png':assert bind(c,'cairo_surface_write_to_png',[ptr,string],integer)(surface,str(HERE/'fig2_method_preview.png').encode())==0
        finish(surface);dc(ctx);destroy(surface)
    bind(g,'g_object_unref',[ptr],None)(handle)

render(export_svg())
print('Created fig2_method.drawio, fig2_method.svg, fig2_method.pdf and SVG-rendered preview.')
