"""只读 Llama/GSM8K 完整分片，生成8窗口汇总及原生 LaTeX 矢量图。"""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from build_results import get_pair, digest, HERE

REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from path_consistency.math_eval import majority_vote, grade_answer


def behavior(row):
    windows = row['windows']
    if not any(w.get('used_prefix_tokens', 0) > 0 for w in windows):
        return 'Never uses a prefix'
    if any('rollback' in w.get('decision', '') for w in windows):
        return 'Prefix with rollback'
    return 'Prefix without rollback'


def cumulative_correct(row, j):
    answers = [t['answer'] for t in row['traces'] if t['window'] <= j]
    return int(grade_answer('gsm8k', majority_vote(answers), row['gold']))


def picture(rows):
    # 标准 LaTeX picture/qbezier：可复制、无额外绘图库、非位图。
    out = [r'\setlength{\unitlength}{1pt}', r'\begin{picture}(470,242)']
    def label(x,y,s,align='l'):
        out.append(rf'\put({x:.2f},{y:.2f}){{\makebox(0,0)[{align}]{{\scriptsize {s}}}}}')
    def segment(x,y,u,v):
        out.append(rf'\qbezier({x:.2f},{y:.2f})({(x+u)/2:.2f},{(y+v)/2:.2f})({u:.2f},{v:.2f})')
    panels = [
        (0,128,'(a) Questions using a prefix (\%)',0,100,[0,50,100], [('prefix_use_pct','PC','blue')]),
        (240,128,'(b) Used prefix tokens / question',0,160,[0,80,160], [('prefix_tokens','PC','blue')]),
        (0,6,'(c) New tokens / window / question',0,1200,[0,600,1200], [('sc_tokens','SC','black'),('pc_tokens','PC','blue')]),
        (240,6,'(d) Cumulative vote accuracy (\%)',85,95,[85,90,95], [('sc_accuracy','SC','black'),('pc_accuracy','PC','blue')]),
    ]
    for ox,oy,title,lo,hi,ticks,series in panels:
        x0,y0,w,h=ox+31,oy+23,184,72
        label(ox+6,oy+109,title)
        segment(x0,y0,x0+w,y0);segment(x0,y0,x0,y0+h)
        for t in ticks:
            y=y0+h*(t-lo)/(hi-lo)
            segment(x0-2,y,x0,y);label(x0-5,y,str(t),'r')
        for j in range(8):
            x=x0+w*j/7
            segment(x,y0,x,y0-2);label(x,y0-8,str(j+1),'c')
        label(x0+w/2,y0-19,'Window','c')
        for si,(key,name,color) in enumerate(series):
            values=[r[key] for r in rows]
            assert all(lo <= v <= hi for v in values), (key,values)
            out.append(r'{\color{'+color+r'}\linethickness{0.7pt}')
            points=[(x0+w*j/7,y0+h*(v-lo)/(hi-lo)) for j,v in enumerate(values)]
            for p,q in zip(points,points[1:]): segment(*p,*q)
            for x,y in points:
                mark=r'\circle*{2.5}' if name=='PC' else r'\framebox(2.5,2.5){}'
                out.append(rf'\put({x:.2f},{y:.2f}){{{mark}}}')
            label(x0+110+si*38,y0+h+5,name)
            out.append('}')
    out.append(r'\end{picture}')
    return '\n'.join(out)+'\n'


def main():
    n=1319
    sc,pc,sources,_,_=get_pair(
        ['7bdaa31cd3ac','6ae254915afa','203bd205a5ef','34219eb19c54'],n,'GSM8K')
    groups={name:dict(n=0,sc_correct=0,pc_correct=0,sc_tokens=0,pc_tokens=0,gains=0,losses=0)
            for name in ['Prefix without rollback','Prefix with rollback','Never uses a prefix']}
    for key,p in pc.items():
        s=sc[key]
        for row,method in [(s,'SC'),(p,'PC')]:
            assert len(row['traces'])==32
            assert [w['window'] for w in row['windows']]==list(range(8))
            assert Counter(t['window'] for t in row['traces'])==Counter({j:4 for j in range(8)})
            assert sum(t['generated_tokens'] for t in row['traces'])==row['total_generated_tokens']
            assert cumulative_correct(row,7)==int(row['predictions'][method]['correct'])
        g=groups[behavior(p)];g['n']+=1
        a=int(s['predictions']['SC']['correct']);b=int(p['predictions']['PC']['correct'])
        g['sc_correct']+=a;g['pc_correct']+=b;g['gains']+=int(b>a);g['losses']+=int(a>b)
        g['sc_tokens']+=s['total_generated_tokens'];g['pc_tokens']+=p['total_generated_tokens']
    rows=[]
    for j in range(8):
        ws=[p['windows'][j] for p in pc.values()]
        row=dict(window=j+1,prefix_questions=sum(w['used_prefix_tokens']>0 for w in ws),
                 prefix_tokens=sum(w['used_prefix_tokens'] for w in ws)/n,
                 rollback_windows=sum('rollback' in w['decision'] for w in ws))
        row['prefix_use_pct']=100*row['prefix_questions']/n
        for name,pool in [('sc',sc),('pc',pc)]:
            row[name+'_correct']=sum(cumulative_correct(p,j) for p in pool.values())
            row[name+'_accuracy']=100*row[name+'_correct']/n
            row[name+'_tokens']=sum(t['generated_tokens'] for p in pool.values()
                                    for t in p['traces'] if t['window']==j)/n
        rows.append(row)
    assert sum(g['n'] for g in groups.values())==n
    assert sum(g['sc_correct'] for g in groups.values())==1215
    assert sum(g['pc_correct'] for g in groups.values())==1210
    assert rows[0]['sc_correct']==rows[0]['pc_correct']
    assert rows[0]['sc_tokens']==rows[0]['pc_tokens']
    for g in groups.values():g['saving_pct']=100*(1-g['pc_tokens']/g['sc_tokens'])
    levels=[]
    for level in range(5):
        ratio=level/5
        pairs=[(key,j) for key,p in pc.items() for j,w in enumerate(p['windows'])
               if abs(w['used_ratio']-ratio)<1e-8]
        assert pairs
        rec=dict(level=level,ratio=ratio,windows=len(pairs),
                 prefix_tokens=sum(pc[k]['windows'][j]['used_prefix_tokens'] for k,j in pairs)/len(pairs))
        for name,pool in [('sc',sc),('pc',pc)]:
            rec[name+'_tokens']=sum(t['generated_tokens'] for k,j in pairs
                                    for t in pool[k]['traces'] if t['window']==j)/len(pairs)
            rec[name+'_seconds']=sum(pool[k]['windows'][j]['generation_seconds'] for k,j in pairs)/len(pairs)
        rec['token_saving']=100*(1-rec['pc_tokens']/rec['sc_tokens'])
        rec['time_reduction']=100*(1-rec['pc_seconds']/rec['sc_seconds'])
        levels.append(rec)
    assert sum(r['windows'] for r in levels)==n*8
    payload=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),n=n,
                 sources=sources,grader_sha256=digest(REPO/'path_consistency/math_eval.py'),
                 grouping='post-hoc observed PC behavior, not randomized treatment',
                 windows=rows,groups=groups,levels=levels)
    (HERE/'data/window_snapshot.json').write_text(json.dumps(payload,indent=2)+'\n')
    table=[]
    for name,g in groups.items():
        table.append(f"{name} & {g['n']} & {g['sc_correct']} & {g['pc_correct']} & {g['gains']}/{g['losses']} & {g['saving_pct']:.1f}"+r' \\')
    (HERE/'tables/behavior_rows.tex').write_text('\n'.join(table)+'\n')
    table=[]
    for r in rows:
        table.append(f"{r['window']} & {r['prefix_questions']} & {r['prefix_tokens']:.1f} & {r['rollback_windows']} & {r['sc_tokens']:.1f} & {r['pc_tokens']:.1f} & {r['sc_correct']} & {r['pc_correct']}"+r' \\')
    (HERE/'tables/window_rows.tex').write_text('\n'.join(table)+'\n')
    table=['% Generated by scripts/build_windows.py; matched question/window SC baseline.']
    for r in levels:
        table.append(f"{r['level']} ({100*r['ratio']:.0f}\\%) & {r['windows']:,} & {r['prefix_tokens']:.1f} & "
                     f"{r['pc_tokens']:.1f} & {r['token_saving']:.1f} & {r['time_reduction']:.1f}"+r' \\')
    (HERE/'tables/prefix_level_rows.tex').write_text('\n'.join(table)+'\n')
    (HERE/'figures').mkdir(exist_ok=True)
    (HERE/'figures/window_evolution.tex').write_text(picture(rows))
    print(json.dumps(dict(windows=rows,groups=groups),indent=2))


if __name__=='__main__':main()
