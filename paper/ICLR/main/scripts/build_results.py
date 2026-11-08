"""只读冻结运行，生成可追溯的小型论文汇总，不导出题目或轨迹。"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parents[1]
ROOT = Path('/data/26xyw/path-consistency-results')
EXCLUDE = {303,457,359,315,316,415,407,370,151,332}
SETS = [
 ('Llama', 'MATH-500',490,['abc27a862c6a']),
 ('Llama', 'AIME 2024',30,['8e57f7657fc7']),
 ('Llama', 'AIME 2025',30,['54f0524edc4c']),
 ('Llama', 'GPQA-D',198,['dc17da48aca4']),
 ('DeepSeek', 'MATH-500',490,['4856d4ef5673']),
 ('DeepSeek', 'AIME 2024',30,['e4cd6dcd6672']),
 ('DeepSeek', 'AIME 2025',30,['771c0b2351c6']),
 ('DeepSeek', 'GPQA-D',198,['a8a29bd70ad3']),
 ('Llama', 'GSM8K',1319,['7bdaa31cd3ac','6ae254915afa','203bd205a5ef','34219eb19c54']),
 ('DeepSeek', 'GSM8K (original)',1319,['4829b0ab9213','7028521973e2','455c760d2e21']),
]

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()

def read_rows(path):
 rows={}
 with path.open() as f:
  for line in f:
   r=json.loads(line); key=r['question_id']
   if key in rows: raise ValueError(f'Duplicate question: {path.name}')
   rows[key]=r
 return rows

def correct(r, method):
 return bool(r['predictions'][method]['correct'])

def summarize(rows, method):
 n=len(rows); vals=list(rows.values())
 events=[w for r in vals for w in r.get('windows',[])]
 proposals=[w['proposed_prefix'] for w in events if 'proposed_prefix' in w]
 return dict(n=n,raw_correct=sum(correct(r,method) for r in vals),
  tok=sum(r['total_generated_tokens'] for r in vals)/n,
  seconds=sum(r['generation_seconds'] for r in vals)/n,
  parser_failures=sum(r.get('parser_failures',0) for r in vals),
  truncations=sum(r.get('truncations',0) for r in vals),
  traces=sum(len(r.get('traces',[])) for r in vals),
  windows=len(events),proposals=len(proposals),
  nonfinite_blc=sum(not math.isfinite(p.get('blc_score',float('-inf'))) for p in proposals),
  rollback_windows=sum('rollback' in w.get('decision','') for w in events),
  prefix_questions=sum(any(w.get('used_prefix_tokens',0)>0 for w in r.get('windows',[])) for r in vals),
  disabled_questions=sum(bool(r.get('prefixing_disabled')) for r in vals))

def paired(sc,pc,sm='SC',pm='PC'):
 assert sc.keys()==pc.keys()
 gains=sum(not correct(sc[k],sm) and correct(pc[k],pm) for k in sc)
 losses=sum(correct(sc[k],sm) and not correct(pc[k],pm) for k in sc)
 d=gains+losses
 p=min(1.,2*sum(math.comb(d,k) for k in range(min(gains,losses)+1))/2**d) if d else 1.
 # Exact two-sided sign/McNemar test on discordant question pairs; no seed inference.
 n=len(sc)
 def interval(k):
  # Each discordance probability receives a 97.5% Clopper--Pearson interval.
  def invert(count,target):
   lo,hi=0.,1.
   for _ in range(65):
    x=(lo+hi)/2
    cdf=sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
        +j*math.log(x)+(n-j)*math.log1p(-x)) for j in range(count+1))
    if cdf>target:lo=x
    else:hi=x
   return (lo+hi)/2
  return (invert(k-1,.9875) if k else 0.,invert(k,.0125) if k<n else 1.)
 gl,gu=interval(gains);ll,lu=interval(losses)
 return dict(gains=gains,losses=losses,delta_pp=100*(gains-losses)/n,
  conservative_95_ci_pp=[100*(gl-lu),100*(gu-ll)],mcnemar_exact_p=p)

def paths_for(fp):
 hits=list(ROOT.glob(f'*/*{fp}.run.json'))
 assert len(hits)==1,(fp,hits)
 return hits[0]

def get_pair(fps,n,ds,sm='sc',pm='pc'):
 left={};right={};sources=[];over={};p1=0
 for fp in fps:
  manifest=paths_for(fp);d=json.loads(manifest.read_text());c=d['config']
  assert d['status']=='complete'
  assert c['max_branches']==32 and c['window_size']==4 and c['beta_threshold']==.8
  assert 'paired-root-v2' in c['protocol_version']
  assert '32B' not in c['model']
  if 'deepseek' in c['model']: assert c['thinking'] is True
  stem=manifest.name.removesuffix('.run.json')
  a=manifest.with_name(stem+'_'+sm+'.jsonl');b=manifest.with_name(stem+'_'+pm+'.jsonl')
  ar=read_rows(a);br=read_rows(b)
  assert ar.keys()==br.keys()
  selected=c.get('selected_question_ids')
  if selected is not None: assert set(selected)==set(ar)
  assert not(left.keys() & ar.keys())
  if ds=='MATH-500': assert not {r['source_row'] for r in ar.values()} & EXCLUDE
  left.update(ar);right.update(br)
  source=dict(tag=stem,fingerprint=fp,manifest_sha256=digest(manifest),
   files={a.name:digest(a),b.name:digest(b)},
   config={k:c.get(k) for k in ['protocol_version','implementation_sha256','model','max_branches','window_size','beta_threshold','max_tokens','temperature','top_p','top_k','thinking','selector']})
  overlay=manifest.with_name(stem+'.numeric_regrade.json')
  if overlay.exists():
   o=json.loads(overlay.read_text());source['regrade_sha256']=digest(overlay)
   for key in ['SC','PC','Pass@1(SC)']:
    if key in o['methods']: over[key]=over.get(key,0)+o['methods'][key]['correct']
  if sm=='sc':p1+=sum(correct(r,'Pass@1(SC)') for r in ar.values())
  sources.append(source)
 assert len(left)==len(right)==n
 return left,right,sources,over,p1

def write(name,text):
 p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)

def main():
 out={'created_at_utc':datetime.now(timezone.utc).isoformat(),'sets':[],'combinations':[]}
 table=[];llama=[];raw=[];diag=[]
 for model,ds,n,fps in SETS:
  a,b,src,over,p1=get_pair(fps,n,ds)
  s=summarize(a,'SC');p=summarize(b,'PC')
  item=dict(model=model,dataset=ds,n=n,sources=src,sc=s,pc=p,raw_pair=paired(a,b),overlay=over,raw_pass1=p1)
  out['sets'].append(item)
  save=100*(1-p['tok']/s['tok']);lat=100*(1-p['seconds']/s['seconds'])
  p1n=over.get('Pass@1(SC)',p1);sn=over.get('SC',s['raw_correct']);pn=over.get('PC',p['raw_correct'])
  n_acc=n
  if ds=='MATH-500':
   # Raw JSONL only. Ten engineering-split items counted correct; n=500.
   p1n=p1+10; sn=s['raw_correct']+10; pn=p['raw_correct']+10; n_acc=500
  if 'GSM8K' not in ds:
   target=llama if model=='Llama' else table
   model_label=model if model=='Llama' or not target else ''
   target.append(f"{model_label} & {ds} & {100*p1n/n_acc:.2f} & {100*sn/n_acc:.2f} & {100*pn/n_acc:.2f} & {s['tok']:,.0f} & {p['tok']:,.0f} & {save:.1f} \\")
  pair=item['raw_pair']
  raw.append(f"{model} & {ds.replace(' (original)','')} & {s['raw_correct']}/{n} & {p['raw_correct']}/{n} & {pair['gains']}/{pair['losses']} & {pair['mcnemar_exact_p']:.3f} \\")
  diag.append(f"{model} & {ds.replace(' (original)','')} & {p['prefix_questions']}/{n} & {p['rollback_windows']}/{p['windows']} & {p['nonfinite_blc']}/{p['proposals']} & {s['seconds']:.1f} & {p['seconds']:.1f} \\")
 # 作者整理的 C=0.8 记录，与 Table 5 一致；不从 original JSONL 推断缺失的 Pass@1。
 # 该行跨来源，正文保留 dagger 与 Appendix threshold 的来源说明。
 table.append(r' & GSM8K$^\dagger$ & -- & 96.82 & 96.82 & 12,514 & 8,230 & 34.2 '+chr(92))
 for parent,fp,pm in [('AC','9d1997bfb7de','pc_ac'),('ESC','ab1a20b858bd','pc_esc')]:
  a,b,src,_,_=get_pair([fp],1319,'GSM8K',parent.lower(),pm)
  sk=next(k for k in next(iter(a.values()))['predictions'] if k.lower()==parent.lower())
  pk=next(k for k in next(iter(b.values()))['predictions'] if 'pass' not in k.lower() and 'offline' not in k.lower())
  out['combinations'].append(dict(parent=parent,sources=src,base=summarize(a,sk),plus=summarize(b,pk),pair=paired(a,b,sk,pk)))
 out['baselines']=[]
 for fp in ['eb5fffd3c21f','77b0d966405d','0703b9899c8f']:
  m=paths_for(fp);d=json.loads(m.read_text());assert d['status']=='complete'
  p=m.with_name(m.name.removesuffix('.run.json')+'_polr.jsonl')
  r=read_rows(p);assert len(r)==1319
  key=next(k for k in next(iter(r.values()))['predictions'] if k.lower()=='polr')
  out['baselines'].append(dict(method='PoLR',tag=m.stem,files={p.name:digest(p)},stats=summarize(r,key)))
 q=ROOT/'gsm8k/reports/sc_prefix_k_llama31_c08.json'
 out['sc_prefix_k']={'file':q.name,'sha256':digest(q),'summary':json.loads(q.read_text())['by_k']}
 q=ROOT/'deepconf-cpt/cpt_llama31_gsm8k_n32.summary.json'
 cpt=json.loads(q.read_text());assert cpt['status']=='complete' and cpt['n_questions']==1319
 out['cpt_adapter']={'file':q.name,'sha256':digest(q),'summary':cpt,'verification':'existing per-question summarizer output; timer scope differs'}
 comparison=[]
 def comp(label,n,c,t,s):
  comparison.append(f'{label} & {100*c/n:.2f} & {t:,.0f} & {s:.1f}'+r' \\')
 gsm=next(r for r in out['sets'] if r['model']=='Llama' and r['dataset']=='GSM8K')
 s=gsm['sc'];comp('SC-32',s['n'],s['raw_correct'],s['tok'],s['seconds'])
 s=out['sc_prefix_k']['summary']['20'];comp('SC-20 (same SC pool)',s['n'],s['correct'],s['tok_per_q'],s['latency_s'])
 for lp,r in zip([64,128,256],out['baselines']):
  if lp != 128: continue
  s=r['stats'];comp(f'PoLR ($L_p={lp}$)',s['n'],s['raw_correct'],s['tok'],s['seconds'])
 comp('CPT (32 workers, Llama adapter)',cpt['n_questions'],cpt['correct'],cpt['tok_per_q'],cpt['latency_s'])
 s=gsm['pc'];comp('PC ($C=0.8$)',s['n'],s['raw_correct'],s['tok'],s['seconds'])
 comparison.append(r'\midrule')
 for r in out['combinations']:
  parent=r['parent'];label=r'AC ($\tau=0.95$)' if parent=='AC' else 'ESC'
  s=r['base'];comp(label,s['n'],s['raw_correct'],s['tok'],s['seconds'])
  s=r['plus'];comp(parent+' + PC',s['n'],s['raw_correct'],s['tok'],s['seconds'])
 write('tables/comparison_rows.tex','\n'.join(comparison)+'\n')
 write('data/results_snapshot.json',json.dumps(out,indent=2,allow_nan=False)+'\n')
 cirows=[]
 for row in out['sets']:
  lo,hi=row['raw_pair']['conservative_95_ci_pp']
  cirows.append(f"{row['model']} & {row['dataset'].replace(' (original)','')} & {row['raw_pair']['delta_pp']:.2f} & [{lo:.2f}, {hi:.2f}]"+r' \\')
 write('tables/ci_rows.tex','\n'.join(cirows)+'\n')
 # TeX rows end in two literal backslashes.
 for name,lines in [('main_rows',table),('llama_modern_rows',llama),('raw_rows',raw),('diagnostic_rows',diag)]:
  write('tables/'+name+'.tex','% Generated by scripts/build_results.py\n'+'\n'.join(x+'\\' for x in lines)+'\n')
 print(f"Wrote {len(out['sets'])} paired settings, {len(out['combinations'])} combinations, and {len(out['baselines'])} PoLR settings to {HERE}")

if __name__=='__main__':main()
