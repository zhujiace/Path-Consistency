"""从真实新协议日志生成黑色前缀、蓝色续写的案例表。"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/data/26xyw/path-consistency-results/gsm8k/gsm8k_llama31_b32_mt2048_c08_seed0_shard2of4_203bd205a5ef_pc.jsonl')
QUESTION = 'gsm8k_1162'


def tex(text):
    # 仅处理日志中的显示转义和结束标记，不改写原句或计算。
    text = text.replace('<|eot_id|>', '').replace(r'\n', ' ').replace('\n', ' ')
    text = text.replace(r'\\boxed', r'\boxed')
    escapes = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '\\': r'\textbackslash{}',
               '^': r'\textasciicircum{}', '~': r'\textasciitilde{}'}
    parts = re.split(r'(\\boxed\{[^{}]*\})', text)
    return ''.join('$'+p+'$' if p.startswith(r'\boxed{') else
                   ''.join(escapes.get(c, c) for c in p) for p in parts)


def main():
    record = next(r for line in SOURCE.open() if (r := json.loads(line))['question_id'] == QUESTION)
    windows = record['windows']
    assert [w['used_ratio'] for w in windows[:5]] == [0, .2, .4, .6, .8]
    by_id = {t['branch_id']: t for t in record['traces']}
    traces = [by_id[w['proposed_prefix']['source_id']] for w in windows[:4]]
    traces.append(next(t for t in record['traces'] if t['window'] == 4))
    question = record['prompt'].split('\n\nPlease reason step by step')[0]
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Progressive prefix reuse on a GSM8K example with Llama-3.1-8B-Instruct. Black text is the reused prefix and blue text is newly generated. One recorded trajectory is shown per window.}',
             r'\label{tab:prefix_case}', r'\begin{tabularx}{\linewidth}{@{}X@{}}',
             r'\toprule', r'\textbf{Question.} '+tex(question)+r' \\', r'\midrule']
    stages = []
    for i, trace in enumerate(traces):
        suffix = trace['suffix_text']
        assert trace['full_text'].endswith(suffix) and suffix
        prefix = trace['full_text'][:-len(suffix)]
        assert trace['window'] == i and not trace['truncated']
        if i:
            assert traces[i-1]['full_text'].startswith(prefix)
            assert trace['prefix_tokens'] == windows[i]['used_prefix_tokens']
        heading = (rf'\textbf{{Window {i+1}, prefix level {20*i}\%}}'
                   +rf' \hfill {trace["prefix_tokens"]} prefix tokens; {trace["generated_tokens"]} generated tokens')
        boundary = r'\ ' if prefix and suffix[0].isspace() else ''
        lines.extend([heading+r' \\', tex(prefix)+boundary+r'{\color{blue}'+tex(suffix)+r'} \\'])
        lines.append(r'\midrule' if i < 4 else r'\bottomrule')
        stages.append(dict(window=i+1, branch_id=trace['branch_id'], prefix_level=.2*i,
                           prefix_tokens=trace['prefix_tokens'], generated_tokens=trace['generated_tokens'],
                           prefix=prefix, suffix=suffix))
    lines.extend([r'\end{tabularx}', r'\end{table}'])
    (ROOT/'tables/prefix_case.tex').write_text('\n'.join(lines)+'\n')
    audit = dict(question_id=QUESTION, question=question, gold=record['gold'],
                 source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 selection='从正确且前五窗达到0/20/40/60/80%级别的1016题中，按展示轨迹总字符数选择最短者，以便完整排版。前四窗展示下一前缀的实际来源轨迹，第五窗展示首条轨迹。此例说明机制，不用于推断总体效果。',
                 stages=stages)
    (ROOT/'data/prefix_case.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(QUESTION, [(s['prefix_tokens'], s['generated_tokens']) for s in stages])


if __name__ == '__main__':
    main()
