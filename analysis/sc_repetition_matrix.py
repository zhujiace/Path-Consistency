"""Offline SC repetition analysis and 32 by 32 token-position matrices.

Analyze with the local transformers environment, render with system matplotlib.
No model inference, external requests, or changes to source results.
"""

import argparse
from collections import defaultdict
from decimal import Decimal
import glob
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.sc_prefix_similarity import DEFAULT_SC_GLOB, DEFAULT_TOKENIZER, branch_sort_key

OUT = ROOT / 'paper/ICLR/data/sc_repetition_matrix'
FIG = ROOT / 'paper/ICLR/figures/sc_repetition_matrix'


def postings(sequences, n):
    if n < 1:
        raise ValueError('n must be positive')
    index = defaultdict(list)
    for row, seq in enumerate(sequences):
        for start in range(len(seq) - n + 1):
            index[tuple(seq[start:start+n])].append((row, start))
    return index


def repetition_masks(sequences, n=8, prompt=()):
    """Union of exact n-gram matches in at least two DISTINCT paths."""
    index = postings(sequences, n)
    prompt_grams = set(postings([prompt], n))
    raw = [[False] * len(seq) for seq in sequences]
    novel = [[False] * len(seq) for seq in sequences]
    for gram, occurrences in index.items():
        if len({r for r, _ in occurrences}) < 2:
            continue
        for row, start in occurrences:
            raw[row][start:start+n] = [True] * n
            if gram not in prompt_grams:
                novel[row][start:start+n] = [True] * n
    return raw, novel


NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
SIGNED = r'[−-]?' + NUMBER
OP = r'(?:\\times|\\cdot|\\div|[+*/×÷−-])'
EXPR = re.compile(r'(?<![\w.])' + SIGNED + r'(?:\s*' + OP + r'\s*\$?\s*' + SIGNED + r')+')
TAIL = re.compile(r'\s*(' + OP + r')\s*\$?\s*(' + SIGNED + ')')


def extract_operations(text):
    """Conservative numeric expressions; source offsets retained, no algebraic equivalence."""
    found = []
    replacements = {'\\times': '*', '\\cdot': '*', '×': '*', '\\div': '/', '÷': '/', '−': '-'}
    for match in EXPR.finditer(text):
        def canonical_number(value):
            return format(Decimal(value.replace(',', '').replace('−', '-')).normalize(), 'f')
        first = re.match(SIGNED, match.group())
        parts = [canonical_number(first.group())]
        for operator, value in TAIL.findall(match.group()[first.end():]):
            parts.extend([replacements.get(operator, operator), canonical_number(value)])
        found.append({'key': ''.join(parts), 'start': match.start(), 'end': match.end()})
    return found


def assign_motifs(sequences, n=8, limit=8):
    """Greedily color exact common fragments, without overlapping assignments.

    Extend each shared n-gram rightward while ALL of its occurrences agree.
    Prefer maximum remaining covered tokens. A color always names one exact
    token sequence, never a transitive similarity cluster.
    """
    candidates = {}
    for gram, occurrences in postings(sequences, n).items():
        if len({r for r, _ in occurrences}) < 2:
            continue
        length = n
        while all(s + length < len(sequences[r]) for r, s in occurrences):
            next_tokens = {sequences[r][s + length] for r, s in occurrences}
            if len(next_tokens) != 1:
                break
            length += 1
        row, start = occurrences[0]
        key = tuple(sequences[row][start:start+length])
        candidates[key] = occurrences
    labels = [[0] * len(seq) for seq in sequences]
    motifs = []
    while candidates and len(motifs) < limit:
        options = []
        for tokens, occurrences in candidates.items():
            free = [(r, s) for r, s in occurrences if not any(labels[r][s:s+len(tokens)])]
            if len({r for r, _ in free}) >= 2:
                options.append((len(tokens) * len(free), len(tokens), tokens, free))
        if not options:
            break
        _, length, tokens, free = max(options, key=lambda x: (x[0], x[1], x[2]))
        del candidates[tokens]
        # Repeated occurrences within one path may overlap. Keep disjoint ones.
        accepted = []
        for row, start in free:
            if not any(labels[row][start:start+length]):
                labels[row][start:start+length] = [len(motifs) + 1] * length
                accepted.append({'path': row + 1, 'start': start, 'end': start + length})
        motifs.append({'tokens': list(tokens), 'occurrences': accepted})
    return motifs, labels


def bin_fractions(labels, bins=32):
    """Exact fractional token coverage of equal-width relative-position bins."""
    if not labels or bins < 1:
        raise ValueError('nonempty labels and positive bins required')
    width = len(labels) / bins
    result = []
    for b in range(bins):
        left, right = b * width, (b + 1) * width
        counts = defaultdict(float)
        for i in range(math.floor(left), min(len(labels), math.ceil(right))):
            counts[labels[i]] += max(0, min(right, i + 1) - max(left, i)) / width
        result.append(dict(counts))
    return result


def operation_labels(operations, offsets, prompt):
    prompt_ops = {o['key'] for o in extract_operations(prompt)}
    support = defaultdict(set)
    for row, ops in enumerate(operations):
        for op in ops:
            if op['key'] not in prompt_ops:
                support[op['key']].add(row)
    shared = {k: rows for k, rows in support.items() if len(rows) >= 2}
    keys = sorted(shared, key=lambda k: (-len(shared[k]), k))
    labels = [[0] * len(offs) for offs in offsets]
    for row, ops in enumerate(operations):
        for op in ops:
            if op['key'] not in shared:
                continue
            color = keys.index(op['key']) + 1
            for i, (start, end) in enumerate(offsets[row]):
                if start < op['end'] and end > op['start']:
                    labels[row][i] = color
    return keys, labels, {key: len(rows) for key, rows in shared.items()}


def prepare(row, tokenizer):
    traces = sorted(row['traces'], key=branch_sort_key)
    assert len(traces) == 32
    texts = []
    for trace in traces:
        text = trace['full_text']
        for special in tokenizer.all_special_tokens:
            text = text.replace(special, '')
        texts.append(text)
    encoded = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    assert all(encoded['input_ids'])
    return traces, texts, encoded['input_ids'], encoded['offset_mapping']


def scan(args):
    from transformers import AutoTokenizer
    start_time = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    files = sorted(glob.glob(args.sc_glob))
    if not files:
        raise ValueError('No input files')
    records, sources = [], []
    seen = set()
    for path in files:
        sources.append({'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                assert row['method'] == 'sc'
                source = int(row['source_row'])
                assert source not in seen
                seen.add(source)
                traces, texts, seqs, offsets = prepare(row, tokenizer)
                prompt = row['prompt']
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                total = sum(map(len, seqs))
                metrics = {}
                for n in (8, 16, 32):
                    raw, novel = repetition_masks(seqs, n, prompt_ids)
                    metrics[str(n)] = {
                        'all': sum(map(sum, raw)) / total,
                        'prompt_filtered': sum(map(sum, novel)) / total,
                    }
                ops = [extract_operations(text) for text in texts]
                keys, labels, support = operation_labels(ops, offsets, prompt)
                records.append({
                    'source_row': source, 'question_id': row['question_id'],
                    'tokens': total, 'truncated_paths': sum(bool(t['truncated']) for t in traces),
                    'exact_coverage': metrics,
                    'paths_with_extracted_operations': sum(bool(o) for o in ops),
                    'shared_operation_types': len(keys),
                    'paths_with_shared_operation': sum(any(x) for x in labels),
                    'repeated_operation_token_coverage': sum(sum(bool(x) for x in lab) for lab in labels) / total,
                    'cross_path_extra_operation_occurrences': sum(v - 1 for v in support.values()),
                    'max_operation_path_support': max(support.values(), default=0),
                })
                if len(records) % 100 == 0:
                    print(f'{len(records)} questions, {time.monotonic()-start_time:.1f}s', flush=True)
    assert len(records) == 1319, f'Expected complete GSM8K, got {len(records)}'
    records.sort(key=lambda r: r['source_row'])
    # Fixed descriptive selection rule, not hand-selected for a striking picture.
    ranked = sorted(records, key=lambda r: (r['exact_coverage']['8']['prompt_filtered'], r['source_row']))
    selected = {ranked[round(q*(len(ranked)-1))]['source_row']: name
                for q, name in [(0.25, 'q25'), (0.5, 'median'), (0.75, 'q75')]}
    examples = []
    for path in files:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if int(row['source_row']) not in selected:
                    continue
                traces, texts, seqs, offsets = prepare(row, tokenizer)
                motifs, labels = assign_motifs(seqs)
                for motif in motifs:
                    motif['text'] = tokenizer.decode(motif['tokens'], clean_up_tokenization_spaces=False)
                ops = [extract_operations(text) for text in texts]
                keys, op_labels, support = operation_labels(ops, offsets, row['prompt'])
                examples.append({
                    'selection': selected[int(row['source_row'])], 'source_row': int(row['source_row']),
                    'question_id': row['question_id'], 'prompt': row['prompt'],
                    'metrics': next(r for r in records if r['source_row'] == int(row['source_row'])),
                    'paths': [{'branch_id': t['branch_id'], 'text': text, 'token_ids': seq,
                               'offsets': off, 'operations': operations,
                               'truncated': bool(t['truncated'])}
                              for t, text, seq, off, operations in zip(traces, texts, seqs, offsets, ops)],
                    'motifs': motifs, 'token_labels': labels,
                    'cell_color_fractions': [bin_fractions(lab) for lab in labels],
                    'operation_keys': keys, 'operation_support': support,
                    'operation_labels': op_labels,
                })
    summary = {
        'questions': len(records), 'paths': 32 * len(records),
        'truncated_paths': sum(r['truncated_paths'] for r in records),
        'exact_coverage_macro': {n: {kind: fmean(r['exact_coverage'][n][kind] for r in records)
                                    for kind in ('all', 'prompt_filtered')} for n in ('8', '16', '32')},
        'questions_with_shared_operation': sum(r['shared_operation_types'] > 0 for r in records),
        'paths_with_operations': sum(r['paths_with_extracted_operations'] for r in records),
        'paths_with_shared_operation': sum(r['paths_with_shared_operation'] for r in records),
        'repeated_operation_token_coverage_macro': fmean(r['repeated_operation_token_coverage'] for r in records),
        'cross_path_extra_operation_occurrences': sum(r['cross_path_extra_operation_occurrences'] for r in records),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {'status': 'ANALYZED', 'tokenizer': args.tokenizer, 'sources': sources,
               'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'definition': 'Exact contiguous token n-grams shared by >=2 distinct paths; union coverage. Prompt-filtered excludes n-grams found verbatim in prompt, not semantic restatements.',
               'example_selection': 'Question ranks nearest 25/50/75 percentile of prompt-filtered exact-8 coverage, tie source_row. No correctness filter.',
               'duration_seconds': time.monotonic() - start_time, 'summary': summary, 'questions': records}
    (args.out / 'snapshot.json').write_text(json.dumps(payload, indent=2) + '\n')
    (args.out / 'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print('Selected', selected, flush=True)


def reference_labels(sequences, motifs, prefix_lengths):
    """Use the fixed SC motif IDs, then mask the actually reused prefix."""
    assert len(sequences) == len(prefix_lengths)
    result = []
    for seq, prefix_length in zip(sequences, prefix_lengths):
        assert 0 <= prefix_length <= len(seq)
        labels = [0] * len(seq)
        for color, motif in enumerate(motifs, 1):
            tokens = motif['tokens']
            for start in range(len(seq) - len(tokens) + 1):
                if seq[start:start+len(tokens)] == tokens:
                    for i in range(start, start+len(tokens)):
                        if labels[i] == 0:
                            labels[i] = color
        labels[:prefix_length] = [-1] * prefix_length
        result.append(labels)
    return result


def prepare_pc(args):
    from transformers import AutoTokenizer
    sc = next(e for e in json.loads((args.out / 'examples.json').read_text()) if e['selection'] == 'q75')
    candidates = []
    for path in sorted(glob.glob(args.sc_glob.replace('_sc.jsonl', '_pc.jsonl'))):
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if int(row['source_row']) == sc['source_row']:
                    candidates.append((path, row))
    assert len(candidates) == 1
    path, row = candidates[0]
    assert row['method'] == 'pc' and row['question_id'] == sc['question_id']
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    traces, texts, sequences, offsets = prepare(row, tokenizer)
    windows = {int(w['window']): w for w in row['windows']}
    prefix_lengths, paths = [], []
    for index, (trace, text, seq, off) in enumerate(zip(traces, texts, sequences, offsets)):
        window = windows[int(trace['window'])]
        prefix_ids = window['used_prefix_token_ids']
        assert len(prefix_ids) == trace['prefix_tokens'] == window['used_prefix_tokens']
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert trace['full_text'] == prefix_text + trace['suffix_text']
        assert text.startswith(prefix_text)
        # Map the real prefix boundary to visible, retokenized full-text positions.
        count = sum(start < len(prefix_text) and end > start for start, end in off)
        prefix_lengths.append(count)
        if int(trace['window']) == 0:
            assert seq == sc['paths'][index]['token_ids']
        paths.append({'branch_id': trace['branch_id'], 'window': int(trace['window']),
                      'text': text, 'token_ids': seq, 'offsets': off,
                      'prefix_token_ids': prefix_ids, 'prefix_char_end': len(prefix_text),
                      'visible_prefix_tokens': count, 'generated_tokens': trace['generated_tokens'],
                      'reused_sc_root': trace['reused_sc_root']})
    labels = reference_labels(sequences, sc['motifs'], prefix_lengths)
    result = {'selection': 'q75_pc', 'method': 'pc', 'question_id': row['question_id'],
              'source_row': row['source_row'], 'source_path': path,
              'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'reference_examples_sha256': hashlib.sha256((args.out / 'examples.json').read_bytes()).hexdigest(),
              'config_fingerprint': row['config_fingerprint'], 'paths': paths,
              'motifs': sc['motifs'], 'token_labels': labels,
              'cell_color_fractions': [bin_fractions(lab) for lab in labels],
              'windows': [{'window': w['window'], 'prefix_tokens': w['used_prefix_tokens'],
                           'used_ratio': w['used_ratio'], 'decision': w['decision']} for w in row['windows']]}
    (args.out / 'q75_pc_example.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('PC example saved; actual prefix lengths:', [w['prefix_tokens'] for w in result['windows']], flush=True)


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import textwrap

    palette = ['#eceff3', '#4e79a7', '#f28e2b', '#59a14f', '#b07aa1', '#e15759', '#76b7b2', '#edc949', '#af7a55']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    args.fig.mkdir(parents=True, exist_ok=True)
    examples = json.loads((args.out / 'examples.json').read_text())
    if (args.out / 'q75_pc_example.json').exists():
        examples.append(json.loads((args.out / 'q75_pc_example.json').read_text()))
    for example in examples:
        pc = example.get('method') == 'pc'
        for mode in (('tokens',) if pc else ('tokens', 'operations')):
            figure_name = f"{example['selection']}_{mode}"
            if args.only and figure_name != args.only:
                continue
            full_cells = figure_name in ('q75_tokens', 'q75_pc_tokens')
            labs = example['token_labels'] if mode == 'tokens' else example['operation_labels']
            fig = plt.figure(figsize=(12, 8.6), facecolor='white')
            ax = fig.add_axes([.075, .16, .57, .75])
            legend = fig.add_axes([.68, .16, .30, .75])
            legend.axis('off')
            for row, labels in enumerate(labs):
                if full_cells:
                    for column, fractions in enumerate(bin_fractions(labels)):
                        colored = {color: fraction for color, fraction in fractions.items()
                                   if 1 <= color <= 8 and fraction > 0}
                        color = max(colored, key=lambda c: (colored[c], -c)) if colored else 0
                        fill = '#a6adb7' if fractions.get(-1, 0) > 0 else palette[color]
                        ax.add_patch(Rectangle((column, row), 1, 1,
                                               facecolor=fill, edgecolor='none'))
                    continue
                length = len(labels)
                start = 0
                for end in range(1, length + 1):
                    if end == length or labels[end] != labels[start]:
                        color = labels[start] if labels[start] <= 8 else 0
                        ax.add_patch(Rectangle((32*start/length, row), 32*(end-start)/length, 1,
                                               facecolor=palette[color], edgecolor='none'))
                        start = end
            ax.set(xlim=(0, 32), ylim=(32, 0), xlabel='Relative token position along each complete path', ylabel='PC reasoning path' if pc else 'SC reasoning path')
            ax.set_xticks([0, 8, 16, 24, 32], ['0%', '25%', '50%', '75%', '100%'])
            ax.set_yticks([r+.5 for r in range(32)], [str(r+1) for r in range(32)], fontsize=8)
            ax.vlines(range(33), 0, 32, colors='white', linewidth=.6)
            ax.hlines(range(33), 0, 32, colors='white', linewidth=.6)
            ax.tick_params(length=0, pad=6)
            for spine in ax.spines.values():
                spine.set_visible(False)
            title = 'Repeated token fragments' if mode == 'tokens' else 'Repeated arithmetic expressions'
            if pc:
                title = 'Prefix reuse with Path-consistency'
            fig.text(.075, .96, title, fontsize=17, fontweight='bold')
            method_note = '32 PC paths · C = 0.8' if pc else '32 independent SC paths'
            fig.text(.075, .925, f"Llama-3.1-8B-Instruct  ·  GSM8K  ·  {example['question_id']}  ·  {method_note}", fontsize=10, color='#596273')
            legend.text(0, 1, 'Same color = same fragment' if mode == 'tokens' else 'Same color = same expression', va='top', fontweight='bold')
            y = .94
            if mode == 'tokens':
                entries = [(re.sub(r'\s+', ' ', m['text']).strip(),
                            (f"{len(m['tokens'])} tokens · same fragment as SC" if pc else
                             f"{len(m['tokens'])} tokens · {len({o['path'] for o in m['occurrences']})}/32 paths"))
                           for m in example['motifs']]
            else:
                entries = [(key, f"{example['operation_support'][key]}/32 paths") for key in example['operation_keys'][:8]]
            for i, (label, detail) in enumerate(entries):
                # Exact full snippets and offsets are retained in examples.json.
                lines = textwrap.wrap(label, width=38)
                if len(lines) > 3:
                    lines = lines[:3]
                    lines[-1] = lines[-1][:34] + '…'
                legend.add_patch(Rectangle((0, y-.023), .035, .023, color=palette[i+1], transform=legend.transAxes))
                legend.text(.055, y, '\n'.join(lines), va='top', fontsize=9, linespacing=1.25)
                y -= .025 * len(lines) + .009
                legend.text(.055, y, detail, va='top', fontsize=8, color='#677181')
                y -= .043
            if pc:
                legend.add_patch(Rectangle((0, .062), .035, .023, color='#a6adb7', transform=legend.transAxes))
                legend.text(.055, .087, 'Reused prefix', va='top', fontsize=9)
                legend.text(0, .015, 'Light gray = other generated content\nFragment colors match the SC figure.', fontsize=8, color='#677181', va='bottom')
            else:
                legend.text(0, .015, 'Gray = outside displayed fragments\nColors apply only within this example.', fontsize=8, color='#677181', va='bottom')
            count = [len(p['token_ids']) for p in example['paths']]
            fig.text(.075, .095, f"32 × 32 grid · each column spans 1/32 of a path · lengths {min(count)}–{max(count)} tokens", fontsize=10)
            cell_note = ('Any highlighted overlap fills the cell; multiple colors use the largest overlap. Filled area is not exact token coverage.'
                         if full_cells else
                         'Subcell color boundaries preserve token positions. Equal horizontal widths need not contain equal token counts across paths.')
            if pc:
                cell_note = 'Reused prefix takes priority in boundary cells. Other highlights fill whole cells using the same rule as SC.'
            fig.text(.075, .065, cell_note, fontsize=9, color='#596273')
            if mode == 'tokens':
                note = 'Exact matches of at least 8 tokens; up to 8 disjoint fragment types shown. Wording overlap alone does not establish reusable computation.'
            else:
                note = 'Numeric expressions only; formatting normalized, operand order preserved. Expressions present in the prompt are excluded.'
            if pc:
                note = 'Rows follow the 8 windows (4 paths each). Root paths are counted as generation. Gray boundaries are rounded to full cells.'
            fig.text(.075, .035, note, fontsize=8.5, color='#596273')
            stem = args.fig / f"{example['selection']}_{mode}"
            for ext in ('png', 'svg', 'pdf'):
                fig.savefig(str(stem) + '.' + ext, dpi=180)
            plt.close(fig)
            print(stem, flush=True)


def render_comparison(args):
    """Paired display using the existing SC motifs and PC prefix masks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.legend_handler import HandlerTuple

    sc = next(e for e in json.loads((args.out / 'examples.json').read_text()) if e['selection'] == 'q75')
    pc = json.loads((args.out / 'q75_pc_example.json').read_text())
    assert sc['question_id'] == pc['question_id']
    assert sc['motifs'] == pc['motifs']
    # Distinct blue-to-yellow fragment colors; purple reserved for prefix reuse.
    palette = ['#e6e6e6', '#174a8b', '#298fd6', '#6cd4e8', '#007d78', '#32b65a', '#a4d53d', '#d4ab00', '#ffe53b']
    prefix_color = '#440154'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(12, 7.2), facecolor='white')
    for x, example, title in ((.055, sc, '(a) SC'), (.555, pc, '(b) PC')):
        ax = fig.add_axes([x, .235, .42, .70])
        for row, labels in enumerate(example['token_labels']):
            for column, fractions in enumerate(bin_fractions(labels)):
                colored = {c: f for c, f in fractions.items() if 1 <= c <= 8 and f > 0}
                color = max(colored, key=lambda c: (colored[c], -c)) if colored else 0
                fill = prefix_color if fractions.get(-1, 0) > 0 else palette[color]
                ax.add_patch(Rectangle((column, row), 1, 1, facecolor=fill,
                                       edgecolor='#ffffff', linewidth=.45))
        ax.set(xlim=(0, 32), ylim=(32, 0), aspect='equal', title=title)
        ax.title.set_fontsize(15)
        ax.title.set_fontweight('bold')
        ax.set_xticks([0, 8, 16, 24, 32], ['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([r+.5 for r in range(32)], [str(r+1) for r in range(32)], fontsize=7)
        ax.set_ylabel('Reasoning path', fontsize=11)
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.515, .17, 'Relative token position along each complete path', ha='center', fontsize=11)
    handles = [tuple(Patch(facecolor=palette[i], edgecolor='none') for i in (1, 3, 5, 8)),
               Patch(facecolor=palette[0], edgecolor='none'),
               Patch(facecolor=prefix_color, edgecolor='none')]
    fig.legend(handles, ['Same token sequences', 'Different token sequences', 'Shared prefix'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc='lower center', bbox_to_anchor=(.515, .082), ncol=3,
               frameon=False, fontsize=11, handlelength=2.2, columnspacing=2.8)
    fig.text(.515, .043, 'Light gray denotes content outside the highlighted reference sequences.',
             ha='center', fontsize=8.5, color='#677181')
    args.fig.mkdir(parents=True, exist_ok=True)
    stem = args.fig / 'q75_sc_pc_comparison'
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(str(stem) + '.' + ext, dpi=200)
    plt.close(fig)
    print(stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['scan', 'render', 'prepare-pc', 'compare'])
    parser.add_argument('--sc-glob', default=DEFAULT_SC_GLOB)
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--fig', type=Path, default=FIG)
    parser.add_argument('--only', help='Render only this figure, e.g. q75_tokens')
    args = parser.parse_args()
    {'scan': scan, 'render': render, 'prepare-pc': prepare_pc, 'compare': render_comparison}[args.mode](args)
