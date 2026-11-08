"""Path of Least Resistance (PoLR; arXiv 2601.21494) without sklearn.

Appendix A/B: sample N prefixes of length L_p, TF-IDF bag-of-words, downsample
to 10 dimensions, agglomerative clustering (average linkage, cosine,
distance_threshold=1.0), expand every prefix in the largest cluster C*, then
majority-vote those completions.

This module is engine-agnostic. Token IDs stay on the vLLM path so prefixes
are continued, not regenerated.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .math_eval import grade_answer, majority_vote

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DEFAULT_DISTANCE_THRESHOLD = 1.0
DEFAULT_FEATURE_DIM = 10


@dataclass(frozen=True)
class PrefixClustering:
    labels: tuple[int, ...]
    dominant_label: int
    dominant_indices: tuple[int, ...]
    cluster_sizes: tuple[tuple[int, int], ...]
    n_clusters: int
    kappa: float

    def as_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "dominant_label": self.dominant_label,
            "dominant_indices": list(self.dominant_indices),
            "cluster_sizes": [list(item) for item in self.cluster_sizes],
            "n_clusters": self.n_clusters,
            "kappa": self.kappa,
        }


def tokenize_prefix(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def tfidf_matrix(texts: Sequence[str]) -> np.ndarray:
    """Sparse-free TF-IDF; rows are documents, columns are vocabulary terms."""
    tokenized = [tokenize_prefix(text) for text in texts]
    n_docs = len(tokenized)
    if n_docs == 0:
        return np.zeros((0, 0), dtype=np.float64)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    vocab = sorted(document_frequency)
    if not vocab:
        return np.zeros((n_docs, 1), dtype=np.float64)
    index = {term: column for column, term in enumerate(vocab)}
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    idf = np.log((1.0 + n_docs) / (1.0 + np.array(
        [document_frequency[term] for term in vocab], dtype=np.float64
    ))) + 1.0
    for row, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for term, count in counts.items():
            matrix[row, index[term]] = count * idf[index[term]]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    nonzero = norms.squeeze(-1) > 1e-12
    matrix[nonzero] = matrix[nonzero] / norms[nonzero]
    return matrix


def downsample_features(matrix: np.ndarray, n_components: int = DEFAULT_FEATURE_DIM) -> np.ndarray:
    """Deterministic truncated SVD, matching the paper's dim=10 downsampling."""
    if matrix.size == 0:
        return matrix
    n_docs, n_features = matrix.shape
    rank = min(n_components, n_docs, n_features)
    if rank <= 0:
        return matrix
    if n_docs == 1 or n_features == 1 or rank == n_features:
        return matrix[:, :rank] if rank < n_features else matrix
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(rank, singular.shape[0])
    if keep <= 0:
        return matrix
    return u[:, :keep] * singular[:keep]


def _cosine_condensed(features: np.ndarray) -> np.ndarray:
    n_docs = features.shape[0]
    if n_docs < 2:
        return np.zeros(0, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    zero = norms.squeeze(-1) <= 1e-12
    safe = np.where(zero[:, None], 1.0, norms)
    normalized = features / safe
    condensed = pdist(normalized, metric="cosine")
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=1.0)
    if not zero.any():
        return condensed
    # Pair (i, j) in condensed pdist order: i < j, row-major.
    pair = 0
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            if zero[i] and zero[j]:
                condensed[pair] = 0.0
            pair += 1
    return np.clip(condensed, 0.0, 2.0)


def cluster_prefixes(
    texts: Sequence[str],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    n_components: int = DEFAULT_FEATURE_DIM,
) -> PrefixClustering:
    n_docs = len(texts)
    if n_docs == 0:
        return PrefixClustering((), 0, (), (), 0, 0.0)
    if n_docs == 1:
        return PrefixClustering((1,), 1, (0,), ((1, 1),), 1, 1.0)

    features = downsample_features(tfidf_matrix(texts), n_components=n_components)
    condensed = _cosine_condensed(features)
    if condensed.size == 0 or float(np.max(condensed)) <= 1e-12:
        labels = np.ones(n_docs, dtype=int)
    else:
        z_matrix = linkage(condensed, method="average")
        labels = fcluster(z_matrix, t=distance_threshold, criterion="distance")
    counts = Counter(int(label) for label in labels)
    # Ties: largest cluster, then lowest label (stable).
    dominant_label = min(
        counts,
        key=lambda label: (-counts[label], label),
    )
    dominant_indices = tuple(
        index for index, label in enumerate(labels) if int(label) == dominant_label
    )
    cluster_sizes = tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return PrefixClustering(
        labels=tuple(int(label) for label in labels),
        dominant_label=int(dominant_label),
        dominant_indices=dominant_indices,
        cluster_sizes=cluster_sizes,
        n_clusters=len(counts),
        kappa=len(dominant_indices) / n_docs,
    )


def prefix_is_complete(trace: dict, prefix_len: int) -> bool:
    if trace.get("finish_reason") == "stop":
        return True
    if trace.get("truncated") or trace.get("finish_reason") == "length":
        return False
    generated = int(trace.get("generated_tokens") or 0)
    return generated < prefix_len


def compact_trace(trace: dict, store_text: bool = False) -> dict:
    compact = {
        "answer": trace.get("answer"),
        "raw_answer": trace.get("raw_answer"),
        "generated_tokens": trace.get("generated_tokens"),
        "prefix_tokens": trace.get("prefix_tokens"),
        "finish_reason": trace.get("finish_reason"),
        "truncated": bool(trace.get("truncated")),
        "cluster_id": trace.get("cluster_id"),
        "expanded": bool(trace.get("expanded")),
        "in_dominant": bool(trace.get("in_dominant")),
    }
    if store_text:
        compact["full_text"] = trace.get("full_text")
    return compact


def assemble_polr_result(
    prefix_traces: Sequence[dict],
    expanded_traces: Sequence[dict],
    clustering: PrefixClustering,
    prefix_seconds: float,
    expand_seconds: float,
    prefix_len: int,
    gold: Optional[str],
    task: str,
    store_text: bool = False,
) -> dict:
    """Combine prefix sampling + C* expansion into one question record."""
    expanded_by_index = {trace["prefix_index"]: trace for trace in expanded_traces}
    voted_traces = []
    kept = []
    prefix_tokens = 0
    expand_tokens = 0
    for index, prefix in enumerate(prefix_traces):
        prefix_tokens += int(prefix.get("generated_tokens") or 0)
        in_dominant = index in clustering.dominant_indices
        row = dict(prefix)
        row["cluster_id"] = clustering.labels[index]
        row["in_dominant"] = in_dominant
        row["expanded"] = False
        row["prefix_index"] = index
        if in_dominant and index in expanded_by_index:
            continued = dict(expanded_by_index[index])
            expand_tokens += int(continued.get("generated_tokens") or 0)
            continued["cluster_id"] = clustering.labels[index]
            continued["in_dominant"] = True
            continued["expanded"] = True
            continued["prefix_index"] = index
            continued["prefix_generated_tokens"] = prefix.get("generated_tokens")
            voted_traces.append(continued)
            kept.append(compact_trace(continued, store_text=store_text))
        elif in_dominant:
            voted_traces.append(row)
            kept.append(compact_trace(row, store_text=store_text))
        else:
            kept.append(compact_trace(row, store_text=store_text))

    voted = majority_vote(trace.get("answer") for trace in voted_traces)
    total_generated = prefix_tokens + expand_tokens
    n_prefixes = len(prefix_traces)
    n_expanded = len(expanded_traces)
    k_star = len(clustering.dominant_indices)
    return {
        "answer": voted,
        "correct": grade_answer(task, voted, gold),
        "gold": gold,
        "n_prefixes": n_prefixes,
        "n_clusters": clustering.n_clusters,
        "dominant_size": k_star,
        "n_expanded": n_expanded,
        "kappa": clustering.kappa,
        "cluster_sizes": [size for _label, size in clustering.cluster_sizes],
        "prefix_generated_tokens": prefix_tokens,
        "expand_generated_tokens": expand_tokens,
        "total_generated_tokens": total_generated,
        "prefix_seconds": prefix_seconds,
        "expand_seconds": expand_seconds,
        "generation_seconds": prefix_seconds + expand_seconds,
        "formula_tokens": prefix_tokens + expand_tokens,
        "prefix_len": prefix_len,
        "clustering": clustering.as_dict(),
        "traces": kept,
        "voted_answers": [trace.get("answer") for trace in voted_traces],
    }


def indices_to_expand(
    prefix_traces: Sequence[dict],
    clustering: PrefixClustering,
    prefix_len: int,
) -> list[int]:
    return [
        index
        for index in clustering.dominant_indices
        if not prefix_is_complete(prefix_traces[index], prefix_len)
    ]


def summarize_polr_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    n_questions = len(rows)
    if n_questions == 0:
        return {
            "questions": 0,
            "correct": 0,
            "accuracy": None,
            "avg_generated_tokens": None,
            "avg_generation_seconds": None,
            "avg_kappa": None,
            "avg_dominant_size": None,
        }
    correct = sum(1 for row in rows if row.get("correct"))
    tokens = [row.get("total_generated_tokens") for row in rows]
    seconds = [row.get("generation_seconds") for row in rows]
    kappas = [row.get("kappa") for row in rows]
    sizes = [row.get("dominant_size") for row in rows]
    return {
        "questions": n_questions,
        "correct": correct,
        "accuracy": correct / n_questions,
        "total_generated_tokens": sum(value or 0 for value in tokens),
        "avg_generated_tokens": sum(value or 0 for value in tokens) / n_questions,
        "total_generation_seconds": sum(value or 0.0 for value in seconds),
        "avg_generation_seconds": sum(value or 0.0 for value in seconds) / n_questions,
        "avg_kappa": sum(value or 0.0 for value in kappas) / n_questions,
        "avg_dominant_size": sum(value or 0 for value in sizes) / n_questions,
        "truncations": sum(1 for row in rows if any(
            (trace or {}).get("truncated") for trace in (row.get("traces") or [])
            if (trace or {}).get("in_dominant")
        )),
    }
