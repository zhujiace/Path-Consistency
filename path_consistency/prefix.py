"""Prefix extraction / sampling, ported from the paper's appendix code.

get_prefix:
  - evaluates the beta confidence over all answers generated so far;
  - if the gate passes, extracts the first (prefix_level + 1) sentences from
    each majority-answer reasoning path of the last window (frq branches)
    and REPLACES the prefix list with these new, longer prefixes;
  - if the gate fails, discards the current prefix list and drops back to
    level 0 (this is the "prefix correction" behaviour described in the
    paper's appendix case study).
"""

import random
import re

from .confidence import confidence_criteria

SENT_SPLIT_RE = re.compile(r"(\.\s|\.\n|\n)")


def split_sentences(text: str) -> list:
    parts = SENT_SPLIT_RE.split(text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + parts[i + 1])
    return sentences


def get_prefix(answers, conf_thresh, gens, prefix_list, prefix_level, frq: int = 5):
    """Return (new_prefix_list, new_prefix_level, extracted_level).

    extracted_level is the level of the newly extracted prefixes (0 when the
    gate failed, so that downstream code can tell branches apart).
    """
    confidence = confidence_criteria(answers, conf_thresh)
    if not confidence["conf"]:
        return [], 0, 0

    new_list = []
    for i, gen in enumerate(gens[-frq:]):
        index = len(gens) - frq + i
        parts = gen.split("answer is")
        if len(parts) > 1 and answers[index] == confidence["most_common"]:
            sentences = split_sentences(parts[0])
            sentence = ""
            for j in range(prefix_level + 1):
                if j < len(sentences) - 1:
                    sentence += sentences[j]
            if sentence:
                new_list.append(sentence)

    if not new_list:
        return [], 0, 0
    return new_list, prefix_level + 1, prefix_level + 1


def sample_prefix(prefix_list: list) -> str:
    if len(prefix_list) > 0:
        return random.choice(prefix_list)
    return ""
