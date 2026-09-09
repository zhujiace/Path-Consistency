"""Top-1-vs-all-rest Beta confidence for prefix gating.

For ``v`` votes for the modal answer among ``n`` valid answers, the gate uses
``P(theta > 0.5)`` under ``theta ~ Beta(v + 1, n - v + 1)``.  This measures
whether the modal answer dominates all alternatives together; it is not a
posterior probability that the answer itself is correct.
"""

from collections import Counter

from scipy.stats import beta


def beta_confidence(answers: list) -> float:
    valid = [answer for answer in answers if answer is not None]
    if not valid:
        return 0.0
    votes = Counter(valid)
    modal_votes = votes.most_common(1)[0][1]
    alternatives = len(valid) - modal_votes
    return float(beta.sf(0.5, modal_votes + 1, alternatives + 1))


def confidence_criteria(answers: list, conf_thresh: float) -> dict:
    valid = [answer for answer in answers if answer is not None]
    most_common = Counter(valid).most_common(1)[0][0] if valid else None
    prob = beta_confidence(answers)
    return {
        "most_common": most_common,
        "prob": prob,
        "conf": prob >= conf_thresh,
    }
