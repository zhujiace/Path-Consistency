## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-18
- Verification Status: ANALYZED
- Version Label: sc_prefix_similarity_validation_v1

## Scope

This report validates the descriptive prefix-similarity analysis for Llama-3.1-8B-Instruct on the GSM8K test set. It does not validate a causal effect of prefix reuse or certify that similar prefixes are correct.

## Data and estimator

- 1,319 questions, 32 independent SC trajectories per question, and 42,208 trajectories in total.
- No trajectory is missing its stored reasoning text or parsed answer. There are 281 truncated trajectories, or 0.67% of the pool.
- Each trajectory is cut at 20%, 40%, 60%, and 80% of its own Llama-tokenized length.
- Similarity is cosine similarity between normalized word-bigram count vectors. Numbers and arithmetic operators are retained.
- The primary estimator averages all 496 unordered trajectory pairs within each question, then macro-averages questions.
- The baseline pairs the same branch index across a seeded random cycle of different questions.
- Intervals are percentile intervals from 5,000 question-cluster bootstrap replicates. They quantify variation across GSM8K questions, not variation across model seeds.

## Findings

| Relative position | Within question | Random cross-question baseline | Paired difference |
|---:|---:|---:|---:|
| 20% | 0.618 [0.611, 0.625] | 0.134 [0.130, 0.138] | 0.484 [0.477, 0.490] |
| 40% | 0.630 [0.624, 0.636] | 0.095 [0.092, 0.098] | 0.536 [0.530, 0.542] |
| 60% | 0.638 [0.633, 0.644] | 0.085 [0.082, 0.088] | 0.553 [0.547, 0.559] |
| 80% | 0.652 [0.646, 0.657] | 0.089 [0.086, 0.093] | 0.562 [0.556, 0.569] |

Within-question similarity exceeds the paired cross-question baseline for every question at all four positions. The within-question median rises from 0.619 at 20% to 0.659 at 80%. The 10th to 90th percentile range is 0.450 to 0.780 at 20% and 0.510 to 0.791 at 80%, so overlap is heterogeneous rather than uniform.

Among the 674 questions containing at least one pair of different final answers, the different-answer pair mean is 0.566 at 20% and 0.589 at 80%. Thus the measured overlap is not limited to trajectories with identical final answers. This is a conditional diagnostic because questions without answer disagreement do not enter that subgroup.

Excluding truncated trajectories changes the primary means by only +0.0011, +0.0014, +0.0017, and +0.0019 at the four positions. The result is therefore insensitive to the 281 truncations under this estimator.

## Interpretation boundary

The experiment supports the motivation that independently sampled SC trajectories contain substantial problem-specific repetition throughout their reasoning. It does not show that repeated text represents an interchangeable hidden state, that the shared material is correct, or that reusing it preserves final voting accuracy. Word-bigram cosine measures surface overlap and can reflect common problem restatements, arithmetic expressions, and solution templates.

## Statistical fallacy scan

Coverage: 11/11 checked.

1. Simpson's paradox: no reversal is observed in the answer-agreement diagnostic, but difficulty and other latent strata were not exhaustively tested.
2. Ecological fallacy: the question-macro mean must not be interpreted as applying to every trajectory pair. Quantiles are reported to expose heterogeneity.
3. Berkson's paradox: the analysis uses the full fixed test set and all stored SC trajectories. Generalization beyond this model and dataset remains unresolved.
4. Collider bias: no post-treatment covariate is controlled, so this pattern is not implicated.
5. Base-rate neglect: no diagnostic-classification metric is interpreted.
6. Regression to the mean: there is no extreme-group pre-post comparison.
7. Survivorship bias: no stored trajectory is dropped from the primary estimator, and truncation exclusion is reported separately.
8. Look-elsewhere effect: all four author-specified positions are reported. No position is selected after seeing the results.
9. Garden of forking paths: the similarity rule was fixed before the full scan. This remains an exploratory, non-preregistered analysis and should be labeled accordingly.
10. Correlation versus causation: the observational overlap cannot establish that sharing prefixes causes token savings or preserves accuracy.
11. Reverse causality: no directional causal relation is estimated.

Overall confidence: CAUTION. The dataset-level overlap estimate is precise and robust to truncation, but evidence is limited to one model, one benchmark, one sampling run, and one surface-form similarity criterion.
