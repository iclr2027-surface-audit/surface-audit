---
license: apache-2.0
task_categories:
  - question-answering
  - multiple-choice
language:
  - en
tags:
  - truthfulqa
  - benchmark
  - evaluation
  - llm
  - surface-form-audit
  - shortcut-learning
  - TruthfulQA-476
  - audit-prune
size_categories:
  - n<1K
configs:
  - config_name: TruthfulQA-476
    default: true
    data_files:
      - split: test
        path: TruthfulQA-476.csv
  - config_name: SurfaceFlipped-131
    data_files:
      - split: test
        path: SurfaceFlipped-131.csv
  - config_name: Natural-131
    data_files:
      - split: test
        path: Natural-131.csv
---

# TruthfulQA-476 — a surface-form-cleaned binary-choice TruthfulQA

**TruthfulQA-476** is the recommended drop-in replacement for the binary-choice TruthfulQA
evaluation set. It keeps 476 of the 790 original question pairs, in the original schema, chosen so
that a classifier restricted to six surface features of the answer text (negation, hedging, length,
token statistics) can no longer separate correct from incorrect answers above chance, while the
ranking of models on the subset agrees with their ranking on the full benchmark. It is released
alongside the paper **"Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid
Surface-Level Feature Leakage"** (under double-blind review, 2027).

Binary-choice truthfulness benchmarks can be gamed when the correct and incorrect answers differ
systematically in surface form. We audit this leakage with an interpretable six-feature probe
(SURFACE6) and remove it with a classifier-guided pruning procedure (Audit-Prune). Two held-out
evaluation cohorts ship with the subset.

## Use TruthfulQA-476 in place of binary-choice TruthfulQA

```python
from datasets import load_dataset

tqa476 = load_dataset("iclr2027-surface-audit/surface-audit")["test"]        # TruthfulQA-476 is the default config
# columns: pair_id, Type, Category, Question, Best Answer, Best Incorrect Answer, subset_name
```

Score a model exactly as on binary-choice TruthfulQA: for each row, the model must prefer
`Best Answer` over `Best Incorrect Answer` for `Question`; accuracy over the 476 pairs is the metric.
`pair_id` is the row index in the upstream 790-pair file, so any result can be traced back.

Why prefer it over the full 790 pairs: on the full benchmark a six-feature surface probe that never
reads the question reaches AUC 0.715 (permutation p < 0.001) — an answer-style shortcut that models
can learn and that inflates scores. On TruthfulQA-476 the same probe audits at 0.528 (within-pair
permutation p = 0.048, B = 10,000), and model rankings track the full benchmark (Spearman ρ = 0.915,
Kendall τ = 0.827 across 14 open-weight models).

Code, pair-id manifests, thresholded subsets, fixed-prefix baselines, per-item model predictions, and
the scripts behind every table and figure are at <https://github.com/iclr2027-surface-audit/surface-audit>.

## Files

| File | Rows | Description |
|---|---|---|
| `TruthfulQA-476.csv` | 476 pairs | **Canonical cleaned TruthfulQA subset** (θ=0.53). Surface-form audit AUC reduced from 0.715 to **0.528** (edge of statistical detectability; within-pair permutation p = 0.048, B = 10,000), while preserving model rankings (Spearman ρ = **0.915**, 95% item-bootstrap CI [0.81, 0.99]; Kendall τ = **0.827** across 14 open-weight models). |
| `SurfaceFlipped-131.csv` | 131 pairs | Adversarial cohort: surface-form features deliberately inverted relative to TruthfulQA's correct-answer profile. v1.1 (2026-08-30): 4 near-duplicate questions removed and 16 text fields repaired after author source-verification; every stored surface-feature value is unchanged. |
| `Natural-131.csv` | 131 pairs | Non-adversarial control cohort (TruthfulQA-Natural): open-ended factual QA pairs generated independently of TruthfulQA across twelve neutral topic domains (`category`), judge-verified (`judge_confidence` >= 0.8). Columns: `id, category, source_category, question, true_answer_natural, false_answer_natural, judge_confidence, model_used`. Surface6 audit AUC inside the cohort: ~0.64, driven mainly by answer length (true answers are longer in ~74% of pairs; fabricated answers also over-use connectives such as "which is why"). v1.1 (2026-08-30): 3 exact-duplicate questions and 1 judgment-based item removed, 6 text fields repaired after author source-verification. |

`TruthfulQA-476.csv` follows the upstream TruthfulQA schema:
`pair_id, Type, Category, Question, Best Answer, Best Incorrect Answer, subset_name`.

The cohorts load as their own configs:

```python
flipped = load_dataset("iclr2027-surface-audit/surface-audit", "SurfaceFlipped-131")["test"]
natural = load_dataset("iclr2027-surface-audit/surface-audit", "Natural-131")["test"]
```

## Verification and v1.1 changelog

Every released cohort pair was source-verified by the authors (five reviewers; the full sheet ships here):

- `verification_sheet_reviews.csv` — all 345 review rows (reviewer, verdicts, source link, note).
- `verification_sheet_adjudications.csv` — every v1.0 -> v1.1 change: original text, released text, reason (33 rows: 25 field repairs, 8 row removals).
- `v1.1_change_manifest.csv` — machine-readable diff; `archive_v1.0/` keeps the v1.0 files verbatim.
- `surfaceflipped_drift_flags.csv` — advisory drift-screen verdicts for the 111 post-pilot pairs (40 flagged, retained by design; the 20 pilot pairs were not screened).

Known, deliberate properties (disclosed rather than "fixed"): a few topically related question families remain in both cohorts (re-worded, non-identical items on popular misconceptions — normal for a misconception benchmark); one SurfaceFlipped item (id 14) is a legal-procedure myth generated inside the "Health Myths" batch and keeps that provenance label; one Natural pair (id 90117, Gandhi/Nehru) was debated during verification and retained — its FALSE side misattributes the leadership role, which the authors judged sufficient. All stored surface-feature columns are bit-identical to recomputation from the released texts.

## Cached embeddings

The frozen-encoder features used by the paper's classifier experiments are 160 MB and are not
distributed here. Rebuild them with the `scripts/build_*_embeddings.py` scripts in the code
repository, which record the exact checkpoints and pooling for every encoder family.

## Loading with pandas

```python
import pandas as pd

df = pd.read_csv("hf://datasets/iclr2027-surface-audit/surface-audit/TruthfulQA-476.csv")
flipped = pd.read_csv("hf://datasets/iclr2027-surface-audit/surface-audit/SurfaceFlipped-131.csv")
natural = pd.read_csv("hf://datasets/iclr2027-surface-audit/surface-audit/Natural-131.csv")
```

## Citation

```bibtex
@misc{anonymous2026judging,
  title  = {Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/iclr2027-surface-audit/surface-audit}
}
```

TruthfulQA itself is from Lin, Hilton & Evans (2022); please cite the original
benchmark when using this subset.
