# Audit-Prune — code and data for *Judging by the Cover*

Code, data, and per-table scripts for

> **Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage.**
> Anonymous authors. Under double-blind review, 2027.

Binary-choice truthfulness benchmarks can be gamed when correct and incorrect answers differ
systematically in surface form (negation, hedging, length). This repository holds the six-feature
audit (**Surface6**), the **Audit-Prune** cleaner (Algorithm 1), the released **TruthfulQA-476**
subset, the two held-out evaluation cohorts, and the script and result file behind every table and
figure in the paper.

The released datasets (TruthfulQA-476 and the two evaluation cohorts) are included here under
`hf_release/` and are also on the Hugging Face Hub at
<https://huggingface.co/datasets/iclr2027-surface-audit/surface-audit>.

## Layout

```
TruthfulQA.csv                       upstream binary-choice TruthfulQA (790 pairs; Lin et al., Apache-2.0)
audits/truthfulqa_style_audit.csv    per-answer surface-feature table read by every audit script
data/subsets/TruthfulQA-Audited/surface6/pair_ids/
                                     pair-id manifests: TruthfulQA-476 (pair_ids_theta053.json), thresholded
                                     subsets theta=0.50..0.65, fixed-prefix baselines, confidence and hybrid variants
data/predictions/                    per-item correctness of the 14-model open-weight panel (790 pairs each), the
                                     restyle/ check of Appendix L (8 open instruct models, 131 pairs x 2 renderings) and,
                                     under frontier/, of the four closed models used in Table 7 (July 2026 runs)
hf_release/                          TruthfulQA-476.csv, SurfaceFlipped-131.csv, Natural-131.csv, verification
                                     sheets, v1.1 change manifest, drift flags, README.md (dataset card), archive_v1.0/ (v1.0 files)
audits/random476_seeds_v3/           the ten random 476-pair draws used as Table 3's control
audits/_eval_dataset_v3_*            surface-flipped cohort pipeline: prompts, raw generations, validator, polarity
                                     gate, truth judge, drift screen, per-batch finals (batches 4-13 + seed pilot)
audits/_eval_dataset_natural_*       natural cohort pipeline: question and answer generations, judge, final
audits/_run_*.py, audits/_build_*.py cohort pipeline scripts (orchestrators and per-stage runners)
artifacts_790train/, artifacts_476train/
                                     the trained classifier heads scored in Table 3 (10 families x 2)
artifacts/embeddings/                cached frozen-encoder features (not distributed, 160 MB; rebuild locally)
results/                             one result file per table/figure; numbers printed in the paper come from here
paper_assets/figures/                the four figure PDFs as included in the paper
scripts/                             everything else (see the map below)
```

## Install

```
pip install -r requirements.txt          # or requirements-lock.txt for the exact pinned versions
```

Python 3.10+. Audits use scikit-learn 1.7 (`StandardScaler` + liblinear logistic regression,
grouped 5-fold CV, pooled out-of-fold predictions).

## Reproduce the released subset

```
python scripts/reproduce_truthfulqa476.py
```

Runs Algorithm 1 (beta-weighted imbalance score, refit at every step, grouped-CV audit, then the
add-back pass) on the Surface6 features and asserts, step by step, that the result equals the
released 476-pair manifest bit for bit. The script's docstring documents the one place where the
recorded removal path is replayed rather than re-derived.

`scripts/run_audit_prune_surface6.py` runs the generic threshold driver
(`run_audit_prune_thresholded.py`) with the Surface6 feature list, for sweeps and the scoring
variants of Table 6.

## Where each number comes from

| Paper item | Script | Result file |
|---|---|---|
| Sec. 2.1 audit (AUC 0.715, p), Sec. 3.1 (TruthfulQA-476, CI, p), Tables 9 (group rows), 12, 13, 15 | `scripts/repair_permutation_b10000.py` | `results/r5_d7_permutation_b10000.{json,csv}` |
| Table 14 (VitaminC) and plain-shuffle rows | `scripts/repair_permutation_plain_shuffle.py` | `results/r5b_d7_plain_shuffle_addendum.json` |
| Table 1 (per-feature ablation) | `scripts/a8_numbers.py` (recomputation) | `results/t3_table7_per_feature_surface6.{json,tex}`, `results/a8_numbers.json` |
| Table 2 (AFLite at matched N) | `scripts/run_aflite_baseline.py`, `scripts/d19_aflite_arms.py`; random rows `scripts/make_table2_random_rows.py` | `results/d19_aflite_arms.json`, `results/table2_random_rows.json` |
| Table 3 (adversarial / natural accuracy, v1.1) | `scripts/table3_v11.py` | `results/v1_1_rescore/table3_v1_1.json` |
| Table 7 (retained vs removed difficulty, Appendix C; also the 14-model panel numbers in its text) | `scripts/d37_difficulty_powered.py` | `results/d37_difficulty_powered.json` |
| Table 8 (IRT anchors) | `scripts/run_tinybenchmarks_truthfulqa_surface_audit.py` | `results/t10_irt_anchors_surface6.json` |
| Table 4 (category retention) | `scripts/make_table5_category_retention.py` | printed |
| Table 5 (scoring variants) | `scripts/run_audit_prune_surface6.py --strategies confidence imbalance` (partial) | `results/t7_appendix_a_surface6.{json,tex}` |
| Table 9 diagnostic rows, Table 10 (per-token) | frozen (see Known gaps) | `results/t3_table7_surface6.*`, `results/t3_per_token_neg_cnt_surface6.*` |
| Table 11 (cohort funnel) | `scripts/make_table9_cohort_funnel.py` | printed |
| Table 6 (ablation at theta = 0.53) | `scripts/run_audit_prune_surface6.py` (partial; see Known gaps) | `results/t5b_audit_prune_trajectory_theta050_surface6.json` (no-add-back row), `results/t5c_sweep_with_fidelity_surface6.json` (fixed-prefix minimum 0.5826) |
| Appendix L (zero-shot restyling check, 8 open instruct models) | `scripts/run_restyle_panel.py` (plain vs surface-inverted, shared A/B seed), `scripts/score_restyle_panel.py` (exact McNemar, Holm) | `data/predictions/restyle/*.csv` (per item, raw continuations), `data/predictions/restyle/summary.json` |
| Table 16 (theta sweep) | `scripts/run_audit_prune_surface6.py` (partial) | `results/t5c_sweep_with_fidelity_surface6.{json,tex}`, `results/audit_prune_surface6_repro/PROVENANCE.json` |
| Sec. 3.2 rank fidelity (rho = 0.915, CI) | `scripts/repair_rank_fidelity.py` | `results/r6_d6_rank_fidelity.json` |
| Sec. 3.2 fifty random subsets | `scripts/run_random_subset_fidelity.py` | `results/random_subset_fidelity.json` |
| Sec. 2.2 HaluBench contamination | `scripts/audit_halubench_contamination.py` | `results/halubench_contamination_results.json` |
| MedHallu audit row (Fig. 3, Table 15) | `scripts/medhallu/medhallu_pool.py` -> `medhallu_perm2000.py` -> `build_medhallu_figure_rows.py` | `results/medhallu_perm2000.json`, `results/medhallu_figure_rows.json` |
| Figure 1 | `scripts/make_figure1_overview.py`, then `scripts/crop_pdf_margins.py` | `paper_assets/figures/audit-prune-overview-v87.pdf` |
| Figure 2 | `paper_assets/fig/surface6_feature_cards_v18.tex` (pdflatex), then `scripts/crop_pdf_margins.py` | `paper_assets/figures/surface6_feature_cards_v18.pdf` |
| Figure 3 | `MEDHALLU_AUDIT_ONLY=1 scripts/render_cross_dataset_figure_v8.py`, then `scripts/crop_pdf_margins.py` | `paper_assets/figures/surface6-datasets_8.pdf` |
| Figure 4 | `scripts/render_theta_sweep_figure.py` (writes `theta_sweep_surface6.pdf`), then `scripts/crop_pdf_margins.py theta_sweep_surface6.pdf theta_sweep_surface6_v2.pdf` | `paper_assets/figures/theta_sweep_surface6_v2.pdf` |

Shared libraries: `scripts/surface_features_text.py` (Surface6 lexicons and extractor),
`truthfulqa_pruning_utils.py`, `audit_subset_evaluator.py`, `search_truthfulqa_pruned_improved.py`,
`repair_common.py`, `f02_clustered_table4.py`. Embedding builders `scripts/build_*_embeddings.py`
record the exact frozen checkpoints and pooling used for every encoder family;
`scripts/run_binary_choice_eval.py` produced the per-item prediction files.

## External data

* `TruthfulQA.csv` is redistributed here (Apache-2.0). Upstream: <https://github.com/sylinrl/TruthfulQA>.
* The cross-dataset audit reads public Hugging Face datasets at run time (HaluEval `qa`, BoolQ,
  VitaminC, SNLI, MultiNLI, ANLI, SuperGLUE RTE/MultiRC, OpenBookQA, SelfCheckGPT wiki_bio, PIQA,
  tinyTruthfulQA, HaluBench, MedHallu). FEVER's shared-task dev file is not redistributed; the
  download location is in `scripts/run_fever_audit.py`.
* Cached embeddings (`artifacts/embeddings/`, `results/v1_1_rescore/emb/`) are 160 MB and are not
  distributed with this release; rebuild them with the `scripts/build_*_embeddings.py` scripts, which
  record the exact frozen checkpoints and pooling for every encoder family.

## Cohorts

Both evaluation cohorts (n = 131 each) were LLM-generated, judge-screened, and then source-verified
by the authors; the full verification sheet, the v1.0 -> v1.1 change manifest, and the advisory
drift-screen verdicts ship in `hf_release/`. The generator and judge outputs are released in full
under `audits/`: every raw generation, validator table, polarity verdict, truth-judge verdict, and
drift verdict, plus the seed-pilot template, the revised template, and each batch's prompt with its
cumulative do-not-repeat list.

## Known gaps

Stated here so nobody has to discover them.

* The original one-shot wrappers behind Tables 1, 5 (hybrid rows), 6 (no-add-back and rule-cleaner
  rows), 9 (two diagnostic rows), 8, and 16 (sweep, fidelity columns) were run from a temporary directory in May 2026 and were not
  committed. Their outputs are the `results/` files listed above; the named scripts are exact
  recomputations (Table 1) or partial reproducers (Tables 5, 6, 16).
* No training script exists for the twenty pickled heads in `artifacts_*train/`; they are read as
  published artifacts by `scripts/table3_v11.py`.
* `audits/random476_seeds_v3/` manifests are self-describing (`default_rng(seed).choice(790, 476)`)
  but their generator script is not committed.
* The pilot batch's 27 judge-passing pairs were reduced to 20 by hand (`_eval_dataset_v3_final.csv`
  -> `_eval_dataset_v3_clean_final.csv`); the removed ids are 6, 8, 20, 21, 27, 37, 40.

## License and citation

Code and released data: Apache-2.0 (see `LICENSE`). TruthfulQA is the work of Lin, Hilton and Evans
(2022) and is redistributed under its Apache-2.0 license; please cite it when using these subsets.
Third-party datasets read at run time keep their own licenses (`THIRD_PARTY_NOTICES.md`).

See `CITATION.cff` for the citation entry.
