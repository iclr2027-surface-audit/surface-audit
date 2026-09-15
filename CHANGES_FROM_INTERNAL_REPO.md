# Changes from the internal working repository

Every file below is a verbatim copy of its internal-repo source except for the edits listed.

* `scripts/table3_v11.py`: moved from results/v1_1_rescore/scripts/; repository root resolved relative to the file; f02 loaded from scripts/
* `scripts/encode_splice_v11.py`: moved from results/v1_1_rescore/scripts/; repository root resolved relative to the file
* `scripts/f02_clustered_table4.py`: moved from audits/iclr_new_checks/scripts/; repository root resolved relative to the file
* `scripts/a8_numbers.py`: moved from audits/iclr_repair_results/scripts/; repository root resolved relative to the file; output written to results/
* `scripts/audit_halubench_contamination.py`: moved from an internal analysis directory; repository root resolved relative to the file; output written to results/halubench_contamination_results.json
* `scripts/render_theta_sweep_figure.py`: removed a preview PNG written to a local desktop path
* `scripts/render_cross_dataset_figure_vertical_v3.py`: MEDHALLU_AUDIT_ONLY=1 (the paper's Figure 3 mode) is now the default; the other modes read files that are not part of the release
* `scripts/medhallu/medhallu_perm2000.py`: audits only the full 10k MedHallu pool (the paper's row); the MedHallu-3101 subset arm was removed with that release
* `scripts/medhallu/build_medhallu_figure_rows.py`: writes only the full-MedHallu row (the paper's); MedHallu-3101 row and a fallback path removed
* `scripts/run_aflite_baseline.py`: docstring run line uses `python` instead of an absolute interpreter path
* `scripts/run_random_subset_fidelity.py`: docstring run line uses `python` instead of an absolute interpreter path
* `CITATION.cff`: software title and paper title updated to the current manuscript
* `results/table2_random_rows.json`: the Table 2 random-subset and Audit-Prune rows extracted from a larger internal results file; only the fields the paper prints are kept
* `results/medhallu_perm2000.json`: only the `MedHallu full` block (the paper's row) is kept
* new scripts: `scripts/run_audit_prune_surface6.py`, `scripts/make_table5_category_retention.py`, `scripts/make_table9_cohort_funnel.py`, `scripts/make_table2_random_rows.py`
* `README.md` and `hf_release/README.md` (dataset card) are new; the dataset card drops the withdrawn HaluEval-645 and MedHallu-3101 releases
