# Third-party notices

| Item | Source | License | Use in this repository |
|---|---|---|---|
| `TruthfulQA.csv` (binary-choice TruthfulQA, 790 pairs) | Lin, Hilton, Evans (2022), https://github.com/sylinrl/TruthfulQA | Apache-2.0 | Redistributed verbatim; `TruthfulQA-476` is a subset of it (pair ids only are changed). |
| HaluEval QA (10,000 answer pairs) | Li et al. (2023), https://github.com/RUCAIBox/HaluEval | MIT | Not redistributed; loaded at run time by `scripts/` for the cross-dataset audit row and the HaluBench contamination check. |
| MedHallu (10,000 answer pairs; HF `UTAustin-AIHealth/MedHallu` revision `5150604`) | Pandit et al. (2025), https://github.com/MedHallu/MedHallu | MIT per the upstream GitHub and HF card body (the HF YAML metadata carries no license field); underlying PubMedQA / PubMed abstract text may carry publisher rights | Not redistributed; the 10,000-pair pool is rebuilt from the pinned HF revision by `scripts/medhallu/medhallu_pool.py` for the audit row in Figure 3. Labels are inherited from MedHallu, not re-adjudicated; not for clinical use. |
| Cross-dataset audit inputs (FEVER, VitaminC, MultiNLI, SNLI, ANLI, RTE, BoolQ, PIQA, OpenBookQA, MultiRC, SelfCheckGPT) | Hugging Face `datasets` / original releases | each dataset's own license | Loaded at run time by `scripts/`; not redistributed. |
| SurfaceFlipped-131, Natural-131 (v1.1; the v1.0 135-pair files are kept under `hf_release/archive_v1.0/`) | generated for this paper with `claude-opus-4-5` (one fallback call `claude-opus-4-7`), label-checked by `gpt-5.4-2026-03-05`, human-verified against written sources | Apache-2.0 (this repository) | LLM-generated evaluation cohorts; released with generation prompts and per-batch screening logs under `audits/`. |
| Closed-model responses (`data/predictions/frontier/`) | OpenAI and Anthropic API outputs, July 2026 | provider terms | Cached per-item responses of the four closed models used as difficulty instruments in Table 7 (Appendix C). |
| Open-weight model checkpoints used for embeddings / evaluation | Hugging Face Hub | per-checkpoint (note: Qwen2.5-3B is under the Qwen Research license, non-commercial) | Not redistributed. |
