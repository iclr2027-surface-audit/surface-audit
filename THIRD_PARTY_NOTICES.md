# Third-party notices

| Item | Source | License | Use in this repository |
|---|---|---|---|
| `TruthfulQA.csv` (binary-choice TruthfulQA, 790 pairs) | Lin, Hilton, Evans (2022), https://github.com/sylinrl/TruthfulQA | Apache-2.0 | Redistributed verbatim; `TruthfulQA-476` is a subset of it (pair ids only are changed). |
| HaluEval QA (10,000 answer pairs) | Li et al. (2023), https://github.com/RUCAIBox/HaluEval | MIT | Not redistributed; `data/subsets/HaluEval-Audited/halueval_cleaned_pair_ids_theta055.json` lists the 645 retained pair ids only. |
| MedHallu (10,000 answer pairs; HF `UTAustin-AIHealth/MedHallu` revision `5150604`) | Pandit et al. (2025), https://github.com/MedHallu/MedHallu | MIT per the upstream GitHub and HF card body (the HF YAML metadata carries no license field); underlying PubMedQA / PubMed abstract text may carry publisher rights | 3,101 pairs redistributed verbatim in `hf_release/MedHallu-3101/` together with a manifest that rebuilds the subset without text. Labels are inherited from MedHallu, not re-adjudicated; not for clinical use. |
| Cross-dataset audit inputs (FEVER, VitaminC, MultiNLI, SNLI, ANLI, RTE, BoolQ, PIQA, OpenBookQA, MultiRC, SelfCheckGPT) | Hugging Face `datasets` / original releases | each dataset's own license | Loaded at run time by `scripts/`; not redistributed except the FEVER/VitaminC feature tables under `data/fever*`. |
| SurfaceFlipped-135, Natural-135 | generated for this paper with `claude-opus-4-5` (one fallback call `claude-opus-4-7`), label-checked by `gpt-5.4-2026-03-05`, human-verified against written sources | Apache-2.0 (this repository) | LLM-generated evaluation cohorts; released with generation prompts and per-batch screening logs under `audits/`. |
| Frontier-model responses (`results/frontier_eval/`) | OpenAI and Anthropic API outputs, July 2026 | provider terms | Cached per-item responses released for reproducibility of the frontier check. |
| Open-weight model checkpoints used for embeddings / evaluation | Hugging Face Hub | per-checkpoint (note: Qwen2.5-3B is under the Qwen Research license, non-commercial) | Not redistributed. |
