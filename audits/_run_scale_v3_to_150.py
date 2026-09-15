#!/usr/bin/env python3
"""
Master orchestrator: scale v3 dataset to n>=150 using batch4 pipeline.

Loops batches starting from batch5 (id offset +4000):
  - cumulative-blacklist the prior questions into a per-batch prompt copy
  - generate -> id-remap -> mechanical -> polarity -> judge -> drift
  - merge new kept rows into running combined CSV
  - update progress markdown after each batch

Stops when any of:
  - cumulative n >= 150
  - 10 batches completed (batch5..batch14)
  - a single batch's net yield (kept/generated) < 30%
"""
from __future__ import annotations
import json, os, re, string, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "audits"
PY = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"

SEED_CSV = AUD / "_eval_dataset_v3_b1b4_final.csv"
RUN_CSV = AUD / "_eval_dataset_v3_scale_combined.csv"
PROGRESS_MD = AUD / "_eval_dataset_v3_scale_progress.md"

BATCH4_PROMPT = AUD / "_eval_dataset_v3_batch4_prompt.txt"

START_BATCH = 5
END_BATCH = 14  # inclusive: 10 batches max
TARGET_N = 150
MIN_YIELD_PCT = 0.30


def norm_q(q: str) -> str:
    q = (q or "").lower().strip()
    q = q.translate(str.maketrans("", "", string.punctuation))
    q = re.sub(r"\s+", " ", q).strip()
    return q


def init_combined():
    """Seed combined CSV with batch1+batch4 if not already started."""
    import pandas as pd
    if RUN_CSV.exists():
        df = pd.read_csv(RUN_CSV)
        print(f"[init] resume: {RUN_CSV.name} has {len(df)} rows", flush=True)
        return df
    df = pd.read_csv(SEED_CSV)
    df.to_csv(RUN_CSV, index=False)
    print(f"[init] seeded {RUN_CSV.name} with {len(df)} rows", flush=True)
    return df


def build_blacklist(combined_df) -> list:
    return list(dict.fromkeys(combined_df["question"].astype(str).tolist()))


def write_batch_prompt(N: int, banned_questions: list) -> Path:
    """Copy batch4 prompt; replace BANNED QUESTIONS section with cumulative list."""
    src = BATCH4_PROMPT.read_text(encoding="utf-8")
    # Section 7 starts with "7. BANNED QUESTIONS" and ends before "OUTPUT FORMAT:"
    pre_match = re.search(r"\n7\. BANNED QUESTIONS[^\n]*\n", src)
    out_match = re.search(r"\nOUTPUT FORMAT:\n", src)
    if not pre_match or not out_match:
        raise RuntimeError("could not locate BANNED/OUTPUT markers in batch4 prompt")
    head = src[: pre_match.start()]
    tail = src[out_match.start():]  # includes leading \n

    bullets = "\n".join(f"   - {q}" for q in banned_questions)
    new_section = (
        f"\n7. BANNED QUESTIONS (already in prior batches — pick DIFFERENT misconceptions):\n"
        f"{bullets}\n"
        f"   Avoid near-paraphrases too. Pick fresh misconceptions from new categories\n"
        f"   if topics are getting crowded.\n"
    )
    out_text = head + new_section + tail
    out_path = AUD / f"_eval_dataset_v3_batch{N}_prompt.txt"
    out_path.write_text(out_text, encoding="utf-8")
    return out_path


def write_remap_script(N: int) -> Path:
    offset = N * 1000
    out = AUD / f"_run_id_remap_batch{N}.py"
    out.write_text(
        '#!/usr/bin/env python3\n'
        f'"""Remap batch{N} raw-JSON ids 1..20 -> {offset+1}..{offset+20} in place."""\n'
        'from __future__ import annotations\n'
        'import json\n'
        'from pathlib import Path\n'
        '\n'
        'REPO = Path(__file__).resolve().parent.parent\n'
        f'RAW = REPO / "audits" / "_eval_dataset_v3_batch{N}_raw.json"\n'
        f'OFFSET = {offset}\n'
        '\n'
        'def main():\n'
        '    data = json.loads(RAW.read_text())\n'
        '    rows = data["rows"] if isinstance(data, dict) else data\n'
        '    bumped = 0\n'
        '    for r in rows:\n'
        '        old = int(r.get("id", 0))\n'
        '        if old < OFFSET:\n'
        '            r["id"] = old + OFFSET\n'
        '            bumped += 1\n'
        '    out = data if isinstance(data, dict) else rows\n'
        '    if isinstance(out, dict):\n'
        '        out["rows"] = rows\n'
        '    RAW.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")\n'
        '    print(f"Remapped {bumped}/{len(rows)} ids by +{OFFSET}; new range = '
        '[{min(r[\'id\'] for r in rows)}, {max(r[\'id\'] for r in rows)}]")\n'
        '    return 0\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    raise SystemExit(main())\n',
        encoding="utf-8",
    )
    return out


def run(cmd: list, env_extra: dict | None = None, label: str = "") -> tuple[int, str]:
    """Run a subprocess, stream output to stdout, return (returncode, captured)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"\n>>> {label or ' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, cwd=REPO)
    chunks = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    rc = proc.wait()
    print(f">>> exit={rc} elapsed={time.time()-t0:.1f}s", flush=True)
    return rc, "".join(chunks)


def run_polarity(N: int) -> Path:
    """Reuse _run_polarity_gate_batch3 by patching its module attrs."""
    in_csv = AUD / f"_eval_dataset_v3_batch{N}_validated.csv"
    out_json = AUD / f"_eval_dataset_v3_batch{N}_polarity.json"
    out_csv = AUD / f"_eval_dataset_v3_batch{N}_polarity_passed.csv"
    code = (
        "import sys, importlib\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"sys.path.insert(0, {str(AUD)!r})\n"
        "import _run_polarity_gate_batch3 as pg\n"
        "from pathlib import Path\n"
        f"pg.IN_CSV  = Path({str(in_csv)!r})\n"
        f"pg.OUT_JSON = Path({str(out_json)!r})\n"
        f"pg.OUT_CSV  = Path({str(out_csv)!r})\n"
        "raise SystemExit(pg.main())\n"
    )
    rc, _ = run([PY, "-c", code], label=f"polarity gate batch{N}")
    if rc != 0:
        raise RuntimeError(f"polarity gate failed for batch{N}")
    return out_csv


def run_judge(N: int) -> Path:
    """Run truth judge on polarity_passed (write a temp validated CSV first)."""
    import pandas as pd
    polar_csv = AUD / f"_eval_dataset_v3_batch{N}_polarity_passed.csv"
    df = pd.read_csv(polar_csv)
    # Judge wants `validated.csv` with all_passes column. Mark rows that
    # already passed mechanics+polarity as all_passes=True.
    df["all_passes"] = True
    judge_input = AUD / f"_eval_dataset_v3_batch{N}p_validated.csv"
    df.to_csv(judge_input, index=False)
    rc, _ = run(
        [PY, str(AUD / "_run_judge_eval_dataset.py")],
        env_extra={"EVAL_DATASET_VERSION": f"v3_batch{N}p"},
        label=f"judge batch{N} (post-polarity)",
    )
    if rc != 0:
        raise RuntimeError(f"judge failed for batch{N}")
    return AUD / f"_eval_dataset_v3_batch{N}p_judged.json"


def finalize_judged(N: int) -> Path:
    """Build per-batch final.csv from polarity_passed + judged JSON."""
    import pandas as pd
    polar_csv = AUD / f"_eval_dataset_v3_batch{N}_polarity_passed.csv"
    judged_json = AUD / f"_eval_dataset_v3_batch{N}p_judged.json"
    final_csv = AUD / f"_eval_dataset_v3_batch{N}_final.csv"

    df = pd.read_csv(polar_csv)
    j = json.loads(judged_json.read_text())
    by_id = {int(v["id"]): v for v in j["verdicts"]}
    keep_ids = [
        i for i, v in by_id.items()
        if v["answer_a_is_true"] and v["answer_b_is_false"]
        and v["same_question"] and v["confidence"] >= 0.8
    ]
    out = df[df["id"].isin(keep_ids)].copy().reset_index(drop=True)
    # backfill judge fields
    out["judge_confidence"] = [by_id[int(i)]["confidence"] for i in out["id"]]
    out["judge_rationale"]  = [by_id[int(i)]["rationale"]  for i in out["id"]]
    out.to_csv(final_csv, index=False)
    print(f"  finalize: kept {len(out)}/{len(df)} after judge", flush=True)
    return final_csv


def run_drift(N: int) -> Path:
    """Reuse _run_drift_screen_batch2 by patching attrs."""
    in_csv = AUD / f"_eval_dataset_v3_batch{N}_final.csv"
    out_csv = AUD / f"_eval_dataset_v3_batch{N}_drift_screened.csv"
    out_json = AUD / f"_eval_dataset_v3_batch{N}_drift_dropped.json"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"sys.path.insert(0, {str(AUD)!r})\n"
        "import _run_drift_screen_batch2 as ds\n"
        "from pathlib import Path\n"
        f"ds.IN_CSV  = Path({str(in_csv)!r})\n"
        f"ds.OUT_CSV = Path({str(out_csv)!r})\n"
        f"ds.OUT_JSON = Path({str(out_json)!r})\n"
        "raise SystemExit(ds.main())\n"
    )
    rc, _ = run([PY, "-c", code], label=f"drift screen batch{N}")
    if rc != 0:
        raise RuntimeError(f"drift failed for batch{N}")
    return out_csv


def merge_into_combined(N: int, combined_df, drift_csv: Path):
    """Append new rows to combined CSV with batch=N column, deduping on normalized question."""
    import pandas as pd
    new = pd.read_csv(drift_csv)
    new["batch"] = N
    existing_norms = set(norm_q(q) for q in combined_df["question"].astype(str).tolist())
    keep = new[~new["question"].astype(str).map(norm_q).isin(existing_norms)].copy()
    n_new = len(keep)
    n_dup = len(new) - n_new
    # add missing columns to keep, then drop extras and align to combined order
    for col in combined_df.columns:
        if col not in keep.columns:
            keep[col] = None
    keep = keep[combined_df.columns]
    out = pd.concat([combined_df, keep], ignore_index=True)
    out.to_csv(RUN_CSV, index=False)
    return out, n_new, n_dup


def update_progress(stats: list, final: bool = False):
    lines = [
        "# v3 dataset scale-up progress",
        "",
        f"Target n: {TARGET_N}    Max batches: 5..{END_BATCH} (10 batches)    "
        f"Min yield: {MIN_YIELD_PCT*100:.0f}%",
        "",
        "## Per-batch results",
        "",
        "| Batch | Generated | Mech | Polarity | Judge | Drift | New (after dedup) | Yield % | Cumulative n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        lines.append(
            f"| {s['batch']} | {s['gen']} | {s['mech']} | {s['polarity']} | "
            f"{s['judge']} | {s['drift']} | {s['new']} | "
            f"{s['yield_pct']*100:.1f}% | {s['cum_n']} |"
        )
    if stats:
        last = stats[-1]
        lines.append("")
        lines.append(f"Latest cumulative n = **{last['cum_n']}** (target {TARGET_N})")
    if final:
        lines.append("")
        lines.append("## Loop terminated")
        lines.append("")
        for s in stats:
            if s.get("stop_reason"):
                lines.append(f"- {s['stop_reason']}")
    PROGRESS_MD.write_text("\n".join(lines), encoding="utf-8")


def per_batch_pipeline(N: int, combined_df) -> tuple[dict, "pd.DataFrame"]:
    import pandas as pd
    print(f"\n{'='*70}\n=== BATCH {N} ===\n{'='*70}", flush=True)

    # 1. cumulative blacklist + prompt
    banned = build_blacklist(combined_df)
    prompt_path = write_batch_prompt(N, banned)
    print(f"[batch{N}] wrote prompt with {len(banned)} banned questions: "
          f"{prompt_path.name}", flush=True)

    # 2. generate
    raw_path = AUD / f"_eval_dataset_v3_batch{N}_raw.json"
    if not raw_path.exists():
        rc, _ = run(
            [PY, str(AUD / "_run_generate_eval_dataset.py")],
            env_extra={
                "EVAL_DATASET_VERSION": f"v3_batch{N}",
                "EVAL_DATASET_PROMPT_VERSION": f"v3_batch{N}",
                "EVAL_DATASET_N_TARGET": "20",
            },
            label=f"generate batch{N}",
        )
        if rc != 0:
            raise RuntimeError(f"generate failed batch{N}")
    else:
        print(f"[batch{N}] raw json exists, skipping generation", flush=True)

    gen_data = json.loads(raw_path.read_text())
    gen_rows = gen_data["rows"] if isinstance(gen_data, dict) else gen_data
    n_gen = len(gen_rows)

    # 3. id remap
    remap_script = write_remap_script(N)
    if max(int(r.get("id", 0)) for r in gen_rows) < N * 1000:
        rc, _ = run([PY, str(remap_script)], label=f"id remap batch{N}")
        if rc != 0:
            raise RuntimeError(f"remap failed batch{N}")
    else:
        print(f"[batch{N}] ids already remapped", flush=True)

    # 4. mechanical validate
    rc, _ = run(
        [PY, str(AUD / "_run_validate_eval_dataset.py")],
        env_extra={"EVAL_DATASET_VERSION": f"v3_batch{N}"},
        label=f"mechanical validate batch{N}",
    )
    if rc != 0:
        raise RuntimeError(f"validate failed batch{N}")
    val_df = pd.read_csv(AUD / f"_eval_dataset_v3_batch{N}_validated.csv")
    n_mech = int(val_df["all_passes"].sum())

    if n_mech == 0:
        return {"batch": N, "gen": n_gen, "mech": 0, "polarity": 0,
                "judge": 0, "drift": 0, "new": 0, "yield_pct": 0.0,
                "cum_n": len(combined_df)}, combined_df

    # 5. polarity gate
    polar_csv = run_polarity(N)
    polar_df = pd.read_csv(polar_csv)
    n_polar = len(polar_df)

    if n_polar == 0:
        return {"batch": N, "gen": n_gen, "mech": n_mech, "polarity": 0,
                "judge": 0, "drift": 0, "new": 0, "yield_pct": 0.0,
                "cum_n": len(combined_df)}, combined_df

    # 6. judge
    run_judge(N)
    final_csv = finalize_judged(N)
    n_judge = len(pd.read_csv(final_csv))

    if n_judge == 0:
        return {"batch": N, "gen": n_gen, "mech": n_mech, "polarity": n_polar,
                "judge": 0, "drift": 0, "new": 0, "yield_pct": 0.0,
                "cum_n": len(combined_df)}, combined_df

    # 7. drift screen
    drift_csv = run_drift(N)
    drift_df = pd.read_csv(drift_csv)
    n_drift = len(drift_df)

    if n_drift == 0:
        return {"batch": N, "gen": n_gen, "mech": n_mech, "polarity": n_polar,
                "judge": n_judge, "drift": 0, "new": 0, "yield_pct": 0.0,
                "cum_n": len(combined_df)}, combined_df

    # 8. merge into combined
    combined_df, n_new, n_dup = merge_into_combined(N, combined_df, drift_csv)
    yield_pct = n_new / max(n_gen, 1)
    return {"batch": N, "gen": n_gen, "mech": n_mech, "polarity": n_polar,
            "judge": n_judge, "drift": n_drift, "new": n_new,
            "yield_pct": yield_pct, "cum_n": len(combined_df),
            "dup": n_dup}, combined_df


def main():
    import pandas as pd
    combined = init_combined()
    stats = []
    update_progress(stats)

    stop_reason = None
    for N in range(START_BATCH, END_BATCH + 1):
        try:
            s, combined = per_batch_pipeline(N, combined)
        except Exception as e:
            print(f"[batch{N}] FATAL: {e}", flush=True)
            stop_reason = f"batch{N} crashed: {e}"
            stats.append({"batch": N, "gen": 0, "mech": 0, "polarity": 0,
                          "judge": 0, "drift": 0, "new": 0, "yield_pct": 0.0,
                          "cum_n": len(combined), "stop_reason": stop_reason})
            update_progress(stats, final=True)
            break
        stats.append(s)
        update_progress(stats)
        print(f"\n[batch{N}] DONE  yield={s['yield_pct']*100:.1f}%  "
              f"cum_n={s['cum_n']}", flush=True)

        if s["cum_n"] >= TARGET_N:
            stop_reason = f"target reached: cum_n={s['cum_n']} >= {TARGET_N}"
            stats[-1]["stop_reason"] = stop_reason
            break
        if s["gen"] > 0 and s["yield_pct"] < MIN_YIELD_PCT:
            stop_reason = (f"yield collapsed: batch{N} yield "
                           f"{s['yield_pct']*100:.1f}% < {MIN_YIELD_PCT*100:.0f}%")
            stats[-1]["stop_reason"] = stop_reason
            break
    else:
        stop_reason = f"reached max batches (batch{END_BATCH})"
        if stats:
            stats[-1]["stop_reason"] = stop_reason

    update_progress(stats, final=True)

    # final summary print
    print(f"\n\n{'='*70}\nSCALE-UP COMPLETE\n{'='*70}")
    print(f"  stop reason : {stop_reason}")
    print(f"  cumulative n: {len(combined)}")
    print(f"  combined CSV: {RUN_CSV.relative_to(REPO)}")
    print(f"  progress MD : {PROGRESS_MD.relative_to(REPO)}")
    print(f"\nPer-batch:")
    for s in stats:
        print(f"  batch{s['batch']}: gen={s['gen']} mech={s['mech']} "
              f"polar={s['polarity']} judge={s['judge']} drift={s['drift']} "
              f"new={s['new']} yield={s['yield_pct']*100:.1f}% "
              f"cum_n={s['cum_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
