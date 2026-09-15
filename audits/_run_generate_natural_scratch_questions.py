#!/usr/bin/env python3
"""
Stage 1 of the natural-from-scratch 135-pair build.

Generate ~1.35x the target count of open-ended factual questions for each
neutral topic_domain. The generator sees ONLY the neutral topic label —
not TruthfulQA, not the existing 135, not any prior adversarial output,
not the original category names.

Output: audits/_eval_dataset_natural_scratch_questions_raw.json
  Each row: id (>= 90001), source_category, topic_domain, question.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_questions_raw.json"

MODEL_CHAIN = ["claude-opus-4-5", "claude-opus-4-7", "claude-sonnet-4-6"]
TEMPERATURE = 0.7
MAX_TOKENS = 32000
NO_TEMPERATURE_MODELS = {"claude-opus-4-7"}
ID_START = 90001

# (source_category, topic_domain, target_count, overgen_count)
PLAN = [
    ("Animal Myths",     "Animal biology and behavior",          24, 32),
    ("Health Myths",     "Human health and medicine",            19, 26),
    ("Food Myths",       "Food and nutrition",                   14, 19),
    ("Pseudoscience",    "Science and physics",                  14, 19),
    ("Body Myths",       "Human anatomy and physiology",         11, 15),
    ("History Myths",    "Historical events and figures",        11, 15),
    ("Superstitions",    "Cultural traditions and practices",    10, 14),
    ("Folk Wisdom",      "Everyday knowledge and practical facts", 8, 11),
    ("Misconceptions",   "General knowledge",                     8, 11),
    ("Mandela Effect",   "Pop culture and media history",         7,  9),
    ("Urban Legends",    "Modern history and society",            6,  8),
    ("Language Myths",   "Language and linguistics",              3,  4),
]

PROMPT_TEMPLATE = """Generate {N} factual questions in the category: "{category}".

RULES - strict:
- Open-ended factual format only. Must start with one of: What, How, When, Where, Who, Why, Which.
- NO yes/no questions. NO "Is it true that..." NO "Does X..." NO "Did X..." NO misconception probes.
- Each question must have a single clear factual answer that an expert would agree on.
- Avoid contested, opinion-based, or subjective topics.
- Topics should be diverse within the category - no near-duplicates.
- Each question should be answerable in 1-3 sentences of natural prose.

Return a JSON object: {{"questions": ["...", "...", ...]}}
Return only the JSON. No commentary."""

OPEN_STARTERS = ("what", "how", "when", "where", "who", "why", "which")


def _load_dotenv(path: Path):
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    return t.strip()


def _parse_obj(text: str) -> dict:
    t = _strip_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("could not parse JSON object from response")


def _norm(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _is_open_ended(q: str) -> bool:
    s = q.strip().lower()
    if not s:
        return False
    first = s.split()[0] if s.split() else ""
    return first.rstrip("?!.,") in OPEN_STARTERS


def _call_one(client, category: str, n: int) -> tuple[list[str], str, dict]:
    prompt = PROMPT_TEMPLATE.format(N=n, category=category)
    last_err = None
    for model_id in MODEL_CHAIN:
        use_temp = model_id not in NO_TEMPERATURE_MODELS
        try:
            kwargs = {
                "model": model_id,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }
            if use_temp:
                kwargs["temperature"] = TEMPERATURE
            with client.messages.stream(**kwargs) as stream:
                for _ in stream.text_stream:
                    pass
                resp = stream.get_final_message()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            obj = _parse_obj(text)
        except Exception as e:
            last_err = f"parse({model_id}): {e} | first200={text[:200]!r}"
            continue
        questions = obj.get("questions") if isinstance(obj, dict) else None
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            last_err = f"bad_shape({model_id}): keys={list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__}"
            continue
        usage = getattr(resp, "usage", None)
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        }
        return [q.strip() for q in questions if q.strip()], model_id, usage_dict
    raise RuntimeError(last_err or "all models failed without error")


def _load_existing() -> dict:
    if not OUT_JSON.exists():
        return {}
    try:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data.get("rows", []) if isinstance(data, dict) else data
    by_topic = {}
    for r in rows:
        by_topic.setdefault(r["topic_domain"], []).append(r)
    return by_topic


def main():
    _load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    from anthropic import Anthropic
    client = Anthropic()

    existing_by_topic = _load_existing()
    if existing_by_topic:
        print(f"Resume: found existing rows for {len(existing_by_topic)} topics", flush=True)

    all_rows: list[dict] = []
    next_id = ID_START
    used_ids: set[int] = set()
    for topic_rows in existing_by_topic.values():
        for r in topic_rows:
            all_rows.append(r)
            used_ids.add(int(r["id"]))
    if used_ids:
        next_id = max(max(used_ids) + 1, ID_START)

    meta = {
        "model_chain": MODEL_CHAIN,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "no_temperature_models": sorted(NO_TEMPERATURE_MODELS),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    t0 = time.time()
    summary = []
    for source_cat, topic, target, overgen in PLAN:
        existing = existing_by_topic.get(topic, [])
        if len(existing) >= overgen:
            print(f"[skip ] {topic:42s} have={len(existing)} >= overgen={overgen}", flush=True)
            summary.append((source_cat, topic, target, overgen, len(existing), len(existing), "cached"))
            continue

        need = overgen - len(existing)
        # Ask for a bit more than needed to absorb dedup losses.
        ask = max(need, int(round(need * 1.15)))
        print(f"[gen  ] {topic:42s} target={target}  overgen={overgen}  "
              f"existing={len(existing)}  asking={ask}", flush=True)

        try:
            qs, model_used, usage = _call_one(client, topic, ask)
        except Exception as e:
            print(f"  HARD-FAIL: {e}", flush=True)
            summary.append((source_cat, topic, target, overgen, len(existing), len(existing), "fail"))
            continue

        # Dedup against existing within this topic, by normalized form,
        # and discard non-open-ended forms.
        seen_norm = {_norm(r["question"]) for r in existing}
        kept_new = []
        dropped = {"dup": 0, "not_open": 0, "blank": 0}
        for q in qs:
            qn = _norm(q)
            if not qn:
                dropped["blank"] += 1; continue
            if not _is_open_ended(q):
                dropped["not_open"] += 1; continue
            if qn in seen_norm:
                dropped["dup"] += 1; continue
            seen_norm.add(qn)
            kept_new.append(q)

        # Keep only enough to reach overgen target (first ones).
        room = overgen - len(existing)
        kept_new = kept_new[:room]

        new_rows = []
        for q in kept_new:
            new_rows.append({
                "id": next_id,
                "source_category": source_cat,
                "topic_domain": topic,
                "question": q,
                "model_used": model_used,
            })
            next_id += 1

        topic_total = existing + new_rows
        existing_by_topic[topic] = topic_total
        all_rows.extend(new_rows)

        print(f"  returned={len(qs)}  kept_new={len(kept_new)}  "
              f"dropped={dropped}  topic_total={len(topic_total)}/{overgen}  "
              f"model={model_used}  in/out={usage['input_tokens']}/{usage['output_tokens']}",
              flush=True)
        summary.append((source_cat, topic, target, overgen,
                        len(existing) + len(kept_new), len(qs),
                        f"ok({model_used})"))

        # Persist after each topic.
        rows_sorted = sorted(all_rows, key=lambda r: int(r["id"]))
        OUT_JSON.write_text(json.dumps({
            **meta,
            "n_rows": len(rows_sorted),
            "rows": rows_sorted,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final trim: each topic to exactly overgen, first-ones-kept by id.
    trimmed_rows = []
    for source_cat, topic, target, overgen in PLAN:
        topic_rows = sorted(existing_by_topic.get(topic, []), key=lambda r: int(r["id"]))
        if len(topic_rows) > overgen:
            print(f"[trim ] {topic:42s} {len(topic_rows)} -> {overgen}", flush=True)
            topic_rows = topic_rows[:overgen]
        trimmed_rows.extend(topic_rows)

    trimmed_rows = sorted(trimmed_rows, key=lambda r: int(r["id"]))
    OUT_JSON.write_text(json.dumps({
        **meta,
        "elapsed_sec": round(time.time() - t0, 2),
        "n_rows": len(trimmed_rows),
        "rows": trimmed_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 90)
    print(f"{'source_category':18s} {'topic_domain':42s} {'tgt':>4s} {'over':>5s} {'have':>5s} {'ret':>5s}  status")
    print("=" * 90)
    for src, topic, target, overgen, have, ret, status in summary:
        print(f"{src:18s} {topic:42s} {target:>4d} {overgen:>5d} {have:>5d} {ret:>5d}  {status}")
    print("=" * 90)
    by_topic = {}
    for r in trimmed_rows:
        by_topic.setdefault(r["topic_domain"], 0)
        by_topic[r["topic_domain"]] += 1
    print()
    print("Final per-topic counts (after trim):")
    for src, topic, target, overgen in PLAN:
        n = by_topic.get(topic, 0)
        ok = "OK" if n == overgen else f"SHORT(-{overgen-n})" if n < overgen else f"OVER(+{n-overgen})"
        print(f"  {topic:42s}  n={n:>3d}  overgen={overgen:>3d}  {ok}")
    print(f"\nTotal questions: {len(trimmed_rows)}")
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
