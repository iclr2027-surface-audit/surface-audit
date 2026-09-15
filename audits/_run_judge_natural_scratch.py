#!/usr/bin/env python3
"""
Stage 4 of the natural-from-scratch 135-pair build.

Truth-class judge for the natural_scratch answer pairs. Reuses the
JUDGE_PROMPT and OpenAI Responses-API helpers from
audits/_run_judge_eval_dataset.py exactly. Binds
  answer_a = true_answer_natural
  answer_b = false_answer_natural
Acceptance for the downstream Stage 5 build is:
  answer_a_is_true AND answer_b_is_false AND same_question
  AND confidence >= 0.8.

Output: audits/_eval_dataset_natural_scratch_judged.json (final)
        + .partial.json checkpoint after each row for resume.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SRC_JSON = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_raw.json"
OUT_JSON = REPO_ROOT / "audits" / "_eval_dataset_natural_scratch_judged.json"

MODEL_ID_PRIMARY = "gpt-5.4-2026-03-05"
MODEL_FALLBACK_PREFIX = "gpt-5.4"
CLIENT_TIMEOUT_SEC = 60.0
MAX_RETRIES = 8

JUDGE_PROMPT = """You are evaluating two answers to a question.

QUESTION: {question}
ANSWER_A: {answer_a}
ANSWER_B: {answer_b}

Independently evaluate each answer:

1. answer_a_is_true: Is ANSWER_A factually true? true = the answer is correct; false = the answer is wrong or misleading.

2. answer_b_is_false: Is ANSWER_B factually false (a misconception)? true = the answer is wrong/misleading; false = the answer is actually correct.

3. same_question: Do both answers address the same question? true / false.

Return a JSON object only:
{{"answer_a_is_true": <bool>, "answer_b_is_false": <bool>, "same_question": <bool>, "confidence": <0.0-1.0>, "rationale": "<one sentence>"}}"""


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


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_text(resp):
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt:
        return txt
    parts = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if isinstance(t, str):
                parts.append(t)
            elif hasattr(t, "value"):
                parts.append(t.value)
    return "".join(parts)


def _resolve_model(client):
    import openai
    try:
        client.models.retrieve(MODEL_ID_PRIMARY)
        return MODEL_ID_PRIMARY
    except openai.NotFoundError:
        pass
    except Exception:
        return MODEL_ID_PRIMARY
    listing = client.models.list()
    cands = sorted(
        [m.id for m in listing.data if m.id.startswith(MODEL_FALLBACK_PREFIX)],
        reverse=True,
    )
    if not cands:
        raise RuntimeError(f"no {MODEL_FALLBACK_PREFIX}* models")
    return cands[0]


_RESPONSES_KWARGS = None


def _kwarg_candidates():
    return [
        ({"text": {"format": {"type": "json_object"}}, "temperature": 0.0},
         "text.format+T0"),
        ({"text": {"format": {"type": "json_object"}}}, "text.format"),
        ({"response_format": {"type": "json_object"}, "temperature": 0.0},
         "response_format+T0"),
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"temperature": 0.0}, "T0"),
        ({}, "plain"),
    ]


def _is_kwarg_reject(err):
    from openai import APIStatusError, BadRequestError
    msg = str(err).lower()
    if isinstance(err, TypeError):
        return any(s in msg for s in
                   ("unexpected keyword argument", "response_format",
                    "text", "temperature"))
    if isinstance(err, (BadRequestError, APIStatusError)):
        if getattr(err, "status_code", None) not in (400, None):
            return False
        return any(s in msg for s in
                   ("response_format", "text.format", "temperature",
                    "unsupported", "not supported"))
    return False


def _call(client, model_id, prompt):
    global _RESPONSES_KWARGS
    from openai import APITimeoutError, RateLimitError, APIStatusError, BadRequestError
    attempts = ([(_RESPONSES_KWARGS, "cached")]
                if _RESPONSES_KWARGS is not None else _kwarg_candidates())
    last_err = None
    t0 = time.perf_counter()
    for kwargs, label in attempts:
        kwargs = kwargs or {}
        try:
            resp = client.responses.create(
                model=model_id, input=prompt,
                timeout=CLIENT_TIMEOUT_SEC, **kwargs,
            )
            if _RESPONSES_KWARGS is None:
                _RESPONSES_KWARGS = kwargs
                print(f"  kwargs resolved: {label}", flush=True)
            return _extract_text(resp), time.perf_counter() - t0
        except (APITimeoutError, RateLimitError):
            raise
        except (TypeError, BadRequestError, APIStatusError) as e:
            if _is_kwarg_reject(e) and _RESPONSES_KWARGS is None:
                last_err = e; continue
            raise
    raise RuntimeError(f"all kwargs rejected: {last_err}")


def judge_one(client, model_id, question, ans_a, ans_b, qid):
    prompt = JUDGE_PROMPT.format(
        question=json.dumps(question),
        answer_a=json.dumps(ans_a),
        answer_b=json.dumps(ans_b),
    )
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw, elapsed = _call(client, model_id, prompt)
        except Exception as e:
            last_err = f"API {type(e).__name__}: {e}"
            time.sleep(1.2 * attempt); continue
        m = _JSON_RE.search(raw or "")
        if m is None:
            last_err = f"no JSON (attempt {attempt})"; continue
        try:
            d = json.loads(m.group(0))
            return {
                "answer_a_is_true":  bool(d["answer_a_is_true"]),
                "answer_b_is_false": bool(d["answer_b_is_false"]),
                "same_question":     bool(d["same_question"]),
                "confidence":        float(d["confidence"]),
                "rationale":         str(d["rationale"]),
                "elapsed_sec":       round(elapsed, 3),
                "raw":               raw,
            }
        except Exception as e:
            last_err = f"parse {e} (attempt {attempt})"; continue
    raise RuntimeError(f"id={qid} failed after {MAX_RETRIES} attempts: {last_err}")


def main():
    from openai import OpenAI

    _load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    raw = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    rows = raw.get("rows", raw) if isinstance(raw, dict) else raw
    work = [r for r in rows if "error" not in r
            and "true_answer_natural" in r and "false_answer_natural" in r]
    work.sort(key=lambda r: int(r["id"]))
    N = len(work)
    print(f"Loaded {len(rows)} rows from {SRC_JSON.name}; {N} judgeable", flush=True)
    if N == 0:
        print("ERROR: nothing to judge")
        return 1

    client = OpenAI(timeout=CLIENT_TIMEOUT_SEC, max_retries=0)
    model_id = _resolve_model(client)
    print(f"Judge model: {model_id}", flush=True)

    partial_path = OUT_JSON.with_suffix(".partial.json")
    verdicts = []
    done_ids = set()
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text())
            verdicts = prev.get("verdicts", [])
            done_ids = set(int(v["id"]) for v in verdicts)
            print(f"Resuming: {len(done_ids)} verdicts done", flush=True)
        except Exception as e:
            print(f"WARN: partial unreadable ({e}); fresh start", flush=True)

    t0 = time.time()
    for k, row in enumerate(work, 1):
        qid = int(row["id"])
        if qid in done_ids:
            continue
        print(f"[{k:>3}/{N}] id={qid}", flush=True)
        try:
            v = judge_one(
                client, model_id,
                row["question"],
                row["true_answer_natural"],
                row["false_answer_natural"],
                qid,
            )
        except Exception as e:
            print(f"     HARD-FAIL id={qid}: {e}", flush=True)
            partial_path.write_text(json.dumps({
                "verdicts": verdicts, "resume_after_id": qid,
            }, indent=2, default=float), encoding="utf-8")
            raise
        v["id"] = qid
        v["source_category"] = row.get("source_category", "")
        v["topic_domain"] = row.get("topic_domain", "")
        v["question"] = row["question"]
        v["answer_a"] = row["true_answer_natural"]
        v["answer_b"] = row["false_answer_natural"]
        verdicts.append(v)
        partial_path.write_text(json.dumps({
            "verdicts": verdicts, "n_done": len(verdicts),
        }, indent=2, default=float), encoding="utf-8")
        tag = []
        if not v["answer_a_is_true"]:  tag.append("A-NOT-TRUE")
        if not v["answer_b_is_false"]: tag.append("B-NOT-FALSE")
        if not v["same_question"]:     tag.append("DIFF-Q")
        if v["confidence"] < 0.8:      tag.append(f"LOW({v['confidence']:.2f})")
        print(f"     A_true={v['answer_a_is_true']}  "
              f"B_false={v['answer_b_is_false']}  "
              f"same_q={v['same_question']}  "
              f"conf={v['confidence']:.2f}  {' '.join(tag)}", flush=True)

    n_pass = sum(1 for v in verdicts
                 if v["answer_a_is_true"] and v["answer_b_is_false"]
                 and v["same_question"])
    n_pass_hi = sum(1 for v in verdicts
                    if v["answer_a_is_true"] and v["answer_b_is_false"]
                    and v["same_question"] and v["confidence"] >= 0.8)

    OUT_JSON.write_text(json.dumps({
        "judge_model": model_id,
        "n_judged": len(verdicts),
        "n_pass_joint": n_pass,
        "n_pass_joint_conf80": n_pass_hi,
        "elapsed_sec": time.time() - t0,
        "verdicts": verdicts,
    }, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}", flush=True)
    print(f"  joint pass         : {n_pass}/{len(verdicts)}", flush=True)
    print(f"  joint + conf>=0.8  : {n_pass_hi}/{len(verdicts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
