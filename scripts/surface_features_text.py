#!/usr/bin/env python3
"""
Per-answer text-to-feature extractor for the 10-dimensional TruthfulQA
"surface10" feature set.

Lexicons match scripts/build_audit_notebook.py, which generated the audit
CSV backing the paper's published results (audits/truthfulqa_style_audit.csv,
N=790). Using these same lexicons here guarantees that the adversarial
experiment tests the actual paper classifier rather than a FEVER-lexicon
variant with the same name. (scripts/run_fever_audit.py uses a different,
stricter negation lexicon; that inconsistency across the cross-dataset
audit is known and tracked separately for camera-ready cleanup.)

The ten features returned by `extract_surface10(text)`:

    neg_lead        1.0 if re.match(p, str(text).strip().lower())
                    matches for any pattern p in NEG_LEADS, else 0.0.
    neg_cnt         Number of NEG_WORDS matches via re.findall on the
                    lowercased raw text. Returned as float.
    hedge_rate      count(HEDGES) / word_count      (per-answer, not a
                                                    pair-level diff)
    auth_rate       count(AUTH_CUES) / word_count
    len_gap         Pair-level feature. extract_surface10 returns 0.0 as
                    a placeholder. Callers are responsible for injecting
                    the correct value: the pair-level rel_len_gap at fit
                    time, or the persisted training-set mean at inference
                    time for singletons.
    word_count      len(text.split())   (whitespace split, per Plan Mod 1)
    sent_count      max(1, number of sentence-terminator segments)
    avg_token_len   Mean character length across simple_tokens; 0.0 on
                    empty strings.
    type_token      len(set(tokens)) / max(1, len(tokens)) with safe
                    handling of empty strings.
    punc_rate       count of characters in PUNC_CHARS ("?!,") divided
                    by len(text); 0.0 on empty strings. Matches the
                    definition used when audits/truthfulqa_style_audit.csv
                    was first built.

FEATURE_COLS gives the canonical column order for the 10 features.
"""
from __future__ import annotations

import re
from typing import Dict

import numpy as np


# --- Lexicons (verbatim from scripts/build_audit_notebook.py) ---------------
#
# The notebook-builder script embeds these inside string literals of cells;
# they are replicated here so surface_features_text.py has no runtime
# dependency on importing that script. Any change upstream should be
# mirrored here and verified by re-running scripts/verify_extract_surface10.py.
NEG_LEADS: list[str] = [
    r"no\b", r"not\b",
    r"there is no\b", r"there are no\b", r"there's no\b",
    r"none\b", r"never\b", r"nothing\b", r"nobody\b",
]
NEG_WORDS: list[str] = [
    r"\bno\b", r"\bnot\b", r"\bnever\b",
    r"\bnone\b", r"\bnothing\b", r"\bnobody\b",
    r"\bcannot\b", r"\bcan't\b",
    r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b",
    r"\bwon't\b", r"\bwasn't\b", r"\bweren't\b",
]
HEDGES: list[str] = [
    r"\bmight\b", r"\bcould\b", r"\bmay\b",
    r"\bgenerally\b", r"\btypically\b", r"\boften\b",
    r"\busually\b", r"\bsometimes\b", r"\blikely\b",
    r"\bprobably\b", r"\bpossibly\b",
    r"\bappears to\b", r"\bseems\b", r"\bsuggests\b",
    r"\broughly\b", r"\bapproximately\b", r"\bare expected to\b",
]
AUTH_CUES: list[str] = [
    r"according to", r"experts say", r"scientists say",
    r"peer[- ]?reviewed", r"study shows",
]

# Punctuation whose density is tracked by punc_rate. Matches the original
# definition (PUNC_CHARS = "?!,") in scripts/build_audit_notebook.py.
PUNC_CHARS = "?!,"

FEATURE_COLS: list[str] = [
    "neg_lead",
    "neg_cnt",
    "hedge_rate",
    "auth_rate",
    "len_gap",
    "word_count",
    "sent_count",
    "avg_token_len",
    "type_token",
    "punc_rate",
]

# Surface6 (current revision feature set). Drops auth_rate, len_gap, sent_count,
# punc_rate from surface10. Order matches the paper's Table 1 spec.
FEATURE_COLS_SURFACE6: list[str] = [
    "word_count",
    "neg_cnt",
    "avg_token_len",
    "type_token",
    "neg_lead",
    "hedge_rate",
]

_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _simple_tokens(text: str) -> list[str]:
    return _SIMPLE_TOKEN_RE.findall(str(text).lower())


def _count_pattern_matches(patterns: list[str], text_lower: str) -> int:
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def starts_with_neg_lead(text: str) -> float:
    """1.0 if any NEG_LEADS regex matches at the start of
    `str(text).strip().lower()`. Matches the notebook's starts_with_any
    helper exactly (whitespace-only strip, no quote stripping)."""
    t = str(text).strip().lower()
    if not t:
        return 0.0
    for p in NEG_LEADS:
        if re.match(p, t):
            return 1.0
    return 0.0


def sentence_count(text: str) -> int:
    s = str(text).strip()
    if not s:
        return 1
    segments = [seg for seg in _SENT_SPLIT_RE.split(s) if seg.strip()]
    return max(1, len(segments))


def extract_surface10(text: str) -> Dict[str, float]:
    """Return all 10 surface10 features for a single answer string.

    DEPRECATED for new analyses: the current revision uses surface6
    (`extract_surface6` / `FEATURE_COLS_SURFACE6`). This function is
    retained for the frozen surface10 artifacts (audit CSV, n=135 cohort,
    bilateral probe) and for the side-by-side appendix.

    `len_gap` is a pair-level feature and cannot be determined from a
    single text. It is returned as 0.0 so that the output dict always has
    the same 10 keys; callers must override it with (a) the true pair-
    level rel_len_gap at feature-rebuild time, or (b) the persisted
    training-set mean at singleton scoring time.
    """
    text = "" if text is None else str(text)
    text_lower = text.lower()

    split_words = text.split()
    word_count = len(split_words)
    wc_denom = max(1, word_count)

    tokens = _simple_tokens(text)
    n_tok = len(tokens)

    neg_lead = starts_with_neg_lead(text)
    neg_cnt = float(_count_pattern_matches(NEG_WORDS, text_lower))

    hedge_cnt = _count_pattern_matches(HEDGES, text_lower)
    auth_cnt = _count_pattern_matches(AUTH_CUES, text_lower)
    hedge_rate = hedge_cnt / wc_denom
    auth_rate = auth_cnt / wc_denom

    sent_count = sentence_count(text)

    if tokens:
        avg_token_len = float(np.mean([len(t) for t in tokens]))
        type_token = len(set(tokens)) / max(1, n_tok)
    else:
        avg_token_len = 0.0
        type_token = 0.0

    char_len = len(text)
    if char_len > 0:
        punc_rate = sum(text.count(c) for c in PUNC_CHARS) / char_len
    else:
        punc_rate = 0.0

    return {
        "neg_lead": float(neg_lead),
        "neg_cnt": float(neg_cnt),
        "hedge_rate": float(hedge_rate),
        "auth_rate": float(auth_rate),
        "len_gap": 0.0,
        "word_count": float(word_count),
        "sent_count": float(sent_count),
        "avg_token_len": float(avg_token_len),
        "type_token": float(type_token),
        "punc_rate": float(punc_rate),
    }


def rel_len_gap(text_true: str, text_false: str) -> float:
    """Pair-level length gap as defined in scripts/build_audit_notebook.py:

        |char_len(true) - char_len(false)| / max(char_len(true),
                                                 char_len(false), 1)

    Used by the rebuild script to fill in the `len_gap` column (identical
    value for both answer rows in a pair).
    """
    la = len(str(text_true))
    lb = len(str(text_false))
    return abs(la - lb) / max(la, lb, 1)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 — surface_features_text.py sanity smoke test")
    print("=" * 60)
    samples = [
        'Not all swans are white, according to ornithologists.',
        "The Great Wall of China is visible from low Earth orbit.",
        '"None of the listed options are correct."',
        "Nothing in particular happens if you break a mirror.",
        "",
    ]
    for s in samples:
        feats = extract_surface10(s)
        print(f"\nText: {s!r}")
        for k in FEATURE_COLS:
            print(f"  {k:15s} = {feats[k]:.6f}")
