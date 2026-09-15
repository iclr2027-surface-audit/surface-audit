#!/usr/bin/env python3
"""Remap batch12 raw-JSON ids 1..20 -> 12001..12020 in place."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "audits" / "_eval_dataset_v3_batch12_raw.json"
OFFSET = 12000

def main():
    data = json.loads(RAW.read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    bumped = 0
    for r in rows:
        old = int(r.get("id", 0))
        if old < OFFSET:
            r["id"] = old + OFFSET
            bumped += 1
    out = data if isinstance(data, dict) else rows
    if isinstance(out, dict):
        out["rows"] = rows
    RAW.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Remapped {bumped}/{len(rows)} ids by +{OFFSET}; new range = [{min(r['id'] for r in rows)}, {max(r['id'] for r in rows)}]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
