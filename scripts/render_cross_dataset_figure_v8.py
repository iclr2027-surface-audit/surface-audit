#!/usr/bin/env python3
# Figure 3 of the paper: paper_assets/figures/surface6-datasets_8.pdf.
# 15 bars: results/t4_cross_dataset_surface6.json (FeverSymmetric excluded) plus the full-MedHallu audit row
# from results/medhallu_figure_rows.json; TruthfulQA-476 is starred.
# Run:  MEDHALLU_AUDIT_ONLY=1 python scripts/render_cross_dataset_figure_v8.py
#       python scripts/crop_pdf_margins.py paper_assets/figures/surface6-datasets_8.pdf
"""Figure 3: cross-dataset Surface6 audit (grouped 5-fold CV AUC per benchmark)."""
from __future__ import annotations
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


def light(c, f=0.52):
    """Blend colour c toward white by fraction f (lightness channel for the Acc bars)."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "paper_assets" / "figures"
SUFFIX = "8"   # _4 = 14-bar version without the two extra releases

MAJ = {"TruthfulQA": 0.5, "TruthfulQA-Audited": 0.5, "HaluEval QA": 0.5, "HaluEval-645": 0.5, "MedHallu": 0.5, "MedHallu-3101": 0.5,
       "PIQA": 0.5, "FEVER": 0.5,
       "ANLI": max(1070, 2132 - 1070) / 2132, "SNLI": max(3329, 6607 - 3329) / 6607,
       "MultiNLI": max(3479, 6692 - 3479) / 6692, "RTE": max(146, 277 - 146) / 277,
       "MultiRC": max(2075, 4848 - 2075) / 4848, "VitaminC": 31484 / 54012,
       "BoolQ": 0.6217, "OpenBookQA": 1500 / 2000, "SelfCheckGPT": max(516, 1908 - 516) / 1908}
CAT = {"Adversarial/hallucination": "#0A5FE0",   # our blue - fixed (the family both released sets belong to)
       "Fact verification": "#2E8B72",            # muted bluish-green (Okabe-Ito #009E73)
       "NLI hypothesis-only": "#E69F00",          # amber (Okabe-Ito) - the blue complement
       "Multiple-choice/cloze": "#B06B92"}        # muted reddish-purple (Okabe-Ito #CC79A7)
DISP = {"TruthfulQA-Audited": "TruthfulQA-476"}
OURS = {"TruthfulQA-476", "HaluEval-645", "MedHallu-3101"}
OURS_C = "#0A5FE0"   # same blue as the hallucination bars
STAR_C = "#C62828"   # star stays red


def main() -> None:
    rows = [r for r in json.load(open(REPO / "results" / "t4_cross_dataset_surface6.json"))["rows"]
            if r["dataset_id"] != "FeverSymmetric"]
    if os.environ.get("MEDHALLU_AUDIT_ONLY"):       # MEDHALLU_AUDIT_ONLY=1 -> 15 bars: + full MedHallu audit row only
        rows.extend(r for r in json.load(open(REPO / "results" / "medhallu_figure_rows.json")) if r["dataset_id"] == "MedHallu")
    elif not os.environ.get("TQA476_ONLY"):        # TQA476_ONLY=1 -> 14 bars, no HaluEval-645 / MedHallu rows
        rows.append(json.load(open(REPO / "results" / "halueval645_figure_row.json")))
        rows.extend(json.load(open(REPO / "results" / "medhallu_figure_rows.json")))
    rows = sorted(rows, key=lambda r: r["auc"], reverse=True)

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    bw = 0.34
    for i, r in enumerate(rows):
        c = CAT[r["category"]]
        ax.bar(i - bw / 2, r["auc"], width=bw, color=c, edgecolor="white", linewidth=0.4, zorder=2,
               yerr=[[max(r["auc"] - r["auc_ci95_lo"], 0)], [max(r["auc_ci95_hi"] - r["auc"], 0)]],
               error_kw=dict(ecolor="#333333", elinewidth=0.9, capsize=1.8))
        ax.bar(i + bw / 2, r["acc"], width=bw, color=c, alpha=0.55, hatch="///",
               edgecolor="white", linewidth=0.5, zorder=2)
        ax.plot(i - bw / 2, r["null_auc_mean"], marker="x", color="black",
                markersize=5, markeredgewidth=1.4, zorder=4)
        ax.plot(i + bw / 2, MAJ[r["dataset_id"]], marker="_", color="black",
                markersize=9, markeredgewidth=1.6, zorder=4)
    names = [DISP.get(r["dataset_id"], r["dataset_id"]) for r in rows]
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=13)
    for lbl, n in zip(ax.get_xticklabels(), names):
        if n in OURS:
            lbl.set_color(OURS_C); lbl.set_fontweight("bold")
    ax.set_ylim(0.35, 0.995); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("AUC / Acc", fontsize=14); ax.tick_params(axis="y", labelsize=13)
    ax.set_title("Cross-dataset surface-form audit (SURFACE6)", fontsize=16, pad=3)
    ax.yaxis.grid(True, linestyle=":", color="#BBBBBB", zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = [Patch(facecolor="#888888", edgecolor="white", label="Solid = AUC"),
           Patch(facecolor="#888888", alpha=0.55, hatch="///", edgecolor="white", label="Hatched = Accuracy"),
           Line2D([], [], marker="x", color="black", linestyle="none", label="AUC null mean"),
           Line2D([], [], marker="_", color="black", linestyle="none", label="Majority baseline (Acc)")]
    leg += [Patch(facecolor=c, edgecolor="white",
                  label=k.replace("Adversarial/hallucination", "Hallucination")
                         .replace("Multiple-choice/cloze", "Multi-choice/cloze"))
            for k, c in CAT.items()]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(1.0, 0.97), fontsize=11,
              framealpha=0.95, ncol=2, borderpad=0.35, labelspacing=0.35)
    fig.tight_layout()
    # overlay the leading star in red (the rest of the label stays green)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for lbl, n in zip(ax.get_xticklabels(), names):
        if n not in OURS:
            continue
        bb = lbl.get_window_extent(renderer=rend)
        fx, fy = inv.transform((bb.x0 + 8.0, bb.y0 + 6.0))
        fig.text(fx, fy, "$\\bigstar$ ", color=STAR_C, fontsize=13, fontweight="bold",
                 rotation=45, ha="right", va="bottom", rotation_mode="anchor", zorder=10)

    fig.savefig(OUT_DIR / f"surface6-datasets_{SUFFIX}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"surface6-datasets_{SUFFIX}.png", dpi=200, bbox_inches="tight")
    print("[render] wrote", OUT_DIR / f"surface6-datasets_{SUFFIX}.pdf")


if __name__ == "__main__":
    main()
