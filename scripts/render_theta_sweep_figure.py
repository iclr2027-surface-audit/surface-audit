#!/usr/bin/env python3
"""Theta-sweep figure v4 — continuous axis 0.50-0.715: frozen sweep (solid) + recomputed fill 0.66-0.71 (open diamonds) + full-790 endpoint."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

TH  = [round(0.50+k*0.01,2) for k in range(16)]
AP_N   = [335,401,408,476,493,492,536,497,588,596,598,614,625,640,656,668]
AP_ACC = [0.4940,0.5262,0.5184,0.5221,0.5436,0.5457,0.5466,0.5523,0.5808,0.5889,0.5970,0.6059,0.6184,0.6312,0.6402,0.6422]
FP_TH = [round(0.59+k*0.01,2) for k in range(13)]
FP_N   = [471,605,614,622,627,646,673,682,697,718,739,760,781]
FP_ACC = [0.5488,0.5950,0.5953,0.6101,0.6196,0.6324,0.6508,0.6518,0.6514,0.6713,0.6752,0.6816,0.6863]
AP_ACC_EXT = [0.6536,0.6593,0.6671,0.6745,0.6860,0.6861]
FULL_ACC, FLOOR_ACC = 0.6892, 0.5600
RC_TH = [0.66,0.67,0.68,0.69,0.70,0.71]          # recomputed, current pipeline
RC_N  = [687,697,721,745,766,782]
FULL_TH, FULL_N = 0.7145, 790
AP_C, FP_C, STAR_C, WALL = "#1F77B4", "#FF7F0E", "#C62828", 0.5826

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.4, 5.8), sharex=True,
                              gridspec_kw={"height_ratios":[2.2,1.15], "hspace":0.38})
for a in (ax, ax2):
    a.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#E3E6EA", zorder=0)
    a.set_axisbelow(True)
    a.axvline(WALL, color="#8A8A8A", linestyle=(0,(1,2.2)), linewidth=1.5, zorder=1)
    for s in ("top","right"): a.spines[s].set_visible(False)
    for s in ("left","bottom"): a.spines[s].set_color("#9AA0A6")
    a.tick_params(colors="black", labelsize=11.5)
    a.set_xlim(0.4945, 0.7215)

# top: N retained
ax.plot(TH, AP_N, "-", color=AP_C, linewidth=2.4, zorder=3)
ax.plot(TH, AP_N, "o", color=AP_C, markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=4)
ax.plot([WALL]+FP_TH, [375]+FP_N, "--", color=FP_C, linewidth=2.0, zorder=3)
ax.plot([WALL], [375], "s", color=FP_C, markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=4)
ax.plot(FP_TH, FP_N, "s", color=FP_C, markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=4)
ax.plot([0.65]+RC_TH, [668]+RC_N, "-", color=AP_C, linewidth=2.4, zorder=3)
ax.plot(RC_TH, RC_N, "o", color=AP_C, markersize=6, markeredgecolor="white", markeredgewidth=1.0, zorder=4)
ax.plot([RC_TH[-1], FULL_TH], [RC_N[-1], FULL_N], linestyle=":", color=AP_C, linewidth=1.7, zorder=2)
ax.plot([FP_TH[-1], FULL_TH], [FP_N[-1], FULL_N], linestyle=":", color=FP_C, linewidth=1.6, zorder=2)
ax.plot([FULL_TH], [FULL_N], marker="o", markersize=7.5, color=STAR_C, markeredgecolor="white", markeredgewidth=1.1, linestyle="none", zorder=5, clip_on=False)
ax.plot([0.53],[476], marker="*", markersize=16, color=STAR_C, markeredgecolor="white", markeredgewidth=1.0, linestyle="none", zorder=5)
ax.text(0.53, 172, "TruthfulQA-476", fontsize=11.5, color="#8A8A8A", ha="center", va="top")
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyArrowPatch
ax.add_patch(FancyArrowPatch(posA=(0.53, 186), posB=(0.53, 448), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#8A8A8A", fill=False, zorder=5))
ax.annotate("baseline cannot\nreach below this AUC", xy=(WALL+0.0015, 375), xytext=(0.6165, 375),
            fontsize=10.5, color="#8A8A8A", ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color="#8A8A8A", lw=1.0, mutation_scale=9, shrinkA=4, shrinkB=4))


ax.text(0.7145, 470, "TruthfulQA", fontsize=11.5, color="#8A8A8A", ha="center", va="top")
ax.add_patch(FancyArrowPatch(posA=(0.7145, 500), posB=(0.7145, 748), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#8A8A8A", fill=False, zorder=5))
ax.set_ylabel("Pairs retained  $N$", fontsize=13.5, color="black")
ax.set_ylim(0, 800); ax.set_yticks([0,100,200,300,400,500,600,700,800])
ax.set_title("$\\theta$-sweep (TruthfulQA)", fontsize=14.5, color="black", pad=10)

# bottom: rho
ax2.plot(TH, AP_ACC, "-", color=AP_C, linewidth=2.2, zorder=3, clip_on=False)
ax2.plot(TH, AP_ACC, "o", color=AP_C, markersize=5, markeredgecolor="white", markeredgewidth=0.9, zorder=4, clip_on=False)
ax2.plot([WALL]+FP_TH, [FLOOR_ACC]+FP_ACC, "--", color=FP_C, linewidth=1.9, zorder=3, clip_on=False)
ax2.plot([WALL], [FLOOR_ACC], "s", color=FP_C, markersize=5, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
ax2.plot(FP_TH, FP_ACC, "s", color=FP_C, markersize=5, markeredgecolor="white", markeredgewidth=0.9, zorder=4, clip_on=False)
ax2.plot([0.53],[0.5221], marker="*", markersize=13, color=STAR_C, markeredgecolor="white", markeredgewidth=0.9, linestyle="none", zorder=5)
ax2.plot([0.65]+RC_TH, [0.6422]+AP_ACC_EXT, "-", color=AP_C, linewidth=2.2, zorder=3, clip_on=False)
ax2.plot(RC_TH, AP_ACC_EXT, "o", color=AP_C, markersize=5, markeredgecolor="white", markeredgewidth=0.9, zorder=4, clip_on=False)
ax2.plot([RC_TH[-1], FULL_TH],[AP_ACC_EXT[-1], FULL_ACC], linestyle=":", color=AP_C, linewidth=1.5, zorder=2, clip_on=False)
ax2.plot([FP_TH[-1], FULL_TH],[FP_ACC[-1], FULL_ACC], linestyle=":", color=FP_C, linewidth=1.4, zorder=2, clip_on=False)
ax2.plot([FULL_TH],[FULL_ACC], marker="o", markersize=6.5, color=STAR_C, markeredgecolor="white", markeredgewidth=1.0, linestyle="none", zorder=5, clip_on=False)
ax2.axhline(0.5, color="#999999", linewidth=0.9, linestyle=(0,(3,3)), zorder=1)
ax2.text(0.7205, 0.503, "chance", fontsize=9, color="#777777", ha="right", va="bottom")
ax2.set_ylabel("Audit accuracy", fontsize=13, color="black")
ax2.set_ylim(0.478, 0.712); ax2.set_yticks([0.50,0.55,0.60,0.65,0.70])
XT_ALL = TH + RC_TH
XT_MAJ = [x for x in XT_ALL if round(x*100) % 2 == 0 and abs(x-0.58) > 1e-9]
TICKS = sorted(XT_MAJ + [WALL])
LBLS = [("0.583" if abs(x-WALL) < 1e-9 else f"{x:.2f}") for x in TICKS]
ax2.set_xticks(TICKS)
ax.tick_params(labelbottom=True)
ax.set_xlabel("Audit threshold $\\theta$ (AUC)", fontsize=12, color="black")
ax2.set_xticklabels(LBLS, fontsize=11)
for _a in (ax, ax2):
    for _lbl in _a.get_xticklabels():
        if _lbl.get_text() == "0.583":
            _lbl.set_color("black"); _lbl.set_fontweight("bold"); _lbl.set_fontsize(9.5)
ax2.set_xticks(XT_ALL, minor=True)
ax2.set_xlabel("Audit threshold $\\theta$ (AUC)", fontsize=13, color="black")

handles = [Line2D([], [], color=AP_C, marker="o", markeredgecolor="white", linewidth=2.4, markersize=7, label="Audit-Prune"),
           Line2D([], [], color=FP_C, marker="s", markeredgecolor="white", linestyle="--", linewidth=2.0, markersize=7, label="Fixed-prefix baseline"),
           Line2D([], [], marker="*", markersize=14, color=STAR_C, markeredgecolor="white", markeredgewidth=0.9, linestyle="none")]
from matplotlib.legend_handler import HandlerTuple
fig.legend(handles=handles, labels=["Audit-Prune", "Fixed-prefix baseline", "Released TruthfulQA-476"],
           handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower center", bbox_to_anchor=(0.5, -0.045),
           ncol=3, fontsize=10.5, frameon=False, columnspacing=1.1, handletextpad=0.5)
import pathlib; fig.savefig(str(pathlib.Path(__file__).parent/"theta_sweep_figure.pdf"), bbox_inches="tight")
print("v4 continuous written")
