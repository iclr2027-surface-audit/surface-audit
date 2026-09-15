import json, os
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); REPO=_os.path.abspath(_os.path.join(_HERE,"..",".."))
A=_HERE+"/"; S=_os.environ.get("MEDHALLU_WORKDIR",_HERE).rstrip("/")+"/"; REL=REPO+"/hf_release/"
_sys.path.insert(0,_HERE); _sys.path.insert(0,REPO+"/scripts")
R=REPO+"/"
perm=json.load(open(S+"medhallu_perm2000.json")) if os.path.exists(S+"medhallu_perm2000.json") else {}
full=perm.get("MedHallu full")
def row(did,c):
    return {"dataset_id":did,"category":"Adversarial/hallucination","n_texts":c["n_rows"],"auc":round(c["auc"],4),"acc":round(c["accuracy"],4),
            "auc_ci95_lo":round(c["auc_ci95"][0],4),"auc_ci95_hi":round(c["auc_ci95"][1],4),"null_auc_mean":round(c["null_mean"],4),"null_auc_sd":round(c["null_sd"],4),
            "perm_p":round(c["perm_p"],5),"perm_B":c["perm_B"],"null_exceedances":c.get("null_exceedances"),"delta_auc":round(c["auc"]-c["null_mean"],4),
            "protocol":"canon_audit.py: Surface6 (scripts/surface_features_text.py), GroupKFold(5) by pair, StandardScaler+LR liblinear seed 42, pooled OOF; within-pair permutation null with refit; pair cluster bootstrap"}
rows=[row("MedHallu",full)]
json.dump(rows,open(R+"results/medhallu_figure_rows.json","w"),indent=1); print(json.dumps(rows,indent=0)[:900]); print("full perm_B:",full.get("perm_B"))
