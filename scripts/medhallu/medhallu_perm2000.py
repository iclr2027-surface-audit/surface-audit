import json,sys,pandas as pd,numpy as np
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__)); REPO=_os.path.abspath(_os.path.join(_HERE,"..",".."))
A=_HERE+"/"; S=_os.environ.get("MEDHALLU_WORKDIR",_HERE).rstrip("/")+"/"; REL=REPO+"/hf_release/"
_sys.path.insert(0,_HERE); _sys.path.insert(0,REPO+"/scripts")
from canon_audit import audit
pq=pd.read_csv(A+"medhallu_panel/medhallu_pairs_with_questions.csv").set_index("pair_id")
out={}
for name,ids in (("MedHallu full",list(pq.index)),):
    rows=pd.DataFrame({"text":list(pq.loc[ids,"text_true"])+list(pq.loc[ids,"text_false"]),"label":[1]*len(ids)+[0]*len(ids),"group":list(ids)*2})
    r=audit(rows,name,"paired",B=2000,n_boot=2000); out[name]={k:r[k] for k in ["n_rows","auc","auc_ci95","accuracy","word_count_only_auc","perm_B","null_exceedances","perm_p","null_mean","null_sd","seconds"]}
    print(name,json.dumps(out[name]),flush=True); json.dump(out,open(S+"medhallu_perm2000.json","w"),indent=1)
print("DONE")
