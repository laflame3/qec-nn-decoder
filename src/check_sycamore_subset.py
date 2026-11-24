"""Validate every d=3 instance of the Zenodo 11403595 subset and tabulate the
dataset's own correlated-matching baselines.

Usage (project root): ../../.venv/bin/python -I src/check_sycamore_subset.py "$PWD/src" data/results/sycamore_d3_google_baselines.csv
"""
import sys, json, csv; sys.path.insert(0, sys.argv[1])
from pathlib import Path
import numpy as np, stim
import sycamore_data as S
rows=[]; bad=[]
dems=["simple","correlations","rl_isolated_correlated_matching","rl_shared_correlated_matching"]
for meta_f in sorted(S.DATA_ROOT.glob("sample_*/d3_*/X/r*/metadata.json")):
    p=meta_f.parent; m=json.loads(meta_f.read_text())
    c=stim.Circuit.from_file(str(p/"circuit_ideal.stim"))
    nbytes=(c.num_detectors+7)//8
    if (p/"detection_events.b8").stat().st_size!=m["shots"]*nbytes or m["distance"]!=3: bad.append(str(p)); continue
    obs=stim.read_shot_data_file(path=str(p/"obs_flips_actual.b8"),format="b8",num_observables=1)[:,0]
    r={"sample":p.parts[-4],"patch":p.parts[-3],"rounds":m["rounds"],"shots":m["shots"],"detectors":c.num_detectors,"undecoded":obs.mean()}
    for d in dems:
        pr=stim.read_shot_data_file(path=str(p/"obs_flips_predicted"/f"dem_{d}"/"correlated_matching.b8"),format="b8",num_observables=1)[:,0]
        r[f"cm_{d}"]=np.mean(pr!=obs)
    rows.append(r)
print("instances ok:",len(rows),"bad:",len(bad))
out=Path(sys.argv[2]); out.parent.mkdir(parents=True,exist_ok=True)
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
by=collections.defaultdict(list)
for r in rows: by[r["rounds"]].append(r)
print("rounds  n   det  undecoded  CM+simple  CM+correlations  CM+rl_isolated  CM+rl_shared")
for k in sorted(by):
    g=by[k]; f=lambda key: np.mean([x[key] for x in g])
    print(f"{k:5d} {len(g):4d} {g[0]['detectors']:4d}   {f('undecoded'):.4f}    {f('cm_simple'):.4f}     {f('cm_correlations'):.4f}          {f('cm_rl_isolated_correlated_matching'):.4f}         {f('cm_rl_shared_correlated_matching'):.4f}")
