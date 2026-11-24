"""Paired exact McNemar test, Transformer vs MLP, same f and seed, same test shots.

Reads data/results/e1_preds/*.npy written by run_e1_nn_fscan.py and writes
data/results/e1_tf_vs_mlp.csv. Run from the project root.
"""

import csv
import pathlib

import numpy as np
from scipy.stats import binomtest

F_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
N_TEST = 1_000_000
res = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"


def wrong(kind, f, s):
    return np.unpackbits(np.load(res / "e1_preds" / f"{kind}_f{f:.2f}_s{s}.npy"))[:N_TEST].astype(bool)


rows = []
for f in F_VALUES:
    for s in SEEDS:
        t, m = wrong("transformer", f, s), wrong("mlp", f, s)
        b, c = int((t & ~m).sum()), int((~t & m).sum())
        rows.append({"f": f, "seed": s, "tf_ler": t.mean(), "mlp_ler": m.mean(),
                     "ratio_tf_over_mlp": t.mean() / m.mean(), "tf_only_wrong": b,
                     "mlp_only_wrong": c, "mcnemar_p": binomtest(b, b + c, 0.5).pvalue})
with open(res / "e1_tf_vs_mlp.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
