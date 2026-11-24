"""Collect every model's training curve and summary into two CSV files (for the public repository).

Reads data/results/e{1,2}_models/*.history.json and the result tables; writes
data/results/training_history.csv (one row per validation check) and
data/results/models_summary.csv (one row per trained model). No training is run.
Run from the project root:  ../../.venv/bin/python src/export_training_history.py
"""

import json
import pathlib
import re

import pandas as pd

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"

rows = []
for exp in ("e1", "e2"):
    for path in sorted((RESULTS / f"{exp}_models").glob("*.history.json")):
        stem = path.name.removesuffix(".history.json")
        if exp == "e1":
            model, f, seed = re.fullmatch(r"(transformer|mlp)_f([\d.]+)_s(\d)", stem).groups()
            key = {"f": float(f), "regime": "simulated"}
        else:
            model, regime, seed = re.fullmatch(r"(transformer|mlp)_(\w+?)_s(\d)", stem).groups()
            key = {"f": None, "regime": regime}
        for i, h in enumerate(json.loads(path.read_text())):
            rows.append({"experiment": exp.upper(), "model": model, **key, "seed": int(seed), "check": i,
                         "shots_seen": h.get("seen"), "epoch": h.get("epoch"), "train_loss": h.get("train_loss"),
                         "val_ler": h["val_ler"], "time_s": h["time_s"]})
hist = pd.DataFrame(rows)
hist.to_csv(RESULTS / "training_history.csv", index=False)

e1 = pd.read_csv(RESULTS / "e1_nn_fscan.csv")
e2 = pd.read_csv(RESULTS / "e2_nn.csv")
summary = pd.concat([
    pd.DataFrame({"experiment": "E1", "model": e1.model, "f": e1.f, "regime": "simulated", "seed": e1.seed,
                  "params": e1.params, "training_shots_simulated": e1.n_train, "real_training_shots": 0,
                  "best_val_ler": e1.best_val_ler, "test_ler": e1.ler, "test_shots": e1.n_test,
                  "train_time_s": e1.train_time_s}),
    pd.DataFrame({"experiment": "E2", "model": e2.model, "f": None, "regime": e2.regime, "seed": e2.seed,
                  "params": e2.n_params, "training_shots_simulated": e2.n_pretrain,
                  "real_training_shots": e2.regime.map({"real": 45000, "ft": 45000, "ftcal": 45000}).fillna(0).astype(int),
                  "best_val_ler": e2.best_val_ler, "test_ler": e2.ler_test, "test_shots": 22500,
                  "train_time_s": e2.train_time_s}),
], ignore_index=True)
summary.to_csv(RESULTS / "models_summary.csv", index=False)
print(f"training_history.csv: {len(hist)} rows from {hist.groupby(['experiment', 'model', 'regime', 'f', 'seed'], dropna=False).ngroups} models; "
      f"models_summary.csv: {len(summary)} models")
