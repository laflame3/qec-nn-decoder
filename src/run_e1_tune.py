"""E1 hyper-parameter selection on independent noise only (f = 0, p = 0.005).

PLAN v1.2: tune once here, then freeze the choice for every noise setting.
Selection uses a validation set only; the E1 test sets are never touched.
Writes data/results/e1_tune.csv and e1_tune.settings.json.
Run from the project root:  ../../.venv/bin/python src/run_e1_tune.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np
import stim
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import (  # noqa: E402
    MLPDecoder, TransformerDecoder, default_device, detector_layout, n_params, train,
)
from noise import correlated_memory_circuit  # noqa: E402

P, F = 0.005, 0.0
N_TRAIN = 10_000_000
N_VAL = 200_000
VAL_SEED = 30000          # validation shots (f index 0)
TRAIN_SEED = 50000        # training-shot sampler seeds for tuning runs
CONFIGS = [
    ("transformer", {"n_layers": 2}, 3e-4),
    ("transformer", {"n_layers": 2}, 1e-3),
    ("transformer", {"n_layers": 3}, 3e-4),
    ("transformer", {"n_layers": 3}, 1e-3),
    ("mlp", {"n_layers": 3}, 3e-4),
    ("mlp", {"n_layers": 3}, 1e-3),
]

out_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"
circuit, _ = correlated_memory_circuit(P, F)
stab, rnd = detector_layout(circuit)
val_det, val_obs = circuit.compile_detector_sampler(seed=VAL_SEED).sample(
    N_VAL, separate_observables=True
)
val_obs = val_obs[:, 0]
device = default_device()

rows = []
for i, (kind, arch, lr) in enumerate(CONFIGS):
    if kind == "transformer":
        model = TransformerDecoder(stab, rnd, **arch)
    else:
        model = MLPDecoder(circuit.num_detectors, **arch)
    print(f"[{i}] {kind} {arch} lr={lr} params={n_params(model)}", flush=True)
    hist = train(model, circuit, val_det, val_obs, n_train=N_TRAIN, lr=lr,
                 sample_seed=TRAIN_SEED + i, torch_seed=0, device=device,
                 log=lambda s: print(s, flush=True))
    best = min(h["val_ler"] for h in hist)
    rows.append({"kind": kind, "n_layers": arch["n_layers"], "lr": lr,
                 "params": n_params(model), "best_val_ler": best,
                 "final_val_ler": hist[-1]["val_ler"], "time_s": hist[-1]["time_s"]})
    with open(out_dir / "e1_tune.csv", "w") as fh:
        cols = list(rows[0].keys())
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

settings = {
    "script": "src/run_e1_tune.py", "p": P, "f": F, "n_train": N_TRAIN, "n_val": N_VAL,
    "val_seed": VAL_SEED, "train_seed_base": TRAIN_SEED, "batch": 1024,
    "optimizer": "AdamW wd=1e-4, 2% warmup + cosine", "device": str(device),
    "transformer": "d_model 64, 4 heads, d_ff 128, CLS, pre-norm", "mlp": "hidden 256, GELU",
    "configs": [[k, a, lr] for k, a, lr in CONFIGS],
    "stim": stim.__version__, "torch": torch.__version__, "numpy": np.__version__,
}
(out_dir / "e1_tune.settings.json").write_text(json.dumps(settings, indent=2))
print("done", flush=True)
