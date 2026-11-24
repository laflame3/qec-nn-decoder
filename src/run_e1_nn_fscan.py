"""E1 core experiment: Transformer and MLP vs PyMatching across correlation fraction f.

p = 0.005 fixed (PLAN v1.2). For every f the test set is the same 10^6 shots
as run_e1_pm_fscan.py (same circuit, same seed). Every decoder is scored on
those shots, and each NN is compared with PM-informed and PM-correlated by a
paired exact McNemar test.

Hyper-parameters come from data/results/e1_tune.csv (best validation LER at
f = 0), frozen for all f. The script is resumable: finished (f, model, seed)
rows in e1_nn_fscan.csv are skipped.

Run from the project root:  ../../.venv/bin/python src/run_e1_nn_fscan.py
"""

import csv
import json
import pathlib
import sys
import warnings

import numpy as np
import stim
import torch
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from decoders import build_pm_decoders, decode_all  # noqa: E402
from model import (  # noqa: E402
    MLPDecoder, TransformerDecoder, default_device, detector_layout, n_params, predict, train,
)
from noise import correlated_memory_circuit  # noqa: E402

P = 0.005
F_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TEST = 1_000_000
N_VAL = 200_000
N_TRAIN = 10_000_000
TORCH_SEEDS = [0, 1, 2]
TEST_SEED_BASE = 20261121   # identical to run_e1_pm_fscan.py
VAL_SEED_BASE = 30000
TRAIN_SEED_BASE = 40000

root = pathlib.Path(__file__).resolve().parent.parent
res = root / "data" / "results"
pred_dir = res / "e1_preds"
pred_dir.mkdir(exist_ok=True)
model_dir = res / "e1_models"   # trained weights + training curve, for cached re-evaluation
model_dir.mkdir(exist_ok=True)
out_csv = res / "e1_nn_fscan.csv"


def best_hparams():
    with open(res / "e1_tune.csv") as fh:
        rows = list(csv.DictReader(fh))
    pick = {}
    for kind in ("transformer", "mlp"):
        r = min((r for r in rows if r["kind"] == kind), key=lambda r: float(r["best_val_ler"]))
        pick[kind] = {"n_layers": int(r["n_layers"]), "lr": float(r["lr"])}
    return pick


def mcnemar(nn_wrong, pm_wrong):
    b = int((nn_wrong & ~pm_wrong).sum())  # NN wrong, PM right
    c = int((~nn_wrong & pm_wrong).sum())  # NN right, PM wrong
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def done_keys():
    if not out_csv.exists():
        return set()
    with open(out_csv) as fh:
        return {(float(r["f"]), r["model"], int(r["seed"])) for r in csv.DictReader(fh)}


def append_row(row):
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


hp = best_hparams()
print("hyper-parameters:", hp, flush=True)
device = default_device()
done = done_keys()

for k, f in enumerate(F_VALUES):
    circuit, info = correlated_memory_circuit(P, f)
    stab, rnd = detector_layout(circuit)
    test_det, test_obs = circuit.compile_detector_sampler(seed=TEST_SEED_BASE + k).sample(
        N_TEST, separate_observables=True)
    truth = test_obs[:, 0]
    val_det, val_obs = circuit.compile_detector_sampler(seed=VAL_SEED_BASE + k).sample(
        N_VAL, separate_observables=True)
    val_obs = val_obs[:, 0]

    pm_wrong = {n: pred != truth for n, pred in decode_all(build_pm_decoders(P, f), test_det).items()}
    pm_file = pred_dir / f"pm_f{f:.2f}.npz"
    if not pm_file.exists():
        np.savez_compressed(pm_file, **{n: np.packbits(w) for n, w in pm_wrong.items()},
                            truth=np.packbits(truth))
    print(f"f={f:.2f} " + " ".join(f"{n}={w.mean():.5f}" for n, w in pm_wrong.items()), flush=True)

    for m_idx, kind in enumerate(("transformer", "mlp")):
        for seed in TORCH_SEEDS:
            if (f, kind, seed) in done:
                continue
            if kind == "transformer":
                model = TransformerDecoder(stab, rnd, n_layers=hp[kind]["n_layers"])
            else:
                model = MLPDecoder(circuit.num_detectors, n_layers=hp[kind]["n_layers"])
            print(f"f={f:.2f} {kind} seed={seed}", flush=True)
            hist = train(model, circuit, val_det, val_obs, n_train=N_TRAIN, lr=hp[kind]["lr"],
                         sample_seed=TRAIN_SEED_BASE + 100 * k + 10 * m_idx + seed,
                         torch_seed=seed, device=device, log=lambda s: print(s, flush=True))
            stem = f"{kind}_f{f:.2f}_s{seed}"
            torch.save({"state_dict": {k_: v.cpu() for k_, v in model.state_dict().items()},
                        "kind": kind, "hparams": hp[kind], "p": P, "f": f, "seed": seed},
                       model_dir / f"{stem}.pt")
            (model_dir / f"{stem}.history.json").write_text(json.dumps(hist, indent=1))
            nn_wrong = predict(model, test_det, device) != truth
            np.save(pred_dir / f"{stem}.npy", np.packbits(nn_wrong))
            ler = float(nn_wrong.mean())
            row = {"p": P, "f": f, "model": kind, "seed": seed, "params": n_params(model),
                   "n_train": N_TRAIN, "n_test": N_TEST, "ler": ler,
                   "ler_err": float(np.sqrt(ler * (1 - ler) / N_TEST)),
                   "best_val_ler": min(h["val_ler"] for h in hist), "train_time_s": hist[-1]["time_s"]}
            for name in ("PM-informed", "PM-correlated"):
                b, c, pval = mcnemar(nn_wrong, pm_wrong[name])
                tag = name.split("-")[1]
                row[f"ratio_vs_{tag}"] = ler / float(pm_wrong[name].mean())
                row[f"nn_only_wrong_vs_{tag}"] = b
                row[f"pm_only_wrong_vs_{tag}"] = c
                row[f"mcnemar_p_vs_{tag}"] = pval
            append_row(row)
            print(f"  -> LER={ler:.5f} ratio/informed={row['ratio_vs_informed']:.3f} "
                  f"ratio/correlated={row['ratio_vs_correlated']:.3f} "
                  f"p={row['mcnemar_p_vs_correlated']:.2g}", flush=True)

settings = {
    "script": "src/run_e1_nn_fscan.py", "p": P, "f_values": F_VALUES, "n_test": N_TEST,
    "n_val": N_VAL, "n_train": N_TRAIN, "torch_seeds": TORCH_SEEDS,
    "test_seed_base": TEST_SEED_BASE, "val_seed_base": VAL_SEED_BASE,
    "train_seed_formula": "TRAIN_SEED_BASE + 100*f_index + 10*model_index + torch_seed",
    "hyperparameters": hp, "batch": 1024, "device": str(device),
    "noise": "PLAN v1.2 §1.2 fixed budget, XX+YY", "mcnemar": "exact binomial, two-sided",
    "stim": stim.__version__, "torch": torch.__version__, "numpy": np.__version__,
}
(res / "e1_nn_fscan.settings.json").write_text(json.dumps(settings, indent=2))
print("done", flush=True)
