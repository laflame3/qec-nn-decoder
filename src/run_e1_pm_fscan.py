"""E1 smoke run: three PyMatching tiers across correlation fraction f at fixed p.

Writes data/results/e1_pm_fscan.csv and a matching .settings.json.
Run from the project root:  ../../.venv/bin/python src/run_e1_pm_fscan.py
"""

import json
import pathlib
import sys
import time

import numpy as np
import pymatching
import stim

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from decoders import build_pm_decoders, decode_all, edge_conflicts  # noqa: E402
from noise import correlated_memory_circuit, hyperedge_fraction  # noqa: E402

P = 0.005
F_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
SHOTS = 1_000_000
SEED_BASE = 20261121  # test-set seeds; training sets will use different seeds

out_dir = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"
rows = []
for k, f in enumerate(F_VALUES):
    circuit, info = correlated_memory_circuit(P, f)
    h = hyperedge_fraction(circuit.detector_error_model())
    seed = SEED_BASE + k
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(
        SHOTS, separate_observables=True
    )
    truth = obs[:, 0]
    decoders = build_pm_decoders(P, f)
    t0 = time.time()
    preds = decode_all(decoders, det)
    dt = time.time() - t0
    row = {"p": P, "f": f, "h": h, "relabelled_mass": edge_conflicts(circuit),
           "shots": SHOTS, "seed": seed,
           "no_decode": float(truth.mean())}
    for name, pred in preds.items():
        n_err = int((pred != truth).sum())
        ler = n_err / SHOTS
        row[name] = ler
        row[name + "_err"] = float(np.sqrt(ler * (1 - ler) / SHOTS))
    rows.append(row)
    print(f"f={f:.2f} h={h:.3f} relab={row['relabelled_mass']:.4f} " + " ".join(
        f"{n}={row[n]:.5f}±{row[n + '_err']:.5f}" for n in preds) + f"  ({dt:.0f}s)")

cols = list(rows[0].keys())
with open(out_dir / "e1_pm_fscan.csv", "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join(str(r[c]) for c in cols) + "\n")
settings = {
    "script": "src/run_e1_pm_fscan.py",
    "p": P, "f_values": F_VALUES, "shots": SHOTS, "seed_base": SEED_BASE,
    "distance": 3, "rounds": 3, "basis": "Z",
    "noise": "PLAN v1.2 §1.2: fixed budget 9p per round, XX+YY on 12 adjacent pairs",
    "stim": stim.__version__, "pymatching": pymatching.__version__,
    "numpy": np.__version__,
}
(out_dir / "e1_pm_fscan.settings.json").write_text(json.dumps(settings, indent=2))
