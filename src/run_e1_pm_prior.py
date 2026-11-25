"""E1 v2 experiment 3, extra control: PM-correlated given the wrong (f = 0) prior.

Added after the experiment-3 results were seen (student decision, 2026-11-24), so it is
a post-hoc control. It decodes the existing f = 1 test shots with the DEM of the f = 0
circuit and compares (a) this PM with the true-prior PM and (b) the experiment-3 networks
with this PM. No training. Existing result files are only read.

Run from the project root:  ../../.venv/bin/python -B src/run_e1_pm_prior.py
"""

import csv
import json
import platform

import numpy as np
import pymatching
import stim

from decoders import build_pm_correlated, decode_pm_correlated
from e1v2_common import RESULTS, STAT_FIELDS, TARGET_TEST_SEED, N_TEST, paired_stats, unpack_bits
from noise import correlated_memory_circuit

OUT_CSV = RESULTS / "e1_pm_prior.csv"
OUT_SETTINGS = RESULTS / "e1_pm_prior.settings.json"
OUT_MASK = RESULTS / "e1v2_preds" / "pm_prior_f0_on_f1.npy"
NN_METHODS = ("zero_shot", "scratch_45k", "finetune_45k", "full_1e7")


def main():
    for path in (OUT_CSV, OUT_SETTINGS, OUT_MASK):
        if path.exists():
            raise FileExistsError(f"{path} exists; this script runs once and never overwrites")

    target, _ = correlated_memory_circuit(0.005, 1.0)
    prior, _ = correlated_memory_circuit(0.005, 0.0)
    det, obs = target.compile_detector_sampler(seed=TARGET_TEST_SEED).sample(N_TEST, separate_observables=True)
    truth = obs[:, 0]
    with np.load(RESULTS / "e1_preds" / "pm_f1.00.npz") as saved:
        if not (unpack_bits(saved["truth"], N_TEST) == truth).all():
            raise ValueError("re-sampled f = 1 test truth differs from the saved one")
        pm_true = unpack_bits(saved["PM-correlated"], N_TEST)

    pm_prior = decode_pm_correlated(build_pm_correlated(prior), det) != truth

    rows = [{"decoder": "PM-correlated, f=0 DEM", "reference": "PM-correlated, f=1 DEM",
             **paired_stats(pm_prior, pm_true)}]
    for kind in ("transformer", "mlp"):
        for method in NN_METHODS:
            wrong = unpack_bits(np.load(RESULTS / "e1v2_preds" / f"finetune_{kind}_{method}_s0.npy"), N_TEST)
            rows.append({"decoder": f"{kind} {method}", "reference": "PM-correlated, f=0 DEM",
                         **paired_stats(wrong, pm_prior)})

    settings = {
        "script": "src/run_e1_pm_prior.py", "status": "post-hoc control, added after results were seen",
        "decided": "2026-11-24, student", "target": "correlated_memory_circuit(0.005, 1.0)",
        "prior": "correlated_memory_circuit(0.005, 0.0)", "decoder": "build_pm_correlated(prior circuit)",
        "test_seed": TARGET_TEST_SEED, "n_test": N_TEST, "nn_masks": "e1v2_preds/finetune_{kind}_{method}_s0.npy",
        "python": platform.python_version(), "stim": stim.__version__,
        "pymatching": pymatching.__version__, "numpy": np.__version__,
    }
    OUT_SETTINGS.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    np.save(OUT_MASK, np.packbits(pm_prior))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["decoder", "reference", *STAT_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['decoder']:28s} vs {row['reference']:24s} P_L={row['pl']:.5f} "
              f"ratio={row['ratio']:.4f} [{row['ratio_ci_low']:.4f}, {row['ratio_ci_high']:.4f}] "
              f"delta={100 * row['delta']:+.3f} pp", flush=True)


if __name__ == "__main__":
    main()
