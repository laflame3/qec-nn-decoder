"""Evaluate a sampled syndrome lookup decoder on the saved E1 test shots.

Run from the project root: ../../.venv/bin/python src/run_e1_lut.py
"""

import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import stim
from scipy.stats import binomtest

from noise import correlated_memory_circuit

P = 0.005
F_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
NN_MODELS = ("transformer", "mlp")
NN_SEEDS = (0, 1, 2)
N_TEST = 1_000_000
TEST_SEED_BASE = 20261121  # Same test sampler as run_e1_nn_fscan.py.
# Other E1 sampler seeds: validation 30000–30004, scan training
# 40000–40412, tuning training 50000–50005, testing 20261121–20261125.
LUT_SEED_BASE = 60000
N_BATCHES = 20
BATCH_SHOTS = 1_000_000
N_DETECTORS = 24
RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
PREDICTIONS = RESULTS / "e1_preds"


def unpack_test_bits(packed):
    return np.unpackbits(packed, count=N_TEST).astype(bool)


def syndrome_indices(detectors):
    """Encode detector j as bit j in each 24-bit lookup index."""
    packed = np.packbits(detectors, axis=1, bitorder="little").astype(np.uint32)
    return packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)


def build_counts(circuit, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    total = np.zeros(1 << N_DETECTORS, dtype=np.uint32)
    ones = np.zeros_like(total)
    for _ in range(N_BATCHES):
        detectors, observables = sampler.sample(BATCH_SHOTS, separate_observables=True)
        indices = syndrome_indices(detectors)
        np.add.at(total, indices, 1)
        np.add.at(ones, indices[observables[:, 0]], 1)
    return total, ones


def evaluate(f, k):
    circuit, _ = correlated_memory_circuit(P, f)
    assert circuit.num_detectors == N_DETECTORS and circuit.num_observables == 1
    detectors, observables = circuit.compile_detector_sampler(seed=TEST_SEED_BASE + k).sample(
        N_TEST, separate_observables=True
    )
    truth = observables[:, 0]
    with np.load(PREDICTIONS / f"pm_f{f:.2f}.npz") as pm:
        assert np.array_equal(truth, unpack_test_bits(pm["truth"])), f"Test truth mismatch at f={f}"
        pm_wrong = unpack_test_bits(pm["PM-correlated"])

    total, ones = build_counts(circuit, LUT_SEED_BASE + k)
    indices = syndrome_indices(detectors)
    counts = total[indices]
    n_ones = ones[indices]
    unseen = counts == 0
    pred = np.where(unseen, truth ^ pm_wrong, n_ones > counts - n_ones)
    lut_wrong = pred != truth
    ler, pm_ler = float(lut_wrong.mean()), float(pm_wrong.mean())

    # Same discordant-pair convention as run_e1_nn_fscan.mcnemar.
    b = int((lut_wrong & ~pm_wrong).sum())
    c = int((~lut_wrong & pm_wrong).sum())
    row = {
        "f": f,
        "ler": ler,
        "ler_err": float(np.sqrt(ler * (1 - ler) / N_TEST)),
        "unseen_frac": float(unseen.mean()),
        "n_table_shots": N_BATCHES * BATCH_SHOTS,
        "ratio_vs_correlated": ler / pm_ler,
        "mcnemar_b_vs_correlated": b,
        "mcnemar_c_vs_correlated": c,
        "mcnemar_p_vs_correlated": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0,
    }
    for model in NN_MODELS:
        mean_ler = np.mean([
            unpack_test_bits(np.load(PREDICTIONS / f"{model}_f{f:.2f}_s{seed}.npy")).mean()
            for seed in NN_SEEDS
        ])
        gap = pm_ler - ler
        row[f"gap_closed_{model}"] = float((pm_ler - mean_ler) / gap) if gap else float("nan")
    return row, np.packbits(lut_wrong)


def main():
    rows, packed_errors = [], {}
    for k, f in enumerate(F_VALUES):
        row, packed_errors[f] = evaluate(f, k)
        rows.append(row)
        print(f"f={f:.2f} LER={row['ler']:.6f} unseen={row['unseen_frac']:.6f}", flush=True)

    for f, packed in packed_errors.items():
        np.save(PREDICTIONS / f"lut_f{f:.2f}.npy", packed)
    with (RESULTS / "e1_lut.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    settings = {
        "script": "src/run_e1_lut.py",
        "p": P,
        "f_values": F_VALUES,
        "lut_seed_base": LUT_SEED_BASE,
        "test_seed_base": TEST_SEED_BASE,
        "sampler_seed_rule": "seed_base + f_index",
        "nn_seeds": NN_SEEDS,
        "n_test": N_TEST,
        "n_table_shots": N_BATCHES * BATCH_SHOTS,
        "table_batches": N_BATCHES,
        "shots_per_batch": BATCH_SHOTS,
        "syndrome_index": "detector j is bit j",
        "tie": "predict 0",
        "unseen_fallback": "PM-correlated prediction (truth XOR saved PM-correlated error)",
        "mcnemar": "exact binomial, two-sided",
        "python": platform.python_version(),
        "stim": stim.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (RESULTS / "e1_lut.settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
