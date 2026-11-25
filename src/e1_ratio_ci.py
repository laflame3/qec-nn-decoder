"""Paired delta-method intervals for E1 logical error rate ratios.

Run src/run_e1_lut.py first, then ../../.venv/bin/python src/e1_ratio_ci.py.
"""

import csv
from statistics import NormalDist

import numpy as np

from run_e1_lut import F_VALUES, NN_MODELS, NN_SEEDS, PREDICTIONS, RESULTS, unpack_test_bits

BASELINES = ("PM-correlated", "PM-informed")
Z_95 = NormalDist().inv_cdf(0.975)


def paired_ratio_ci(errors, baseline_errors):
    """Apply the ratio gradient to each pair before estimating its variance."""
    baseline_ler = baseline_errors.mean()
    ratio = float(errors.mean() / baseline_ler)
    # Var(X - ratio * Y) keeps the covariance between the two errors on each shot.
    residual = errors - ratio * baseline_errors
    se = residual.std(ddof=1) / (np.sqrt(errors.size) * baseline_ler)
    return ratio, float(ratio - Z_95 * se), float(ratio + Z_95 * se)


def ratio_row(f, model, seed, baseline, errors, baseline_errors):
    ratio, ci_low, ci_high = paired_ratio_ci(errors, baseline_errors)
    return {"f": f, "model": model, "seed": seed, "baseline": baseline,
            "ratio": ratio, "ci_low": ci_low, "ci_high": ci_high}


def main():
    rows = []
    for f in F_VALUES:
        with np.load(PREDICTIONS / f"pm_f{f:.2f}.npz") as pm:
            baseline_errors = {baseline: unpack_test_bits(pm[baseline]) for baseline in BASELINES}

        for model in NN_MODELS:
            errors_by_seed = [
                unpack_test_bits(np.load(PREDICTIONS / f"{model}_f{f:.2f}_s{seed}.npy"))
                for seed in NN_SEEDS
            ]
            pooled_errors = np.mean(errors_by_seed, axis=0)
            for baseline, pm_wrong in baseline_errors.items():
                for seed, errors in zip(NN_SEEDS, errors_by_seed, strict=True):
                    rows.append(ratio_row(f, model, seed, baseline, errors, pm_wrong))
                rows.append(ratio_row(f, model, "pooled", baseline, pooled_errors, pm_wrong))

        lut_wrong = unpack_test_bits(np.load(PREDICTIONS / f"lut_f{f:.2f}.npy"))
        rows.append(ratio_row(f, "lut", "", "PM-correlated", lut_wrong, baseline_errors["PM-correlated"]))

    with (RESULTS / "e1_ratio_ci.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired ratio intervals", flush=True)


if __name__ == "__main__":
    main()
