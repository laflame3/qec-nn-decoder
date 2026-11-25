"""Check E1 v2 circuits, cached outcomes, metrics and training-pool alignment."""

import numpy as np

from decoders import basis_decomposed_dem, build_pm_correlated, decode_pm_correlated, edge_conflicts
from e1v2_common import (
    MODELS, N_TEST, P_BG, PM, RESULTS, TARGET_TEST_SEED, legacy_model,
    legacy_pm, legacy_row, legacy_wrong, paired_stats, prefix_sha256, read_rows, sample, target_circuit,
)
from model import default_device, predict
from noise import calibrate_idle_prob, correlated_memory_circuit, mean_detection_density, structure_memory_circuit
from run_e1_datasize import datasize_pool
from run_e1_finetune import finetune_pool
from run_e1_structure import KINDS

CAL_SHOTS = 200_000
REFERENCES = {
    "X": (0.0055, 0.021), "Y": (0.0034, 0.015),
    "XX": (0.0048, 0.082), "YY": (0.0028, 0.056),
}


def check_dep():
    dep, info = structure_memory_circuit("DEP", P_BG)
    original, _ = correlated_memory_circuit(P_BG, 0.0)
    assert str(dep) == str(original)
    assert (dep.num_detectors, info["n_data"], info["n_pairs"]) == (24, 9, 12)
    _, truth = sample(dep, N_TEST, 20261121)
    truth_saved, _ = legacy_pm(0.0)
    np.testing.assert_array_equal(truth, truth_saved)
    print("1 PASS  DEP circuit text identical; 24 detectors, 9 data qubits, 12 pairs; legacy DEP truth identical")


def check_calibration():
    dep, _ = correlated_memory_circuit(P_BG, 0.0)
    target = mean_detection_density(dep, CAL_SHOTS, seed=70000)
    calibrated = {}
    for ki, kind in enumerate(KINDS):
        if kind == "DEP":
            prob, density = P_BG, target
        else:
            prob, record = calibrate_idle_prob(kind, target, CAL_SHOTS, seed=70000 + ki)
            density = record["density"]
        circuit, _ = structure_memory_circuit(kind, prob)
        basis_decomposed_dem(circuit)
        calibrated[kind] = (circuit, prob, density)
        print(f"2 PASS  {kind:>3} relabelled_mass={edge_conflicts(circuit):.8f}")
    print(f"3       target density={target:.6f} detectors/shot ({CAL_SHOTS:,} shots)")
    for ki, kind in enumerate(KINDS[1:], start=1):
        circuit, prob, density = calibrated[kind]
        det, truth = sample(circuit, CAL_SHOTS, seed=71000 + ki)
        pm_pl = float((decode_pm_correlated(build_pm_correlated(circuit), det) != truth).mean())
        prob_ref, pl_ref = REFERENCES[kind]
        prob_shift, pl_shift = prob / prob_ref - 1, pm_pl / pl_ref - 1
        assert abs(density / target - 1) < 0.01
        print(f"3 PASS  {kind:>3} idle_prob={prob:.8f} density={density:.6f} PM P_L={pm_pl:.6f} "
              f"(vs reference: idle {prob_shift:+.1%}, PM {pl_shift:+.1%})")
    return calibrated


def check_even_parity(calibrated):
    for ki, kind in enumerate(("XX", "YY"), start=3):
        circuit, _ = structure_memory_circuit(kind, calibrated[kind][1], p_bg=0)
        det, truth = sample(circuit, 100_000, seed=75000 + ki)
        wrong = int(np.count_nonzero(decode_pm_correlated(build_pm_correlated(circuit), det) != truth))
        print(f"4 {'PASS' if wrong == 0 else 'FAIL'}  {kind} only, p_bg=0: {wrong:,}/100,000 wrong "
              f"(d=3; pair faults preserve X parity)")
        assert wrong == 0


def check_target(circuit):
    det, truth = sample(circuit, N_TEST, seed=TARGET_TEST_SEED)
    truth_saved, pm_saved = legacy_pm(1.0)
    np.testing.assert_array_equal(truth, truth_saved)
    pm_wrong = decode_pm_correlated(build_pm_correlated(circuit), det) != truth
    np.testing.assert_array_equal(pm_wrong, pm_saved)
    print(f"5 PASS  f=1 truth and PM masks bit-identical on {N_TEST:,} shots (PM P_L={pm_wrong.mean():.6f})")
    return det, truth, pm_saved


def check_checkpoints(circuit, det, truth, device):
    for kind in MODELS:
        model = legacy_model(kind, 1.0, circuit, device)
        wrong = predict(model, det, device) != truth
        differences = int(np.count_nonzero(wrong != legacy_wrong(kind, 1.0)))
        pl = float(wrong.mean())
        pl_difference = abs(pl - float(legacy_row(kind, 1.0)["ler"]))
        print(f"6 {'PASS' if differences <= 50 and pl_difference <= 5e-5 else 'FAIL'}  {kind}: "
              f"{differences} different bits, P_L={pl:.6f}, |P_L - legacy|={pl_difference:.8f}")
        assert differences <= 50
        assert pl_difference <= 5e-5


def check_stats(pm_wrong):
    stats = paired_stats(legacy_wrong("transformer", 1.0), pm_wrong)
    row = next(row for row in read_rows(RESULTS / "e1_ratio_ci.csv")
               if float(row["f"]) == 1.0 and row["model"] == "transformer"
               and row["seed"] == "0" and row["baseline"] == PM)
    for key, old in (("ratio", "ratio"), ("ratio_ci_low", "ci_low"), ("ratio_ci_high", "ci_high")):
        assert abs(stats[key] - float(row[old])) <= 1e-6
    assert abs(stats["delta"] - (-0.0055)) <= 5e-5
    print(f"7 PASS  ratio={stats['ratio']:.9f}, CI=[{stats['ratio_ci_low']:.9f}, "
          f"{stats['ratio_ci_high']:.9f}], delta={stats['delta']:.6f}")


def check_pool():
    sizes, pool = datasize_pool(smoke=False)
    for small, large in zip(sizes, sizes[1:]):
        larger_prefix = tuple(bits[:large] for bits in pool)
        assert prefix_sha256(larger_prefix, small) == prefix_sha256(pool, small)
    n, ft_pool = finetune_pool(smoke=False)
    assert n == sizes[0] == 45_000
    digest = prefix_sha256(pool, n)
    assert digest == prefix_sha256(ft_pool, n)
    print(f"8 PASS  all {len(sizes)} prefixes nested; experiment 2/3 45k SHA-256={digest}")


def main():
    device = default_device()
    print(f"E1 v2 checks (device={device})", flush=True)
    check_dep()
    calibrated = check_calibration()
    check_even_parity(calibrated)
    circuit = target_circuit()
    det, truth, pm_wrong = check_target(circuit)
    check_checkpoints(circuit, det, truth, device)
    check_stats(pm_wrong)
    check_pool()
    print("All 8 checks passed.")


if __name__ == "__main__":
    main()
