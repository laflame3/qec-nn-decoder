"""E1 v2 experiment 1: match detection density across idle noise structures."""

import sys

if __name__ == "__main__" and "--verify" in sys.argv[1:]:
    sys.dont_write_bytecode = True

import numpy as np

from decoders import build_pm_correlated, decode_pm_correlated, edge_conflicts
from e1v2_common import (
    BATCH, LEGACY_MODELS, MODELS, N_LEGACY_TRAIN, N_VAL, P_BG, PM, STAT_FIELDS, TORCH_SEED,
    WEIGHT_DECAY, ResultRow, Results, best_hparams, common_settings, error_rate, evaluate,
    legacy_model, legacy_pm, legacy_row, legacy_wrong, paired_stats, parse_args,
    sample, test_sizes,
)
from model import default_device, detector_layout, train
from noise import calibrate_idle_prob, mean_detection_density, structure_memory_circuit
from run_e2_sycamore import make_model

KINDS = ("DEP", "X", "Y", "XX", "YY")
FIELDS = (
    "kind", "decoder", "seed", "idle_prob", "density_target", "density_cal",
    "density_test", "flip_rate_test", "relabelled_mass", "n_train", "n_val",
    "n_test", *STAT_FIELDS, "best_val_pl", "train_time_s", "source",
)


def prepare_kind(kind, settings, target):
    seeds = settings["seeds"][kind]
    n_test, p_bg = settings["n_test"], settings["p_bg"]
    if kind == "DEP":
        idle_prob, density_cal = p_bg, target
    else:
        idle_prob, record = calibrate_idle_prob(kind, target, settings["calibration_shots"],
                                               seeds["calibration"], p_bg)
        density_cal = record["density"]
    circuit, _ = structure_memory_circuit(kind, idle_prob, p_bg)
    test_data = sample(circuit, n_test, seeds["test"])
    test_det, truth = test_data
    reused = kind == "DEP" and not settings["smoke"]
    pm_wrong = decode_pm_correlated(build_pm_correlated(circuit), test_det) != truth
    if reused:
        saved_truth, saved_pm = legacy_pm(0.0)
        np.testing.assert_array_equal(truth, saved_truth, err_msg="DEP legacy test truth differs")
        if not np.array_equal(pm_wrong, saved_pm):
            raise ValueError("regenerated DEP PM mask differs from the legacy mask")
    row_base = {
        "kind": kind, "idle_prob": idle_prob, "density_target": target,
        "density_cal": density_cal, "density_test": float(test_det.sum() / n_test),
        "flip_rate_test": float(truth.mean()), "relabelled_mass": edge_conflicts(circuit),
        "n_test": n_test,
    }
    source = "reused_e1_fscan_f0_s0" if reused else "pm_correlated"
    pm_spec = ResultRow(
        {**row_base, "decoder": PM, "seed": None, "n_train": 0, "n_val": 0, "source": source,
         **dict.fromkeys(STAT_FIELDS[2:]), "best_val_pl": None, "train_time_s": None},
        f"structure_{kind}_{PM}", reference_wrong=pm_wrong,
    )
    specs = {PM: pm_spec}
    for model_kind in MODELS:
        columns = {**row_base, "decoder": model_kind, "seed": settings["torch_seed"],
                   "n_train": settings["n_train"], "n_val": settings["n_val"], "source": "trained"}
        checkpoint = source_weights = reference = None
        if reused:
            old = legacy_row(model_kind, 0.0)
            columns.update(n_train=int(old["n_train"]), n_val=N_VAL, source=source,
                           best_val_pl=float(old["best_val_ler"]), train_time_s=float(old["train_time_s"]))
            reference = legacy_wrong(model_kind, 0.0)
        elif kind == "DEP":
            columns.update(n_train=0, source="legacy_checkpoint_smoke", train_time_s=None)
        else:
            checkpoint = {
                "kind": model_kind, "hparams": settings["hyperparameters"][model_kind],
                "condition": {"kind": kind, "idle_prob": idle_prob, "sample_seed": seeds["train"][model_kind]},
            }
        if kind == "DEP":
            source_weights = LEGACY_MODELS / f"{model_kind}_f0.00_s0.pt"
        specs[model_kind] = ResultRow(columns, f"structure_{kind}_{model_kind}_s{TORCH_SEED}",
                                      checkpoint=checkpoint, source_weights=source_weights,
                                      reference_wrong=reference)
    return specs, circuit, test_data, pm_wrong


def run_kind(kind, prepared, pending, output, settings, device):
    specs, circuit, test_data, pm_wrong = prepared
    row_base = specs[PM].columns
    n_val = settings["n_val"]
    reused = kind == "DEP" and not settings["smoke"]
    print(f"{kind} idle_prob={row_base['idle_prob']:.7f} density_cal={row_base['density_cal']:.5f} "
          f"density_test={row_base['density_test']:.5f}", flush=True)

    if PM in pending:
        pl, pl_se = error_rate(pm_wrong)
        output.save_wrong(specs[PM].stem, pm_wrong)
        output.append({**row_base, "pl": pl, "pl_se": pl_se})

    nn_pending = [decoder for decoder in pending if decoder in MODELS]
    if not nn_pending:
        return
    val_data = None if reused else sample(circuit, n_val, settings["seeds"][kind]["validation"])
    for model_kind in nn_pending:
        spec = specs[model_kind]
        row = dict(spec.columns)
        if reused:
            wrong = spec.reference_wrong
        elif kind == "DEP":
            model = legacy_model(model_kind, 0.0, circuit, device)
            wrong = evaluate(model, test_data, device)
            row["best_val_pl"] = float(evaluate(model, val_data, device).mean())
        else:
            hparams = settings["hyperparameters"]
            condition = spec.checkpoint["condition"]
            model = make_model(model_kind, TORCH_SEED, hparams, *detector_layout(circuit), device)
            history = train(
                model, circuit, *val_data, n_train=row["n_train"], lr=hparams[model_kind]["lr"],
                batch=BATCH, weight_decay=WEIGHT_DECAY, chunk=settings["train_chunk"],
                sample_seed=condition["sample_seed"], torch_seed=TORCH_SEED, device=device,
                log=lambda message: print(message, flush=True),
            )
            row["best_val_pl"] = min(entry["val_ler"] for entry in history)
            row["train_time_s"] = history[-1]["time_s"]
            wrong = evaluate(model, test_data, device)
            output.save_model(spec.stem, model, model_kind, hparams[model_kind], history, settings, condition)
        output.save_wrong(spec.stem, wrong)
        output.append({**row, **paired_stats(wrong, pm_wrong)})


def main():
    args = parse_args(__doc__)
    smoke = args.smoke
    device, hparams = default_device(), best_hparams()
    output = Results("structure", smoke, FIELDS, ("kind", "decoder", "seed"))
    n_test, n_val = test_sizes(smoke)
    seeds = {
        kind: {"calibration": 70_000 + ki, "test": 71_000 + ki, "validation": 72_000 + ki,
               "train": {model: 73_000 + 10 * ki + m for m, model in enumerate(MODELS)}}
        for ki, kind in enumerate(KINDS)
    }
    if not smoke:
        seeds["DEP"].update({
            "test": 20261121, "validation": 30_000, "train": {"transformer": 40_000, "mlp": 40_010},
        })
    settings = {
        **common_settings("structure", smoke, hparams, output.recorded_device() if args.verify else device),
        "kinds": list(KINDS), "seeds": seeds, "calibration_target": "DEP at idle_prob=0.005",
        "density_unit": "mean lit detectors per shot", "calibration_shots": 50_000 if smoke else 1_000_000,
        "calibration_bounds": [1e-4, 5e-2], "calibration_max_iterations": 40,
        "calibration_stop_relative_error": 0.002, "calibration_failure_relative_error": 0.01,
        "n_test": n_test, "n_val": n_val, "n_train": 20_000 if smoke else N_LEGACY_TRAIN,
        "train_chunk": 500_000, "warmup_fraction": 0.02, "decay": "cosine",
        "streaming_validation_batch": 65536,
        "dep_source": "legacy checkpoint evaluated on reduced sample" if smoke
        else "legacy f=0 masks and CSV, with same test set regenerated for density",
    }
    dep, _ = structure_memory_circuit("DEP", P_BG)
    target = mean_detection_density(dep, settings["calibration_shots"], seeds["DEP"]["calibration"])
    prepared = {kind: prepare_kind(kind, settings, target) for kind in KINDS}
    specs = [spec for cases, *_ in prepared.values() for spec in cases.values()]
    with output.open(settings, specs, read_only=args.verify, require_complete=args.verify):
        if args.verify:
            print("PASS structure: 5 regenerated PM masks match bit-for-bit")
            return
        for kind, data in prepared.items():
            pending = [decoder for decoder, spec in data[0].items() if output.pending(**spec.columns)]
            if pending:
                run_kind(kind, data, pending, output, settings, device)


if __name__ == "__main__":
    main()
