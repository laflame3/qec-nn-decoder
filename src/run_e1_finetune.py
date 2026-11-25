"""E1 v2 experiment 3: transfer from f=0 to f=1."""

import sys

if __name__ == "__main__" and "--verify" in sys.argv[1:]:
    sys.dont_write_bytecode = True

import numpy as np

from e1v2_common import (
    LEGACY_MODELS, MODELS, N_LEGACY_TRAIN, P_BG, RESULTS, STAT_FIELDS, TORCH_SEED,
    ResultRow, Results, best_hparams, common_settings, evaluate, history_summary,
    legacy_model, legacy_row, legacy_wrong, paired_stats, parse_args,
    prefix_settings, prefix_sha256, sample_pool, target_circuit, target_data, train_fixed,
)
from model import default_device, detector_layout
from noise import correlated_memory_circuit
from run_e1_datasize import (
    SIZES, SMOKE_SIZES, datasize_pool, datasize_results, datasize_settings, datasize_specs, datasize_stem,
)

METHODS = ("zero_shot", "scratch_45k", "finetune_45k", "full_1e7")
FIELDS = (
    "model", "method", "source_model", "n_target_shots", "epochs_run",
    "best_epoch", "seed", "n_val", "n_test", *STAT_FIELDS, "best_val_pl",
    "train_time_s", "source",
)
FT_LR = 1e-4


def finetune_pool(smoke):
    n, n_max = (SMOKE_SIZES[0], 20_000) if smoke else (SIZES[0], SIZES[-1])
    return n, sample_pool(n_max)


def scratch_results(settings, pool, verify):
    smoke, n = settings["smoke"], settings["target_shots"]
    datasize = datasize_results(smoke)
    if not datasize.settings_path.exists() and not datasize.csv_path.exists() and not verify:
        _, reference_pool = datasize_pool(smoke)
        if settings["prefix_sha256"][str(n)] != prefix_sha256(reference_pool, n):
            raise ValueError(f"experiment 2 vs 3 prefix SHA-256 mismatch for {n:,} shots")
        return {}

    # Use the scratch schedule from experiment 2 and the pool and architecture
    # expected by experiment 3. The fine-tuning schedule itself is different.
    sizes = SMOKE_SIZES if smoke else SIZES
    expected = datasize_settings(smoke, settings["hyperparameters"], datasize.recorded_device(), sizes, pool)
    if expected["prefix_sha256"][str(n)] != settings["prefix_sha256"][str(n)]:
        raise ValueError(f"experiment 2 vs 3 prefix SHA-256 mismatch for {n:,} shots")
    specs = datasize_specs(expected)
    cached = {}
    with datasize.open(expected, specs, read_only=True):
        for kind in MODELS:
            stem = datasize_stem(kind, n)
            key = datasize.key({"model": kind, "n_train_unique": n, "seed": settings["torch_seed"],
                                "max_epochs": expected["max_epochs"][str(n)]})
            if key in datasize.rows:
                cached[kind] = (datasize.rows[key], datasize.read_wrong(stem, settings["n_test"]),
                                datasize.model_path(stem))
            elif verify:
                raise ValueError(f"missing scratch result for {kind} at {n:,} shots: {datasize.csv_path}")
    return cached


def finetune_specs(settings, scratch):
    n, smoke = settings["target_shots"], settings["smoke"]
    specs = []
    for method in METHODS:
        for kind in MODELS:
            columns = {"model": kind, "method": method, "seed": settings["torch_seed"],
                       "n_val": settings["n_val"], "n_test": settings["n_test"]}
            source_model = LEGACY_MODELS / f"{kind}_f0.00_s0.pt"
            checkpoint = reference = max_epochs = None
            if method == "scratch_45k":
                if kind not in scratch:
                    print(f"{kind} scratch_45k 等待实验 2 ({n:,} shots)", flush=True)
                    continue
                old, reference, source_model = scratch[kind]
                columns.update(n_target_shots=n, source="reused_e1_datasize")
                for name in (*STAT_FIELDS, "epochs_run", "best_epoch", "best_val_pl", "train_time_s"):
                    columns[name] = old[name]
            elif method == "finetune_45k":
                columns.update(n_target_shots=n, source="finetuned_e1_fscan_f0_s0")
                hparams = {**settings["hyperparameters"][kind], "lr": settings["finetune_lr"]}
                checkpoint = {"kind": kind, "hparams": hparams, "condition": {
                    "method": method, "n_target_shots": n, "source_model": source_model.name,
                }}
                max_epochs = settings["max_epochs"]
            elif method == "zero_shot":
                columns.update(n_target_shots=0, epochs_run=0, best_epoch=0, train_time_s=None,
                               source="legacy_checkpoint_f0")
            else:
                source_model = LEGACY_MODELS / f"{kind}_f1.00_s0.pt"
                columns["n_target_shots"] = settings["legacy_train_shots"]
                if smoke:
                    columns.update(epochs_run=0, best_epoch=0, train_time_s=None, source="legacy_checkpoint_smoke")
                else:
                    old = legacy_row(kind, 1.0)
                    columns.update(epochs_run=1, best_epoch=1, source="reused_e1_fscan_f1_s0",
                                   best_val_pl=float(old["best_val_ler"]), train_time_s=float(old["train_time_s"]))
                    reference = legacy_wrong(kind, 1.0)
            columns["source_model"] = source_model.relative_to(RESULTS.parent.parent).as_posix()
            specs.append(ResultRow(columns, f"finetune_{kind}_{method}_s{TORCH_SEED}", checkpoint=checkpoint,
                                   source_weights=source_model, reference_wrong=reference, max_epochs=max_epochs))
    return specs


def run_method(spec, output, settings, pool, pretrain, circuit, test, val, pm_wrong, device):
    row = dict(spec.columns)
    kind, method = row["model"], row["method"]
    if spec.reference_wrong is not None:
        wrong = spec.reference_wrong
    else:
        f = 1.0 if method == "full_1e7" else 0.0
        model = legacy_model(kind, f, circuit if f else pretrain, device)
        if spec.checkpoint is not None:
            n = row["n_target_shots"]
            history = train_fixed(
                model, (pool[0][:n], pool[1][:n]), val,
                lr=settings["finetune_lr"], max_epochs=settings["max_epochs"],
                patience=settings["patience"], warmup_fraction=settings["warmup_fraction"],
                evaluate_initial=True, device=device,
            )
            row.update(history_summary(history))
            output.save_model(spec.stem, model, kind, spec.checkpoint["hparams"], history, settings,
                              spec.checkpoint["condition"])
        else:
            row["best_val_pl"] = float(evaluate(model, val, device).mean())
        wrong = evaluate(model, test, device)

    output.save_wrong(spec.stem, wrong)
    output.append({**row, **paired_stats(wrong, pm_wrong)})


def main():
    args = parse_args(__doc__)
    smoke = args.smoke
    device, hparams = default_device(), best_hparams()
    pretrain = correlated_memory_circuit(P_BG, 0.0)[0]
    circuit = target_circuit()
    for before, after in zip(detector_layout(pretrain), detector_layout(circuit), strict=True):
        np.testing.assert_array_equal(before, after, err_msg="f=0 and f=1 detector layout differs")
    n, pool = finetune_pool(smoke)
    output = Results("finetune", smoke, FIELDS, ("model", "method", "n_target_shots", "seed"))
    settings = {
        **common_settings("finetune", smoke, hparams, output.recorded_device() if args.verify else device),
        **prefix_settings(smoke, (n,), len(pool[1]), pool),
        "methods": list(METHODS), "pretrain_f": 0.0, "target_shots": n,
        "finetune_lr": FT_LR, "max_epochs": 2 if smoke else 30,
        "patience": 8, "warmup_fraction": 0.05, "decay": "cosine",
        "fixed_train_steps": "max_epochs * ceil(n_target_shots / batch)",
        "epoch_order": "torch.randperm on CPU before transfer to training device",
        "evaluate_epoch_zero": True,
        "legacy_f0_train_seeds": {"transformer": 40000, "mlp": 40010},
        "legacy_f1_train_seeds": {"transformer": 40400, "mlp": 40410},
        "legacy_epochs": "single streaming pass with validation by chunk",
        "legacy_train_shots": N_LEGACY_TRAIN,
        "full_source": "legacy checkpoint" if smoke else "legacy f=1 masks and CSV",
    }
    scratch = scratch_results(settings, pool, args.verify)
    specs = finetune_specs(settings, scratch)
    with output.open(settings, specs, read_only=args.verify, require_complete=args.verify):
        if args.verify:
            print("PASS finetune: 2 scratch rows match experiment 2 schedule, prefix SHA-256 and masks")
            return
        pending = [spec for spec in specs if output.pending(**spec.columns)]
        if pending:
            test, val, pm_wrong = target_data(circuit, smoke)
            for spec in pending:
                run_method(spec, output, settings, pool, pretrain, circuit, test, val, pm_wrong, device)


if __name__ == "__main__":
    main()
