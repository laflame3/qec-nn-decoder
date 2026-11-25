"""E1 v2 experiment 2: train from scratch on nested f=1 prefixes."""

import sys

if __name__ == "__main__" and "--verify" in sys.argv[1:]:
    sys.dont_write_bytecode = True

from e1v2_common import (
    LEGACY_MODELS, MODELS, N_LEGACY_TRAIN, STAT_FIELDS, TORCH_SEED, ResultRow, Results, best_hparams,
    common_settings, evaluate, history_summary, legacy_model, legacy_row,
    legacy_wrong, paired_stats, parse_args, prefix_settings, sample_pool,
    target_circuit, target_data, train_fixed,
)
from model import default_device, detector_layout
from run_e2_sycamore import make_model

SIZES = (45_000, 150_000, 500_000, 1_000_000, 2_000_000, 5_000_000)
SMOKE_SIZES = (5_000, 10_000)
FIELDS = (
    "model", "n_train_unique", "max_epochs", "epochs_run", "samples_seen",
    "seed", "n_val", "n_test", *STAT_FIELDS, "best_val_pl", "best_epoch",
    "train_time_s", "source",
)


def datasize_pool(smoke):
    sizes, n_max = (SMOKE_SIZES, 20_000) if smoke else (SIZES, SIZES[-1])
    return sizes, sample_pool(n_max)


def datasize_stem(kind, n):
    return f"datasize_{kind}_n{n}_s{TORCH_SEED}"


def datasize_results(smoke):
    return Results("datasize", smoke, FIELDS, ("model", "n_train_unique", "max_epochs", "seed"))


def datasize_specs(settings):
    specs = []
    endpoint = settings["legacy_endpoint"]
    for n in (*settings["prefix_sizes"], endpoint["n_train_unique"]):
        for kind in MODELS:
            columns = {"model": kind, "n_train_unique": n, "seed": settings["torch_seed"],
                       "n_val": settings["n_val"], "n_test": settings["n_test"]}
            stem = datasize_stem(kind, n)
            if n != endpoint["n_train_unique"]:
                epochs = settings["max_epochs"][str(n)]
                columns.update(max_epochs=epochs, source="trained")
                checkpoint = {"kind": kind, "hparams": settings["hyperparameters"][kind],
                              "condition": {"n_train_unique": n, "max_epochs": epochs}}
                specs.append(ResultRow(columns, stem, checkpoint=checkpoint, max_epochs=epochs))
                continue
            columns["max_epochs"] = endpoint["max_epochs"]
            reference = None
            if settings["smoke"]:
                columns.update(epochs_run=0, best_epoch=None, train_time_s=None,
                               samples_seen=0, source="legacy_checkpoint_smoke")
            else:
                old = legacy_row(kind, 1.0)
                columns.update(epochs_run=1, best_epoch=1, samples_seen=n, source="reused_e1_fscan_f1_s0",
                               best_val_pl=float(old["best_val_ler"]), train_time_s=float(old["train_time_s"]))
                reference = legacy_wrong(kind, 1.0)
            specs.append(ResultRow(columns, stem, reference_wrong=reference,
                                   source_weights=LEGACY_MODELS / f"{kind}_f1.00_s0.pt"))
    return specs


def run_condition(spec, output, settings, pool, circuit, test, val, pm_wrong, device):
    smoke = settings["smoke"]
    row = dict(spec.columns)
    kind, n = row["model"], row["n_train_unique"]
    max_epochs = row["max_epochs"]
    if spec.checkpoint is None:
        if smoke:
            model = legacy_model(kind, 1.0, circuit, device)
            wrong = evaluate(model, test, device)
            row["best_val_pl"] = float(evaluate(model, val, device).mean())
        else:
            wrong = spec.reference_wrong
    else:
        hparams = settings["hyperparameters"]
        model = make_model(kind, TORCH_SEED, hparams, *detector_layout(circuit), device)
        history = train_fixed(
            model, (pool[0][:n], pool[1][:n]), val, lr=hparams[kind]["lr"],
            max_epochs=max_epochs, patience=settings["patience"],
            warmup_fraction=settings["warmup_fraction"], device=device,
        )
        row.update(history_summary(history))
        row["samples_seen"] = n * row["epochs_run"]
        wrong = evaluate(model, test, device)
        output.save_model(spec.stem, model, kind, hparams[kind], history, settings, spec.checkpoint["condition"])

    output.save_wrong(spec.stem, wrong)
    output.append({**row, **paired_stats(wrong, pm_wrong)})


def datasize_settings(smoke, hparams, device, sizes, pool):
    return {
        **common_settings("datasize", smoke, hparams, device),
        **prefix_settings(smoke, sizes, len(pool[1]), pool),
        "max_epochs": {str(n): min(2 if smoke else 50, N_LEGACY_TRAIN // n) for n in sizes},
        "epoch_cap": 2 if smoke else 50, "samples_budget": N_LEGACY_TRAIN,
        "patience": 10, "warmup_fraction": 0.02, "decay": "cosine",
        "fixed_train_steps": "max_epochs * ceil(n_train_unique / batch)",
        "epoch_order": "torch.randperm on CPU before transfer to training device",
        "legacy_endpoint": {"n_train_unique": N_LEGACY_TRAIN, "max_epochs": 1,
                            "sample_seeds": {"transformer": 40400, "mlp": 40410},
                            "validation": "legacy streaming chunks within the single pass",
                            "source": "legacy checkpoint" if smoke else "legacy f=1 masks and CSV"},
    }


def main():
    args = parse_args(__doc__)
    smoke = args.smoke
    device, hparams = default_device(), best_hparams()
    sizes, pool = datasize_pool(smoke)
    output = datasize_results(smoke)
    settings = datasize_settings(smoke, hparams, output.recorded_device() if args.verify else device, sizes, pool)
    specs = datasize_specs(settings)
    with output.open(settings, specs, read_only=args.verify, require_complete=args.verify):
        if args.verify:
            print(f"PASS datasize: {len(sizes)} training-pool prefix SHA-256 values match")
            return
        pending = [spec for spec in specs if output.pending(**spec.columns)]
        if pending:
            circuit = target_circuit()
            test, val, pm_wrong = target_data(circuit, smoke)
            for spec in pending:
                print(f"{spec.columns['model']} n_train_unique={spec.columns['n_train_unique']:,}", flush=True)
                run_condition(spec, output, settings, pool, circuit, test, val, pm_wrong, device)


if __name__ == "__main__":
    main()
