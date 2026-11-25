"""Run the E2 real-data learning curve or pretraining and transfer experiment."""

import argparse
import json
import sys

sys.dont_write_bytecode = True

import numpy as np
import torch

from e1v2_common import (
    BATCH, EVAL_BATCH, STAT_FIELDS, TORCH_SEED, WEIGHT_DECAY, Results, best_hparams,
    evaluate, history_summary, mcnemar, paired_stats, prefix_sha256, read_rows, train_fixed,
)
from model import default_device, detector_layout
from rescore_e2_v2 import (
    BASELINES, PM, PM_SIMPLE, SPLIT, legacy_mask_path, legacy_wrong, load_references, sha256,
)
from run_e2_sycamore import (
    N_DETECTORS, N_SHOTS, N_TRAIN, PACKET, RESULTS, SPLIT_SEED, load_packet, make_model,
)

MODELS = ("mlp", "transformer")
POOL = PACKET.parent / "e2_pool"
POOL_SIZE = 3_900_000
SAMPLE14 = 14
SMOKE_POOL_SIZE = 20_000
SIZES = {
    "mlp": (45_000, 150_000, 500_000, 1_000_000, 2_000_000, POOL_SIZE + N_TRAIN),
    "transformer": (45_000, 500_000, POOL_SIZE + N_TRAIN),
}
SMOKE_SIZES = (5_000, 10_000)
A_SEED, B_SEED = 20261126, 20261127
TRAIN_BUDGET = 10_000_000
TRAIN_WARMUP, FT_WARMUP = 0.02, 0.05
FT_LR = 1e-4
TRAIN_PATIENCE, FT_PATIENCE = 10, 8
BASE_FIELDS = ("seed", "n_val", "n_test", *STAT_FIELDS, "ratio_pm_simple", "p_pm_simple",
               "ratio_google_ref", "p_google_ref", "best_val_pl", "train_time_s")
A_FIELDS = ("model", "n_train_unique", "n_from_sample14", "max_epochs",
            "epochs_run", "best_epoch", *BASE_FIELDS)
B_FIELDS = ("model", "method", "pretrain_source", "n_pretrain", "n_target_shots",
            "epochs_run", "best_epoch", *BASE_FIELDS, "source")
PRETRAIN, FINETUNE, SCRATCH = "pretrain_zero_shot", "finetune_45k", "scratch_45k_sample14"
LEGACY_METHODS = {f"legacy_{regime}_s0": regime for regime in ("pre0", "ft", "real")}
B_METHODS = (PRETRAIN, FINETUNE, SCRATCH, *LEGACY_METHODS)
LEGACY_CSV = RESULTS / "e2_nn.csv"


class E2Results(Results):
    def __init__(self, experiment, smoke, fields, key_fields):
        super().__init__(experiment, smoke, fields, key_fields)
        stem = f"e2v2_{experiment}{self.suffix}"
        self.csv_path = RESULTS / f"{stem}.csv"
        self.settings_path = RESULTS / f"{stem}.settings.json"
        self.pred_dir = RESULTS / f"e2v2_preds{self.suffix}"
        self.model_dir = RESULTS / f"e2v2_models{self.suffix}"


def load_pool(n):
    manifest = json.loads((POOL / "manifest.json").read_text(encoding="utf-8"))
    digest = sha256(POOL / "pool.npz")
    if manifest["output_files"]["pool.npz"] != digest:
        raise ValueError("pool.npz does not match the export manifest")
    with np.load(POOL / "pool.npz", allow_pickle=False) as file:
        det, obs = file["detectors"], file["obs"]
    if (det.shape != (POOL_SIZE, N_DETECTORS // 8) or det.dtype != np.uint8
            or obs.shape != (POOL_SIZE,) or obs.dtype != bool):
        raise ValueError("Unexpected pool detector/observation shape or dtype")
    return (det[:n], obs[:n]), digest


def shuffled_data(packed, obs, order):
    det = np.unpackbits(packed[order], axis=1, count=N_DETECTORS, bitorder="big")
    return det.view(bool), obs[order]


def experiment_a_pool(external, target, sizes):
    n_target = len(target[1])
    packed = np.concatenate((np.packbits(target[0], axis=1, bitorder="big"), external[0]))
    obs = np.concatenate((target[1], external[1]))
    order = np.random.default_rng(A_SEED).permutation(len(obs))
    data = shuffled_data(packed, obs, order[:max(sizes)])
    prefixes = {str(n): prefix_sha256(data, n) for n in sizes}
    counts = {str(n): int(np.count_nonzero(order[:n] < n_target)) for n in sizes}
    return data, prefixes, counts


def experiment_b_pool(external):
    order = np.random.default_rng(B_SEED).permutation(len(external[1]))
    return shuffled_data(*external, order)


def legacy_history_path(kind, regime):
    return RESULTS / "e2_models" / f"{kind}_{regime}_s0.history.json"


def test_scores(wrong, masks, google_name):
    row = paired_stats(wrong, masks[PM])
    for tag, name in (("pm_simple", PM_SIMPLE), ("google_ref", google_name)):
        comparator = masks[name]
        row[f"ratio_{tag}"] = float(wrong.mean() / comparator.mean())
        row[f"p_{tag}"] = mcnemar(wrong, comparator)[2]
    return row


def publish(output, settings, stem, condition, summary, wrong, masks):
    output.save_wrong(stem, wrong)
    output.append({**condition, **summary, "n_val": settings["n_val"], "n_test": settings["n_test"],
                   **test_scores(wrong, masks, settings["google_reference"]["name"])})


def train(model, data, val, plan, device):
    return train_fixed(model, data, val, device=device, **plan)


def run_datasize(output, settings, selected_models, data, val, test, masks, layout, device):
    for kind in selected_models:
        for n in settings["sizes_by_model"][kind]:
            max_epochs = settings["max_epochs"][str(n)]
            condition = {"model": kind, "n_train_unique": n, "max_epochs": max_epochs, "seed": TORCH_SEED}
            if not output.pending(**condition):
                continue
            condition["n_from_sample14"] = settings["n_from_sample14"][str(n)]
            stem = f"datasize_{kind}_n{n}_s{TORCH_SEED}"
            print(f"{kind} n_train_unique={n:,}", flush=True)
            model = make_model(kind, TORCH_SEED, settings["hyperparameters"], *layout, device)
            plan = {**settings["train"], "lr": settings["hyperparameters"][kind]["lr"],
                    "max_epochs": max_epochs}
            history = train(model, (data[0][:n], data[1][:n]), val, plan, device)
            output.save_model(stem, model, kind, settings["hyperparameters"][kind], history, settings, condition)
            publish(output, settings, stem, condition, history_summary(history), evaluate(model, test, device), masks)


def transfer_condition(kind, method, n_external, legacy):
    if legacy is None:
        n_pretrain = n_external if method in (PRETRAIN, FINETUNE) else 0
        pretrain_source = "external_sycamore_samples" if n_pretrain else "none"
    else:
        n_pretrain = int(legacy["n_pretrain"])
        pretrain_source = "simple.dem" if n_pretrain else "none"
    return {
        "model": kind, "method": method, "pretrain_source": pretrain_source,
        "n_pretrain": n_pretrain, "n_target_shots": 0 if method in (PRETRAIN, "legacy_pre0_s0") else N_TRAIN,
        "seed": TORCH_SEED, "source": "trained" if legacy is None else "reused_e2_v1",
    }


def legacy_summary(kind, regime, row):
    history = json.loads(legacy_history_path(kind, regime).read_text(encoding="utf-8"))
    best = min(history, key=lambda entry: entry["val_ler"])
    return {"epochs_run": int(row["epochs_run"]), "best_epoch": best.get("epoch"),
            "best_val_pl": float(row["best_val_ler"]), "train_time_s": float(row["train_time_s"])}


def restore_pretrain(output, stem, model, condition, settings):
    path = output.model_dir / f"{stem}{output.suffix}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["condition"] != condition or checkpoint["settings"] != settings:
        raise ValueError(f"Pretraining checkpoint configuration differs: {path}")
    model.load_state_dict(checkpoint["state_dict"])


def run_transfer(output, settings, selected_models, data, target, val, test, masks, layout, device):
    legacy_rows = {(row["model"], row["regime"]): row for row in read_rows(LEGACY_CSV)
                   if int(row["seed"]) == TORCH_SEED}
    for kind in selected_models:
        hparams = settings["hyperparameters"]
        pretrain_stem = f"transfer_{kind}_{PRETRAIN}_s{TORCH_SEED}"
        pretrain_condition = transfer_condition(kind, PRETRAIN, len(data[1]), None)
        for method in B_METHODS:
            regime = LEGACY_METHODS.get(method)
            legacy = legacy_rows[(kind, regime)] if regime else None
            condition = transfer_condition(kind, method, len(data[1]), legacy)
            if not output.pending(**condition):
                continue
            stem = f"transfer_{kind}_{method}_s{TORCH_SEED}"
            print(f"{kind} {method}", flush=True)
            if legacy is not None:
                wrong = legacy_wrong(kind, regime, TORCH_SEED)
                summary = legacy_summary(kind, regime, legacy)
            else:
                model = make_model(kind, TORCH_SEED, hparams, *layout, device)
                train_data = data if method == PRETRAIN else target
                plan = dict(settings["plans"][method])
                if method == FINETUNE:
                    restore_pretrain(output, pretrain_stem, model, pretrain_condition, settings)
                else:
                    plan["lr"] = hparams[kind]["lr"]
                history = train(model, train_data, val, plan, device)
                summary, wrong = history_summary(history), evaluate(model, test, device)
                output.save_model(stem, model, kind, hparams[kind], history, settings, condition)
            publish(output, settings, stem, condition, summary, wrong, masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=("datasize", "transfer"), required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--smoke", action="store_true", help="Use reduced budgets on CPU and separate outputs.")
    args = parser.parse_args()
    models = tuple(kind for kind in MODELS if kind in args.models)
    if args.smoke:
        torch.set_num_threads(4)
    device = torch.device("cpu") if args.smoke else default_device()
    hparams = best_hparams()
    if any(hparams[kind] != {"n_layers": 3, "lr": 1e-3} for kind in MODELS):
        raise ValueError(f"Tuning table does not yield the preregistered hyperparameters: {hparams}")

    circuit, _, det, obs, google = load_packet()
    (train_idx, val_idx, test_idx), masks, _, reference = load_references(obs, google)
    if (len(obs), len(train_idx)) != (N_SHOTS, N_TRAIN):
        raise ValueError("Unexpected sample_14 split sizes")
    target, val, test = ((det[idx], obs[idx]) for idx in (train_idx, val_idx, test_idx))
    external, pool_hash = load_pool(SMOKE_POOL_SIZE if args.smoke else POOL_SIZE)
    layout = detector_layout(circuit)
    epoch_cap = 2 if args.smoke else 50
    base_plan = {"patience": TRAIN_PATIENCE, "warmup_fraction": TRAIN_WARMUP}
    inputs = [BASELINES, SPLIT, PACKET / "r05.npz", PACKET / "circuit_ideal.stim", RESULTS / "e1_tune.csv"]
    if args.exp == "transfer":
        inputs += [LEGACY_CSV]
        inputs += [path for kind in MODELS for regime in LEGACY_METHODS.values()
                   for path in (legacy_mask_path(kind, regime, TORCH_SEED), legacy_history_path(kind, regime))]
    settings = {
        "script": "src/run_e2_v2.py", "experiment": args.exp, "smoke": args.smoke, "device": str(device),
        "torch_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "torch_seed": TORCH_SEED, "hyperparameters": hparams,
        "n_val": len(val[1]), "n_test": len(test[1]), "target_sample": SAMPLE14, "split_seed": SPLIT_SEED,
        "pm_baseline": PM, "pm_reference": PM_SIMPLE, "google_reference": reference,
        "google_selection": "minimum validation P_L, tie broken by name",
        "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY, "batch": BATCH, "eval_batch": EVAL_BATCH,
        "loss": "BCEWithLogitsLoss", "decay": "cosine", "train_budget": TRAIN_BUDGET,
        "validation_selection": "earliest minimum validation P_L", "prediction_bits": "packbits, big",
        "external_pool_size": len(external[1]), "external_pool_sha256": pool_hash,
        "input_sha256": {path.relative_to(RESULTS.parents[1]).as_posix(): sha256(path) for path in inputs},
    }
    if args.exp == "datasize":
        sizes_by_model = {kind: list(SMOKE_SIZES if args.smoke else SIZES[kind]) for kind in MODELS}
        sizes = sorted({n for values in sizes_by_model.values() for n in values})
        pool, hashes, counts = experiment_a_pool(external, target, sizes)
        settings.update({
            "pool_seed": A_SEED, "n_pool": len(target[1]) + len(external[1]), "train": base_plan,
            "sizes_by_model": sizes_by_model, "prefix_sha256": hashes, "n_from_sample14": counts,
            "prefix_hash_rule": "SHA-256 of unpacked bool detector bytes followed by bool obs bytes in C order",
            "max_epochs": {str(n): min(epoch_cap, TRAIN_BUDGET // n) for n in sizes},
        })
        output = E2Results("datasize", args.smoke, A_FIELDS, ("model", "n_train_unique", "max_epochs", "seed"))
        with output.open(settings):
            run_datasize(output, settings, models, pool, val, test, masks, layout, device)
    else:
        pool = experiment_b_pool(external)
        settings.update({
            "pool_seed": B_SEED, "pool_sha256": prefix_sha256(pool, len(pool[1])),
            "methods": list(B_METHODS), "n_target": len(target[1]),
            "plans": {
                PRETRAIN: {**base_plan, "max_epochs": min(epoch_cap, TRAIN_BUDGET // len(pool[1]))},
                FINETUNE: {"lr": FT_LR, "warmup_fraction": FT_WARMUP, "patience": FT_PATIENCE,
                           "max_epochs": min(epoch_cap, 30), "evaluate_initial": True},
                SCRATCH: {**base_plan, "max_epochs": epoch_cap},
            },
        })
        output = E2Results("transfer", args.smoke, B_FIELDS,
                           ("model", "method", "pretrain_source", "n_pretrain", "n_target_shots", "seed", "source"))
        with output.open(settings):
            run_transfer(output, settings, models, pool, target, val, test, masks, layout, device)


if __name__ == "__main__":
    main()
