"""Rescore the saved E2 v1 test masks against correlated PM with simple.dem."""

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from e1v2_common import STAT_FIELDS, atomic_write, paired_stats, unpack_bits
from run_e2_sycamore import (
    CALIBRATED_DEM, MODELS, N_TEST, PACKET, REGIMES, RESULTS, SEEDS, SPLIT_SEED,
    load_packet, split_indices,
)

PREDS = RESULTS / "e2_preds"
BASELINES = PREDS / "baselines.npz"
SPLIT = RESULTS / "e2_split.npz"
PM = "PM-corr+simple"
PM_SIMPLE = "PM+simple"
FITTED_PRIOR = "per_batch_fitted_likely_saw_test"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_references(obs, google):
    """Align the saved masks and select the Google reference on validation only."""
    train, val, test = split_indices()
    with np.load(SPLIT, allow_pickle=False) as saved:
        for name, indices in (("train_idx", train), ("val_idx", val), ("test_idx", test)):
            np.testing.assert_array_equal(indices, saved[name], err_msg=name)
    with np.load(BASELINES, allow_pickle=False) as saved:
        np.testing.assert_array_equal(obs[test], unpack_bits(saved["truth"], N_TEST),
                                      err_msg="legacy test truth differs")
        masks = {name: unpack_bits(saved[name], N_TEST) for name in saved.files if name != "truth"}

    val_pl = {}
    for (dem, decoder), prediction in google.items():
        name = f"Google {decoder}+{dem}"
        val_pl[name] = float(np.mean(prediction[val] != obs[val]))
        np.testing.assert_array_equal(prediction[test] != obs[test], masks[name], err_msg=name)
    selected = min(val_pl, key=lambda name: (val_pl[name], name))
    reference = {"name": selected, "validation_pl": val_pl[selected],
                 "test_pl": float(masks[selected].mean())}
    return (train, val, test), masks, val_pl, reference


def legacy_mask_path(model, regime, seed):
    return PREDS / f"{model}_{regime}_s{seed}.npy"


def legacy_wrong(model, regime, seed):
    return unpack_bits(np.load(legacy_mask_path(model, regime, seed), allow_pickle=False), N_TEST)


def csv_bytes(fields, rows):
    file = io.StringIO(newline="")
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return file.getvalue().encode()


def main():
    _, _, _, obs, google = load_packet()
    (_, val, test), masks, val_pl, reference = load_references(obs, google)
    paths = {legacy_mask_path(model, regime, seed)
             for model in MODELS for regime in REGIMES for seed in SEEDS}
    if paths != set(PREDS.glob("*.npy")):
        raise ValueError("Expected exactly the 30 E2 v1 model/condition/seed masks")

    rows = []
    for model in MODELS:
        for regime in REGIMES:
            stage = "preregistered" if regime in ("real", "pre0", "ft") else "post_hoc"
            stats = [paired_stats(legacy_wrong(model, regime, seed), masks[PM]) for seed in SEEDS]
            values = [row["pl"] for row in stats]
            rows.extend(
                {"model": model, "regime": regime, "seed": seed, "protocol_stage": stage,
                 **scores, "pl_seed_min": min(values), "pl_seed_max": max(values)}
                for seed, scores in zip(SEEDS, stats, strict=True)
            )

    references = []
    for name, wrong in masks.items():
        if name == PM or not name.startswith(("Google ", "PM+", "PM-corr+")):
            continue
        dem = name.split("+", 1)[1]
        # simple_cal was fit locally to the training split (E2 v1 calibrate_dem).
        prior = {"simple": False, CALIBRATED_DEM: "train_only"}.get(dem, FITTED_PRIOR)
        references.append({"name": name, "prior_fitted": prior, "val_pl": val_pl.get(name),
                           **paired_stats(wrong, masks[PM])})

    settings = {
        "script": "src/rescore_e2_v2.py", "main_baseline": PM,
        "split_seed": SPLIT_SEED, "n_val": len(val), "n_test": len(test),
        "google_selection": "minimum validation P_L, tie broken by name",
        "google_reference": reference, "seeds": list(SEEDS),
        "input_sha256": {str(path.relative_to(RESULTS.parents[1])): sha256(path)
                         for path in sorted(paths | {BASELINES, SPLIT, PACKET / "r05.npz"})},
    }
    outputs = {
        RESULTS / "e2v2_rescore.csv": csv_bytes(
            ("model", "regime", "seed", "protocol_stage", *STAT_FIELDS, "pl_seed_min", "pl_seed_max"), rows),
        RESULTS / "e2v2_reference.csv": csv_bytes(("name", "prior_fitted", "val_pl", *STAT_FIELDS), references),
        RESULTS / "e2v2_rescore.settings.json": (json.dumps(settings, indent=2, sort_keys=True) + "\n").encode(),
    }
    for path, payload in outputs.items():
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Existing rescore output differs: {path}")
    for path, payload in outputs.items():
        if not path.exists():
            atomic_write(path, lambda file: file.write(payload), exclusive=True)
    print(f"Rescored {len(rows)} seed rows and {len(references)} references against {PM}.")
    print(f"{reference['name']}: validation P_L={reference['validation_pl']:.6f}, "
          f"test P_L={reference['test_pl']:.6f}")


if __name__ == "__main__":
    main()
