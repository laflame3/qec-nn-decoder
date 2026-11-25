"""Offline E3 v2 analysis of the retrieved IBM repetition-code experiment."""

import argparse
import csv
import hashlib
import io
import json
import platform
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import pymatching
import scipy
import stim
import torch

from analyze_e3 import P_UNIFORM, detectors, fit_edge_probs, graph_edges, matching
from e1v2_common import (
    BATCH, EVAL_BATCH, MODELS, RESULTS, STAT_FIELDS, TORCH_SEED, WEIGHT_DECAY,
    ResultRow, Results, atomic_write, best_hparams, evaluate, history_summary,
    paired_stats, train_fixed,
)
from model import default_device
from run_e2_sycamore import make_model

RAW = RESULTS / "e3_ibm_v2_raw.npz"
JOB = RESULTS / "e3_ibm_v2_job.json"
ROUNDS, N_DET, PUB_SHOTS = 9, 20, 10_000
TAUS = (0, 5, 15)
BLOCKS = {"train": (0, 1, 2), "val": (3,), "test": (4,)}
STAB = np.tile([0, 1], ROUNDS + 1)
RND = np.repeat(np.arange(ROUNDS + 1), 2)
EDGES = graph_edges(ROUNDS)
SUBSETS = ("all", "L0", "L1")
REAL_REFS = ("PM-fitted", "PM-uniform", "majority")
SIM_REFS = ("PM-fitted",)
EXP = ("main", "graph", "transfer", "datasize", "diag")
TRANSFER = ("zero_shot", "scratch_60k", "finetune_60k", "sim_only_test")
SIM_SEEDS = {"train": 81000, "val": 82000, "test": 83000, "pretrain": 84000}
SUMMARY = ("epochs_run", "best_epoch", "best_val_pl", "train_time_s")
EXTRA_STATS = ("pm_pl", "ratio", "b", "c", "mcnemar_p")
EXTRA_FIELDS = tuple(
    f"{ref}_{field}" for ref in ("pm_uniform", "majority") for field in EXTRA_STATS
)
DIAG_FIELDS = (
    "n_test", "detector_rate",
    *(f"round_{r}_rate" for r in range(ROUNDS + 1)),
    *(f"d{d}_equals_L" for d in range(3)),
    "n_unseen", "unseen_fraction",
    "pm_fitted_unseen_pl", *(f"{kind}_unseen_pl" for kind in MODELS),
)


@dataclass(frozen=True)
class Split:
    pub_ids: np.ndarray
    det: np.ndarray
    data: np.ndarray
    logical: np.ndarray

    @property
    def xy(self):
        return self.det, self.data[:, 0] ^ self.logical


def load_raw():
    """Read PUBs in retrieval order and verify their metadata before splitting."""
    job = json.loads(JOB.read_text())
    order = job["pub_order"]
    expected_grid = {(block, tau, logical) for block in range(5)
                     for tau in TAUS for logical in (0, 1)}
    if (len(order) != 30 or job["rounds"] != ROUNDS or job["block_shots"] != PUB_SHOTS
            or job["status"] != "DONE"):
        raise ValueError("unexpected job metadata")
    with np.load(RAW, allow_pickle=False) as raw:
        metadata = np.column_stack([raw[f"pub_{name}"] for name in ("block", "tau_us", "logical")])
        expected_order = [(o["block"], o["tau_us"], o["logical"]) for o in order]
        if ([o["pub"] for o in order] != list(range(30))
                or set(expected_order) != expected_grid
                or not np.array_equal(metadata, expected_order)):
            raise ValueError("raw PUB metadata differs from the recorded job order")
        pubs = []
        for i in range(len(order)):
            syn, data = raw[f"syn_p{i}"], raw[f"data_p{i}"]
            if (syn.shape != (PUB_SHOTS, ROUNDS, 2) or data.shape != (PUB_SHOTS, 3)
                    or syn.dtype != np.uint8 or data.dtype != np.uint8
                    or syn.max() > 1 or data.max() > 1):
                raise ValueError(f"unexpected data in PUB {i}")
            pubs.append((detectors(syn, data).reshape(-1, N_DET), data))
    return job, metadata, pubs


def real_splits(metadata, pubs, tau):
    splits = {}
    for name, blocks in BLOCKS.items():
        ids = np.flatnonzero(np.isin(metadata[:, 0], blocks) & (metadata[:, 1] == tau))
        splits[name] = Split(
            ids,
            np.concatenate([pubs[i][0] for i in ids]),
            np.concatenate([pubs[i][1] for i in ids]),
            np.repeat(metadata[ids, 2].astype(np.uint8), PUB_SHOTS),
        )
    return splits


def train_pool(split, tau, smoke):
    ids = np.random.default_rng(85000 + tau).permutation(len(split.logical))
    if smoke:
        # Keep the shuffle order and take 2500 shots of each initial state.
        logical = split.logical[ids]
        keep = np.sort(np.concatenate([np.flatnonzero(logical == L)[:2500] for L in (0, 1)]))
        ids = ids[keep]
    det, y = split.xy
    return det[ids], y[ids]


def majority_wrong(split):
    return (split.data.sum(axis=1) >= 2) != split.logical


def make_dem(probs):
    dem = stim.DetectorErrorModel()
    for i, j in EDGES:
        targets = [stim.target_relative_detector_id(i)]
        if j is not None:
            targets.append(stim.target_relative_detector_id(j))
        elif i % 2 == 0:
            targets.append(stim.target_logical_observable_id(0))
        dem.append("error", probs[i, j], targets)
    return dem


def sim_size(split, smoke):
    return 10_000 if smoke else {
        "train": 60_000, "val": 20_000, "test": 200_000, "pretrain": 1_000_000,
    }[split]


def sample_dem(dem, tau, split, smoke):
    det, obs, _ = dem.compile_sampler(seed=SIM_SEEDS[split] + tau).sample(sim_size(split, smoke))
    return det, obs[:, 0]


def graph_pm_wrong(dem, probs, test_xy):
    det, obs = test_xy
    original = matching(EDGES, probs, N_DET).decode_batch(det)[:, 0]
    from_dem = pymatching.Matching.from_detector_error_model(dem).decode_batch(det)[:, 0]
    np.testing.assert_array_equal(from_dem, original, err_msg="DEM and fitted graph PM disagree")
    return from_dem != obs


def subset_mask(test, subset):
    return slice(None) if subset == "all" else test.logical == int(subset[1])


def real_baselines(probs, test):
    det, y = test.xy
    return {
        "PM-fitted": matching(EDGES, probs, N_DET).decode_batch(det)[:, 0] != y,
        "PM-uniform": matching(EDGES, dict.fromkeys(EDGES, P_UNIFORM), N_DET).decode_batch(det)[:, 0] != y,
        "majority": majority_wrong(test),
    }


def write_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(path, lambda file: file.write(payload))


def record_wrong(output, stem, wrong):
    if output.pred_path(stem).exists():
        np.testing.assert_array_equal(output.read_wrong(stem, len(wrong)), wrong, err_msg=stem)
    else:
        output.save_wrong(stem, wrong)


class E3Results(Results):
    """Use the shared settings lock and artifact helpers with E3 output paths."""

    def __init__(self, exp, smoke, columns, keys):
        super().__init__(exp, smoke, columns, keys)
        stem = f"e3v2_{exp}{self.suffix}"
        self.csv_path = RESULTS / f"{stem}.csv"
        self.settings_path = RESULTS / f"{stem}.settings.json"
        self.pred_dir = RESULTS / f"e3v2_preds{self.suffix}"
        self.model_dir = RESULTS / f"e3v2_models{self.suffix}"

    def validate_row(self, row, spec, settings):
        self.check_columns(row, spec.columns)
        if "weights" in row:
            n_test = int(row["n_test"])
            wrong = self.read_wrong(row["prediction"], n_test)
            path = self.model_path(row["weights"])
            if not path.is_file():
                raise ValueError(f"missing weights: {path}")
            history = json.loads(path.with_suffix(".history.json").read_text())
            self.check_columns(row, history_summary(history))
            refs = SIM_REFS if row["test_domain"] == "graph" else REAL_REFS
            masks = {ref: self.read_wrong(baseline_stem(row, ref), n_test) for ref in refs}
            self.check_columns(row, metrics(wrong, masks, "subset" in row))
        else:
            n_test, n_unseen = int(row["n_test"]), int(row["n_unseen"])
            if not 0 <= n_unseen <= n_test:
                raise ValueError("invalid unseen syndrome count")
            self.check_columns(row, {"unseen_fraction": n_unseen / n_test})
            for name in DIAG_FIELDS:
                if name in ("n_test", "n_unseen") or (not n_unseen and name.endswith("_unseen_pl")):
                    continue
                if not 0 <= float(row[name]) <= 1:
                    raise ValueError(f"invalid diagnostic rate: {name}")

    def append(self, row):
        # Publish complete CSV rows atomically after their masks and weights exist.
        key = self.key(row)
        if key in self.rows:
            raise ValueError(f"duplicate result: {key}")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=self.fields)
        writer.writeheader()
        writer.writerows(self.rows.values())
        writer.writerow(row)
        atomic_write(self.csv_path, lambda file: file.write(buffer.getvalue().encode()))
        self.rows[key] = row
        rate = f" P_L={row['pl']:.5f}" if "pl" in row else ""
        print(f"saved {self.csv_path.stem}: {' / '.join(key)}{rate}", flush=True)


def baseline_stem(row, ref):
    return f"{row['test_domain']}_tau{row['tau']}_{ref}_{row.get('subset', 'all')}"


def metrics(wrong, refs, extra=False):
    stats = paired_stats(wrong, refs["PM-fitted"])
    if extra:
        for ref, prefix in (("PM-uniform", "pm_uniform"), ("majority", "majority")):
            values = paired_stats(wrong, refs[ref])
            stats.update({f"{prefix}_{key}": values[key] for key in EXTRA_STATS})
    return stats


def save_result(output, row, wrong, refs, history):
    record_wrong(output, row["prediction"], wrong)
    output.append({**row, **metrics(wrong, refs, "subset" in row), **history_summary(history)})


def fit_model(output, stem, kind, hparams, settings, train, val, device, recipe, initial=None):
    """Resume a complete fit, or run it and publish its checkpoint and history."""
    path = output.model_path(stem)
    history_path = path.with_suffix(".history.json")
    torch.manual_seed(TORCH_SEED)
    model = make_model(kind, TORCH_SEED, hparams, STAB, RND, device)
    if path.exists() and history_path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key, value in {"kind": kind, "hparams": hparams[kind], "settings": settings,
                           "condition": stem}.items():
            if checkpoint[key] != value:
                raise ValueError(f"checkpoint {key} differs from settings: {path}")
        model.load_state_dict(checkpoint["state_dict"])
        return model, json.loads(history_path.read_text())
    if initial is not None:
        model.load_state_dict(initial.state_dict())
    print(f"fit {stem}: {len(train[1]):,} training shots, {device}", flush=True)
    history = train_fixed(model, train, val, device=device, **recipe)
    output.save_model(stem, model, kind, hparams[kind], history, settings, stem)
    return model, history


def curve_sizes(smoke):
    return (500, 1667, 5000) if smoke else (6000, 20_000, 60_000)


def training_recipe(stage, smoke):
    if stage == "pretrain":
        return dict(max_epochs=1, patience=10, warmup_fraction=0.02)
    if stage == "finetune":
        return dict(max_epochs=2 if smoke else 30, patience=8, warmup_fraction=0.05,
                    lr=1e-4, evaluate_initial=True)
    return dict(max_epochs=2 if smoke else 50, patience=10, warmup_fraction=0.02)


def make_settings(job, smoke, taus, hparams, device):
    hashes = {}
    for path in (RAW, JOB):
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "script": "src/run_e3_v2.py", "job_id": job["job_id"], "input_sha256": hashes,
        "smoke": smoke, "taus_us": list(taus), "rounds": ROUNDS, "n_detectors": N_DET,
        "blocks": {name: list(values) for name, values in BLOCKS.items()},
        "observable": "final d0 XOR initial logical state", "nn_inputs": "20 detectors",
        "layout": {"stab": STAB.tolist(), "rnd": RND.tolist()},
        "torch_seed": TORCH_SEED, "hparams": hparams, "device": str(device),
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "batch": BATCH, "eval_batch": EVAL_BATCH, "weight_decay": WEIGHT_DECAY,
        "optimizer": "AdamW, linear warmup and cosine decay",
        "validation_selection": "earliest minimum validation P_L",
        "prediction_rule": "logit > 0", "mask_format": "NumPy packbits, big bit order",
        "real_mask_order": "original PUB order then original shot order within each PUB",
        "pm": {"p_uniform": P_UNIFORM, "fit": "training shots only, both initial states pooled",
               "correlations": "graphlike repetition code without hyperedges, PM-correlated equals ordinary PM"},
        "sim_seed_bases": SIM_SEEDS, "sim_seed_rule": "seed base + tau in microseconds",
        "sim_draw_sizes": {split: sim_size(split, smoke) for split in SIM_SEEDS},
        "sim_train_used": curve_sizes(smoke)[-1],
        "sim_labels": "Stim DEM observable L0; even detector boundary edges carry L0",
        "pretrain_validation": "real validation block",
        "curve_sizes": list(curve_sizes(smoke)), "curve_seed_base": 85000,
        "curve_order": "concatenate in PUB order, then one shuffled nested prefix per tau"
                       + (" with a balanced 2500/2500 cap for smoke" if smoke else ""),
        "training_recipes": {stage: training_recipe(stage, smoke)
                             for stage in ("scratch", "pretrain", "finetune")},
        "statistics": {"delta": "paired NN minus PM, 95% Wald CI",
                       "ratio": "paired delta method, 95% CI", "mcnemar": "two-sided exact binomial"},
        "versions": {"python": platform.python_version(), "stim": stim.__version__,
                     "torch": str(torch.__version__), "numpy": np.__version__,
                     "pymatching": pymatching.__version__, "scipy": scipy.__version__},
    }


def experiment_rows(exp, tau, smoke):
    n_real = curve_sizes(smoke)[-1]
    n_sim_test, n_pre = (sim_size(split, smoke) for split in ("test", "pretrain"))
    if exp == "diag":
        return [{"tau": tau, "subset": s, "n_test": 2 * PUB_SHOTS if s == "all" else PUB_SHOTS}
                for s in SUBSETS]
    rows = []
    for kind in MODELS:
        base = f"tau{tau}_{kind}"
        main = f"main_{base}"
        pre = f"pretrain_{base}"
        if exp == "main":
            for s in SUBSETS:
                rows.append(dict(tau=tau, model=kind, subset=s, test_domain="real", n_train=n_real,
                                 n_test=2 * PUB_SHOTS if s == "all" else PUB_SHOTS,
                                 weights=main, prediction=f"{main}_{s}"))
        elif exp == "graph":
            stem = f"graph_{base}"
            rows.append(dict(tau=tau, model=kind, test_domain="graph", n_train=n_real, n_test=n_sim_test,
                             weights=stem, prediction=stem))
        elif exp == "transfer":
            for mode in TRANSFER:
                sim = mode == "sim_only_test"
                weights = {"scratch_60k": main, "finetune_60k": f"finetune_{base}"}.get(mode, pre)
                prediction = f"{main}_all" if mode == "scratch_60k" else f"{mode}_{base}"
                rows.append(dict(tau=tau, model=kind, mode=mode, test_domain="graph" if sim else "real",
                                 n_train=n_pre if weights == pre else n_real,
                                 n_sim_pretrain=0 if weights == main else n_pre,
                                 n_test=n_sim_test if sim else 2 * PUB_SHOTS,
                                 weights=weights, prediction=prediction))
        elif exp == "datasize":
            for n in curve_sizes(smoke):
                weights = main if n == n_real else f"datasize_{base}_n{n}"
                rows.append(dict(tau=tau, model=kind, n_train=n, test_domain="real", n_test=2 * PUB_SHOTS,
                                 weights=weights, prediction=f"{main}_all" if n == n_real else weights))
    return rows


def save_baselines(output, tau, domain, refs, test=None):
    for subset in SUBSETS if test is not None else ("all",):
        mask = slice(None) if test is None else subset_mask(test, subset)
        key = {"tau": tau, "test_domain": domain, "subset": subset}
        for name, wrong in refs.items():
            record_wrong(output, baseline_stem(key, name), wrong[mask])


def diagnostics(output, rows, test, train_det, refs):
    # Pack all 20 detector bits so membership compares whole shot patterns.
    seen = np.unique(np.packbits(train_det, axis=1), axis=0)
    patterns = np.packbits(test.det, axis=1)
    # View each row as one hashable key, including its deterministic padding bits.
    key_type = np.dtype((np.void, patterns.shape[1]))
    unseen = ~np.isin(patterns.view(key_type).ravel(), seen.view(key_type).ravel())
    for row in rows:
        mask = subset_mask(test, row["subset"])
        det = test.det[mask].reshape(-1, ROUNDS + 1, 2)
        final = test.data[mask] == test.logical[mask, None]
        new = unseen[mask]
        pred = {"pm_fitted": refs["PM-fitted"][mask]}
        for kind in MODELS:
            stem = f"main_tau{row['tau']}_{kind}_{row['subset']}"
            pred[kind] = output.read_wrong(stem, len(det))
        output.append({
            **row, "detector_rate": float(det.mean()),
            **{f"round_{r}_rate": float(value) for r, value in enumerate(det.mean(axis=(0, 2)))},
            **{f"d{d}_equals_L": float(value) for d, value in enumerate(final.mean(axis=0))},
            "n_unseen": int(new.sum()), "unseen_fraction": float(new.mean()),
            **{f"{name}_unseen_pl": float(wrong[new].mean()) if new.any() else None
               for name, wrong in pred.items()},
        })


def run_tau(tau, splits, probs, outputs, settings, smoke, hparams, device):
    pending = {}
    for exp, output in outputs.items():
        rows = experiment_rows(exp, tau, smoke)
        pending[exp] = [row for row in rows if output.key(row) not in output.rows]
        skipped = len(rows) - len(pending[exp])
        if skipped:
            print(f"skip completed {exp} tau={tau}: {skipped}/{len(rows)} rows", flush=True)
    if not any(pending.values()):
        return

    test, val = splits["test"], splits["val"].xy
    pool = train_pool(splits["train"], tau, smoke)
    needs_real = any(pending.get(exp) for exp in ("main", "transfer", "datasize", "diag"))
    if needs_real:
        real_refs = real_baselines(probs, test)
        save_baselines(outputs["main"], tau, "real", real_refs, test)

    if pending.get("graph") or pending.get("transfer"):
        dem = make_dem(probs)
        sim_test = sample_dem(dem, tau, "test", smoke)
        sim_refs = {"PM-fitted": graph_pm_wrong(dem, probs, sim_test)}
        save_baselines(outputs["graph"] if "graph" in outputs else outputs["transfer"],
                       tau, "graph", sim_refs)
        if pending.get("graph"):
            det, y = sample_dem(dem, tau, "train", smoke)
            sim_train = det[:len(pool[1])], y[:len(pool[1])]
            sim_val = sample_dem(dem, tau, "val", smoke)
        if any(row["mode"] != "scratch_60k" for row in pending.get("transfer", [])):
            sim_pre = sample_dem(dem, tau, "pretrain", smoke)

    for exp in ("main", "graph", "transfer", "datasize"):
        rows = pending.get(exp)
        if not rows:
            continue
        output = outputs[exp]
        for kind in MODELS:
            def fit(stem, train, validation, stage="scratch", initial=None):
                recipe = {"lr": hparams[kind]["lr"], **training_recipe(stage, smoke)}
                return fit_model(output, stem, kind, hparams, settings[exp], train, validation,
                                 device, recipe, initial)

            if exp == "main":
                selected = [row for row in rows if row["model"] == kind]
                if not selected:
                    continue
                model, history = fit(selected[0]["weights"], pool, val)
                wrong = evaluate(model, test.xy, device)
                for row in selected:
                    mask = subset_mask(test, row["subset"])
                    save_result(output, row, wrong[mask],
                                {name: ref[mask] for name, ref in real_refs.items()}, history)
                continue

            pre = None
            for row in (r for r in rows if r["model"] == kind):
                mode = row.get("mode")
                sim = row["test_domain"] == "graph"
                refs = sim_refs if sim else real_refs
                if mode == "scratch_60k" or (exp == "datasize" and row["n_train"] == len(pool[1])):
                    wrong = output.read_wrong(row["prediction"], row["n_test"])
                    history = json.loads(output.model_path(row["weights"]).with_suffix(".history.json").read_text())
                else:
                    if exp == "transfer" and pre is None:
                        pre = fit(f"pretrain_tau{tau}_{kind}", sim_pre, val, "pretrain")
                    if mode in ("zero_shot", "sim_only_test"):
                        model, history = pre
                    elif mode == "finetune_60k":
                        model, history = fit(row["weights"], pool, val, "finetune", pre[0])
                    elif exp == "graph":
                        model, history = fit(row["weights"], sim_train, sim_val)
                    else:
                        prefix = tuple(array[:row["n_train"]] for array in pool)
                        model, history = fit(row["weights"], prefix, val)
                    wrong = evaluate(model, sim_test if sim else test.xy, device)
                save_result(output, row, wrong, refs, history)
    if pending.get("diag"):
        diagnostics(outputs["diag"], pending["diag"], test, pool[0], real_refs)


def run(exp, smoke):
    requested = set(EXP if exp == "all" else (exp,))
    if requested & {"transfer", "datasize", "diag"}:
        requested.add("main")
    device = torch.device("cpu") if smoke else default_device()
    taus = (0,) if smoke else TAUS
    job, metadata, pubs = load_raw()
    hparams = best_hparams()
    common = make_settings(job, smoke, taus, hparams, device)
    splits = {tau: real_splits(metadata, pubs, tau) for tau in taus}
    settings, outputs = {}, {}
    with ExitStack() as stack:
        # Enter every selected settings lock in the same order before training.
        edges_output = E3Results("edges", smoke, (), ())
        stack.enter_context(edges_output.open({**common, "experiment": "edges"}, []))
        for name in EXP:
            if name not in requested:
                continue
            rows = [row for tau in taus for row in experiment_rows(name, tau, smoke)]
            keys = {"main": ("tau", "model", "subset"), "graph": ("tau", "model"),
                    "transfer": ("tau", "model", "mode"), "datasize": ("tau", "model", "n_train"),
                    "diag": ("tau", "subset")}[name]
            if name == "diag":
                fields = ("tau", "subset", *DIAG_FIELDS)
            else:
                fields = (*rows[0], *STAT_FIELDS, *SUMMARY, *(EXTRA_FIELDS if name == "main" else ()))
            outputs[name] = E3Results(name, smoke, fields, keys)
            settings[name] = {**common, "experiment": name, "row_keys": list(keys)}
            stack.enter_context(outputs[name].open(settings[name], [ResultRow(row, "") for row in rows]))

        probs = {tau: fit_edge_probs(train_pool(splits[tau]["train"], tau, smoke)[0], EDGES)
                 for tau in taus}
        edges = {str(tau): {f"{i}-{'B' if j is None else j}": p[i, j] for i, j in EDGES}
                 for tau, p in probs.items()}
        path = edges_output.csv_path.with_suffix(".json")
        if path.exists():
            if json.loads(path.read_text()) != edges:
                raise ValueError(f"fitted edges differ from the training blocks: {path}")
            print(f"skip completed {path.name}", flush=True)
        else:
            write_json(path, edges)
        for tau in taus:
            run_tau(tau, splits[tau], probs[tau], outputs, settings, smoke, hparams, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run tau=0 on CPU with reduced training.")
    parser.add_argument("--exp", choices=(*EXP, "all"), default="all")
    args = parser.parse_args()
    run(args.exp, args.smoke)


if __name__ == "__main__":
    main()
