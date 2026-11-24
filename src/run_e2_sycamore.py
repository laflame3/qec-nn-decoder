"""E2 decoder baselines and neural training on the exported Sycamore r05 packet."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pymatching
import scipy
import stim
import torch
from scipy.stats import binomtest
from torch import nn

from model import MLPDecoder, TransformerDecoder, default_device, detector_layout, n_params, predict


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data" / "e2_r05_packet"
RESULTS = ROOT / "data" / "results"
TUNE_CSV = RESULTS / "e1_tune.csv"
SPLIT_SEED = 20261123
N_TRAIN, N_VAL, N_TEST = 45_000, 7_500, 22_500
N_SHOTS, N_DETECTORS = N_TRAIN + N_VAL + N_TEST, 56
DRIFT_BLOCK = 5_000
N_PRETRAIN = 2_000_000
PRETRAIN_CHECK_EVERY = 200_000
N_PRETRAIN_CAL = 10_000_000
PRETRAIN_CAL_CHECK_EVERY = 500_000
CALIBRATION_SHOTS, SIM_TEST_SHOTS = 200_000, 200_000
CALIBRATION_SEED, SIM_TEST_SEED = 20261124, 20261125
CALIBRATION_RTOL, MAX_ERROR_PROBABILITY = 0.01, 0.5
CALIBRATED_DEM = "simple_cal"
CALIBRATED_COMPARATOR = "pm_corr_simple_cal"
PRETRAIN_BATCH, REAL_BATCH, EVAL_BATCH = 1024, 256, 1024
TRAIN_WARMUP, FT_WARMUP = 0.02, 0.05
FT_LR, WEIGHT_DECAY = 1e-4, 1e-4
MAX_EPOCHS_REAL, MAX_EPOCHS_FT = 50, 30
PATIENCE_REAL, PATIENCE_FT = 10, 8
MODELS, REGIMES, SEEDS = ("transformer", "mlp"), ("real", "pre0", "ft", "pcal", "ftcal"), (0, 1, 2)
FINETUNE_FROM = {"ft": "pre0", "ftcal": "pcal"}
PM_DEMS = ("simple", "correlations", "rl_isolated_correlated_matching", "rl_shared_correlated_matching")
GOOGLE_DECODERS = ("correlated_matching", "belief_matching", "harmony")
NN_FIELDS = (
    "model", "regime", "seed", "n_params", "n_pretrain", "epochs_run",
    "best_val_ler", "ler_test", "se_test", "train_time_s",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pretrain", type=int, default=N_PRETRAIN)
    parser.add_argument("--n-pretrain-cal", type=int, default=N_PRETRAIN_CAL)
    parser.add_argument("--max-epochs-real", type=int, default=MAX_EPOCHS_REAL)
    parser.add_argument("--max-epochs-ft", type=int, default=MAX_EPOCHS_FT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", choices=MODELS, nargs="+", default=MODELS)
    parser.add_argument("--regimes", choices=REGIMES, nargs="+", default=REGIMES)
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()
    if min(args.n_pretrain, args.n_pretrain_cal, args.max_epochs_real, args.max_epochs_ft) < 1:
        parser.error("training budgets must be positive")
    if any(not 0 <= seed < 2**64 for seed in args.seeds):
        parser.error("seeds must be unsigned 64-bit integers")
    if set(args.seeds).intersection((CALIBRATION_SEED, SIM_TEST_SEED)):
        parser.error("training seeds must differ from the calibration and simulated test seeds")
    if not re.fullmatch(r"(?:_[A-Za-z0-9_-]+)?", args.out_suffix):
        parser.error("out-suffix must be empty or start with an underscore")
    return args


def load_packet():
    circuit = stim.Circuit.from_file(PACKET / "circuit_ideal.stim")
    metadata = json.loads((PACKET / "metadata.json").read_text())
    assert (metadata["shots"], circuit.num_detectors, circuit.num_observables) == (N_SHOTS, N_DETECTORS, 1)
    dem_paths = sorted((PACKET / "dems").glob("*.dem"))
    dem_names = [path.stem for path in dem_paths]
    assert len(dem_names) == 8 and set(PM_DEMS).issubset(dem_names)
    dems = {path.stem: stim.DetectorErrorModel.from_file(path) for path in dem_paths}
    with np.load(PACKET / "r05.npz", allow_pickle=False) as packet:
        det = np.unpackbits(packet["detectors"], axis=1, count=N_DETECTORS).astype(bool)
        obs = packet["obs"]
        google = {
            (dem, decoder): np.unpackbits(packet[f"google_pred_{dem}_{decoder}"], count=N_SHOTS).astype(bool)
            for dem in dem_names for decoder in GOOGLE_DECODERS
        }
    assert det.shape == (N_SHOTS, N_DETECTORS) and obs.shape == (N_SHOTS,) and obs.dtype == bool
    return circuit, dems, det, obs, google


def split_indices():
    perm = np.random.default_rng(SPLIT_SEED).permutation(N_SHOTS)
    train_idx, val_idx, test_idx = np.split(perm, (N_TRAIN, N_TRAIN + N_VAL))
    assert not np.intersect1d(train_idx, val_idx).size
    assert not np.intersect1d(train_idx, test_idx).size
    assert not np.intersect1d(val_idx, test_idx).size
    return train_idx, val_idx, test_idx


def write_csv(path, fields, rows):
    pending = path.with_name(f".pending_{path.name}")
    with pending.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def drift_rows(det, obs):
    for start in range(0, len(obs), DRIFT_BLOCK):
        stop = min(start + DRIFT_BLOCK, len(obs))
        yield {
            "shot_start": start, "shot_stop": stop,
            "mean_lit_detectors": float(det[start:stop].sum(axis=1).mean()),
            "undecoded_flip_rate": float(obs[start:stop].mean()),
        }


def ler_and_se(wrong):
    ler = float(wrong.mean())
    return ler, math.sqrt(ler * (1 - ler) / len(wrong))


def scale_dem(instructions, alpha):
    scaled = stim.DetectorErrorModel()
    for inst in instructions:
        if inst.type == "error":
            probability = min(alpha * inst.args_copy()[0], MAX_ERROR_PROBABILITY)
            scaled.append("error", probability, inst.targets_copy())
        else:
            scaled.append(inst)
    return scaled


def calibrate_dem(simple_dem, train_det):
    """Fit one scale using only the mean detector count of real training shots."""
    target = float(train_det.sum(axis=1).mean())
    instructions = list(simple_dem.flattened())
    probabilities = [inst.args_copy()[0] for inst in instructions if inst.type == "error"]
    positive = [p for p in probabilities if p > 0]
    if not positive or target <= 0:
        raise ValueError("DEM calibration requires positive error probabilities and a positive training mean")

    def sample_mean(dem):
        det, _, _ = dem.compile_sampler(seed=CALIBRATION_SEED).sample(CALIBRATION_SHOTS)
        return float(det.sum(axis=1).mean())

    original = sample_mean(simple_dem)
    # XOR marginals increase in expectation while each independent error probability is at most 1/2.
    max_alpha = max(1.0, MAX_ERROR_PROBABILITY / min(positive))
    low, high, alpha = 0.0, None, 1.0
    for _ in range(64):
        calibrated = scale_dem(instructions, alpha)
        achieved = sample_mean(calibrated)
        relative_error = abs(achieved - target) / target
        if relative_error < CALIBRATION_RTOL:
            return calibrated, {
                "source_dem": "simple", "dem": CALIBRATED_DEM, "fit_split": "train",
                "alpha": alpha, "target_mean_lit_detectors": target,
                "original_mean_lit_detectors": original, "achieved_mean_lit_detectors": achieved,
                "relative_error": relative_error, "relative_tolerance": CALIBRATION_RTOL,
                "seed": CALIBRATION_SEED, "shots_per_probe": CALIBRATION_SHOTS,
                "max_error_probability": MAX_ERROR_PROBABILITY,
            }
        if achieved < target:
            low = alpha
            if high is None:
                if alpha == max_alpha:
                    break
                alpha = min(2 * alpha, max_alpha)
                continue
        else:
            high = alpha
        alpha = (low + high) / 2
    raise ValueError(f"Could not calibrate simple.dem to train mean {target:.5f}; last mean={achieved:.5f}")


def predict_pm(dem, det, correlated):
    matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)
    return matcher.decode_batch(det, enable_correlations=correlated)[:, 0].astype(bool)


def score_baselines(dems, det, obs, google, test_idx):
    rows, test_errors = [], {}

    def record(name, family, dem, decoder, correlated, actual_pred):
        wrong = actual_pred != obs
        test_errors[name] = wrong[test_idx]
        ler_test, se_test = ler_and_se(test_errors[name])
        rows.append({
            "name": name, "family": family, "dem": dem, "decoder": decoder,
            "correlated": correlated,
            "prior_fitted_on_this_data": {"": False, "simple": False, CALIBRATED_DEM: "train_only"}.get(dem, "unknown"),
            "ler_test": ler_test, "se_test": se_test, "ler_all": float(wrong.mean()),
        })

    def record_pm(dem):
        for corr, label in ((False, "PM"), (True, "PM-corr")):
            prediction = predict_pm(dems[dem], det, corr)
            record(f"{label}+{dem}", "pm", dem, "pymatching", corr, prediction)

    record("none", "none", "", "constant_zero", False, False)
    for dem in PM_DEMS:
        record_pm(dem)
    for (dem, decoder), prediction in google.items():
        record(f"Google {decoder}+{dem}", "google", dem, decoder, "", prediction)
    record_pm(CALIBRATED_DEM)

    expected = {
        "PM+simple": (0.17513, 0.0001), "none": (0.29711, 0.000005),
        "Google harmony+simple": (0.15359, 0.000005),
        "Google correlated_matching+rl_shared_correlated_matching": (0.14525, 0.000005),
    }
    for row in rows:
        if row["name"] in expected:
            target, tolerance = expected[row["name"]]
            assert abs(row["ler_all"] - target) <= tolerance, row
            print(f"{row['name']}: all-shot LER={row['ler_all']:.5f}", flush=True)

    google_best = min(
        (row for row in rows if row["family"] == "google"),
        key=lambda row: (row["ler_test"], row["name"]),
    )["name"]
    comparators = {
        "pm_simple": "PM+simple",
        "pm_corr_simple": "PM-corr+simple",
        "pm_corr_correlations": "PM-corr+correlations",
        "google_correlated_matching_rl_isolated": "Google correlated_matching+rl_isolated_correlated_matching",
        "google_best": google_best,
        CALIBRATED_COMPARATOR: f"PM-corr+{CALIBRATED_DEM}",
    }
    print(f"Lowest test LER among Google baselines: {google_best}", flush=True)
    return rows, test_errors, comparators


def best_hparams():
    with TUNE_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    pick = {}
    for kind in MODELS:
        row = min((row for row in rows if row["kind"] == kind), key=lambda row: float(row["best_val_ler"]))
        pick[kind] = {"n_layers": int(row["n_layers"]), "lr": float(row["lr"])}
    return pick


def make_model(kind, seed, hparams, stab, rnd, device):
    torch.manual_seed(seed)
    layers = hparams[kind]["n_layers"]
    if kind == "transformer":
        return TransformerDecoder(stab, rnd, n_layers=layers).to(device)
    return MLPDecoder(len(stab), n_layers=layers).to(device)


def evaluate(model, data, device):
    det, obs = data
    return predict(model, det, device, batch=EVAL_BATCH) != obs


def copy_weights(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_batches(model, opt, loss_fn, data, batch, lr, warmup, total_steps, step, device):
    """Train on one epoch or DEM chunk, with the E1 learning-rate schedule."""
    det, obs = data
    order = torch.randperm(len(obs)).to(device)
    det = torch.as_tensor(det, device=device)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    model.train()
    loss_sum = torch.zeros((), device=device)
    for idx in order.split(batch):
        if step < warmup:
            rate = lr * (step + 1) / warmup
        else:
            frac = (step - warmup) / max(1, total_steps - warmup)
            rate = lr * 0.5 * (1 + math.cos(math.pi * frac))
        for group in opt.param_groups:
            group["lr"] = rate
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(det[idx]), obs[idx])
        loss.backward()
        opt.step()
        loss_sum += loss.detach() * len(idx)
        step += 1
    return loss_sum.item() / len(obs), step


def train_from_dem(model, dem, val_data, *, n_shots, check_every, lr, seed, device):
    """Pretrain with fresh DEM shots and select by real validation LER."""
    sampler = dem.compile_sampler(seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    chunks, tail = divmod(n_shots, check_every)
    total_steps = chunks * math.ceil(check_every / PRETRAIN_BATCH) + math.ceil(tail / PRETRAIN_BATCH)
    warmup = max(1, int(TRAIN_WARMUP * total_steps))
    history, best, step = [], (math.inf, None), 0
    start = time.perf_counter()
    for offset in range(0, n_shots, check_every):
        count = min(check_every, n_shots - offset)
        chunk_start = time.perf_counter()
        det, obs, _ = sampler.sample(count)
        loss, step = train_batches(
            model, opt, loss_fn, (det, obs[:, 0]), PRETRAIN_BATCH, lr, warmup, total_steps, step, device,
        )
        train_s = time.perf_counter() - chunk_start
        val_ler = float(evaluate(model, val_data, device).mean())
        seen = offset + count
        history.append({
            "seen": seen, "train_loss": loss, "val_ler": val_ler,
            "sampling_and_training_time_s": train_s, "time_s": time.perf_counter() - start,
        })
        print(f"  seen={seen:,} loss={loss:.4f} val_LER={val_ler:.5f} ({history[-1]['time_s']:.1f}s)", flush=True)
        if val_ler < best[0]:
            best = val_ler, copy_weights(model)
    model.load_state_dict(best[1])
    return history


def train_real_data(model, train_data, val_data, *, regime, lr, max_epochs, device):
    """Train on shuffled real shots, keeping the best real validation checkpoint."""
    fine_tune = regime in FINETUNE_FROM
    warmup_fraction, patience = (FT_WARMUP, PATIENCE_FT) if fine_tune else (TRAIN_WARMUP, PATIENCE_REAL)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    total_steps = max_epochs * math.ceil(len(train_data[1]) / REAL_BATCH)
    warmup = max(1, int(warmup_fraction * total_steps))
    history, best, best_epoch, step = [], (math.inf, None), 0, 0
    start = time.perf_counter()
    if fine_tune:
        val_ler = float(evaluate(model, val_data, device).mean())
        best = val_ler, copy_weights(model)
        history.append({"epoch": 0, "train_loss": None, "val_ler": val_ler, "time_s": time.perf_counter() - start})
        print(f"  epoch=0 val_LER={val_ler:.5f}", flush=True)
    for epoch in range(1, max_epochs + 1):
        loss, step = train_batches(
            model, opt, loss_fn, train_data, REAL_BATCH, lr, warmup, total_steps, step, device,
        )
        val_ler = float(evaluate(model, val_data, device).mean())
        history.append({"epoch": epoch, "train_loss": loss, "val_ler": val_ler, "time_s": time.perf_counter() - start})
        print(f"  epoch={epoch} loss={loss:.4f} val_LER={val_ler:.5f} ({history[-1]['time_s']:.1f}s)", flush=True)
        if val_ler < best[0]:
            best, best_epoch = (val_ler, copy_weights(model)), epoch
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best[1])
    return history


def mcnemar(a_wrong, b_wrong):
    b = int((a_wrong & ~b_wrong).sum())
    c = int((~a_wrong & b_wrong).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def comparison_stats(wrong, reference, tag):
    b, c, p = mcnemar(wrong, reference)
    return {f"ratio_{tag}": float(wrong.mean()) / float(reference.mean()),
            f"b_{tag}": b, f"c_{tag}": c, f"p_{tag}": p}


def run_stem(kind, regime, seed):
    return f"{kind}_{regime}_s{seed}"


def load_test_errors(pred_dir, stem, suffix):
    packed = np.load(pred_dir / f"{stem}{suffix}.npy", allow_pickle=False)
    return np.unpackbits(packed, count=N_TEST).astype(bool)


def pairwise_rows(done, pred_dir, suffix):
    wrong = {stem: load_test_errors(pred_dir, stem, suffix) for stem in done}
    seeds = sorted({int(row["seed"]) for row in done.values()})
    pairs = []
    for seed in seeds:
        for kind in MODELS:
            for a, b in (("ft", "pre0"), ("ft", "real"), ("ftcal", "pcal"), ("ftcal", "ft"), ("pcal", "pre0")):
                pairs.append((run_stem(kind, a, seed), run_stem(kind, b, seed)))
        for regime in REGIMES:
            pairs.append((run_stem("transformer", regime, seed), run_stem("mlp", regime, seed)))
    for a, b in pairs:
        if a in wrong and b in wrong:
            ab, ba, p = mcnemar(wrong[a], wrong[b])
            yield {"a": a, "b": b, "n_a_wrong_b_right": ab, "n_a_right_b_wrong": ba, "p": p}


def sim_check_rows(dems, done, model_dir, suffix, hparams, stab, rnd, device):
    for dem, regime in (("simple", "pre0"), (CALIBRATED_DEM, "pcal")):
        det, obs, _ = dems[dem].compile_sampler(seed=SIM_TEST_SEED).sample(SIM_TEST_SHOTS)
        obs = obs[:, 0]
        for correlated, decoder in ((False, "PM"), (True, "PM-corr")):
            wrong = predict_pm(dems[dem], det, correlated) != obs
            yield {"dem": dem, "decoder": decoder, "correlated": correlated, "n_shots": len(obs),
                   "ler": float(wrong.mean())}
        for stem, row in done.items():
            if row["regime"] != regime:
                continue
            kind, seed = row["model"], int(row["seed"])
            model = make_model(kind, seed, hparams, stab, rnd, device)
            checkpoint = torch.load(model_dir / f"{stem}{suffix}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            wrong = evaluate(model, (det, obs), device)
            yield {"dem": dem, "decoder": "NN", "model": kind, "regime": regime, "seed": seed,
                   "n_shots": len(obs), "ler": float(wrong.mean())}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_previous_settings(previous, current, prefix=""):
    """Allow added keys, including nested keys, while preserving previous values."""
    for key, value in previous.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            check_previous_settings(value, current[key], f"{prefix}{key}.")
        elif key not in current or value != current[key]:
            raise ValueError(f"{prefix}{key}")


def save_settings(path, args, hparams, layout, device, comparators, calibration, regimes):
    hashes = {p.relative_to(PACKET).as_posix(): sha256(p) for p in sorted(PACKET.rglob("*")) if p.is_file()}
    settings = {
        "script": "src/run_e2_sycamore.py",
        "out_suffix": args.out_suffix,
        "packet_file_sha256": hashes,
        "packet_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "e1_tune_sha256": sha256(TUNE_CSV),
        "split_seed": SPLIT_SEED, "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "n_shots": N_SHOTS, "n_detectors": N_DETECTORS, "drift_block": DRIFT_BLOCK,
        "torch_seed": "run seed, reset before constructing each model",
        "dem_seed": "run seed, shared by both models", "pretrain_dem": "simple",
        "hparams": hparams, "detector_layout": layout, "eval_batch": EVAL_BATCH,
        "architecture": {"transformer": {"d_model": 64, "n_heads": 4, "d_ff": 128, "dropout": 0.0},
                         "mlp": {"hidden": 256, "activation": "GELU"}},
        "optimizer": "AdamW", "loss": "BCEWithLogitsLoss", "weight_decay": WEIGHT_DECAY,
        "lr_schedule": "linear warmup then cosine decay over planned optimizer steps",
        "pre0": {"n_pretrain": args.n_pretrain, "batch": PRETRAIN_BATCH,
                 "val_every_shots": PRETRAIN_CHECK_EVERY, "warmup": TRAIN_WARMUP},
        "real": {"max_epochs": args.max_epochs_real, "batch": REAL_BATCH,
                 "patience": PATIENCE_REAL, "warmup": TRAIN_WARMUP},
        "ft": {"max_epochs": args.max_epochs_ft, "batch": REAL_BATCH, "patience": PATIENCE_FT,
               "lr": FT_LR, "warmup": FT_WARMUP, "validate_epoch_zero": True},
        "pcal": {"n_pretrain": args.n_pretrain_cal, "batch": PRETRAIN_BATCH,
                 "val_every_shots": PRETRAIN_CAL_CHECK_EVERY, "warmup": TRAIN_WARMUP, "dem": CALIBRATED_DEM},
        "ftcal": {"same_training_as": "ft", "initialized_from": "pcal"},
        "calibration": calibration,
        "sim_check": {"seed": SIM_TEST_SEED, "shots_per_dem": SIM_TEST_SHOTS,
                      "dem_regimes": {"simple": "pre0", CALIBRATED_DEM: "pcal"}},
        "test_comparators": comparators, "mcnemar": "two-sided exact binomial",
        "pm_dems": list(PM_DEMS), "google_decoders": list(GOOGLE_DECODERS),
        "baseline_correlated_column": "PyMatching correlation flag; blank for Google predictions",
        "packed_bitorder": "big", "device": str(device), "torch_cpu_threads": torch.get_num_threads(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "stim": stim.__version__,
                     "pymatching": pymatching.__version__, "torch": torch.__version__, "scipy": scipy.__version__},
    }
    selection = {"models": set(args.models), "regimes": regimes, "seeds": set(args.seeds)}
    previous_selection = {}
    if path.exists():
        previous = json.loads(path.read_text())
        previous_selection = previous.pop("selection")
        try:
            check_previous_settings(previous, settings)
        except ValueError as mismatch:
            raise ValueError(f"Experiment setting {mismatch} differs from {path.name}; use a separate --out-suffix") from mismatch
    settings["selection"] = {
        key: previous_selection.get(key, []) + sorted(values.difference(previous_selection.get(key, [])))
        for key, values in selection.items()
    }
    pending = path.with_name(f".pending_{path.name}")
    pending.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def run(args, model_dir, pred_dir):
    suffix = args.out_suffix
    circuit, dems, det, obs, google = load_packet()
    train_idx, val_idx, test_idx = split_indices()
    train_data = det[train_idx], obs[train_idx]
    val_data = det[val_idx], obs[val_idx]
    test_data = det[test_idx], obs[test_idx]
    stab, rnd = detector_layout(circuit)
    final = rnd == rnd.max()
    assert np.unique(stab[final]).size == final.sum() == 4
    assert not np.intersect1d(stab[final], stab[~final]).size
    layout = {"n_stabs": int(stab.max()) + 1, "n_rounds": int(rnd.max()) + 1,
              "final_stabs": stab[final].tolist()}
    hparams, device = best_hparams(), default_device()
    print(f"Device: {device}; hyperparameters: {hparams}; layout: {layout}", flush=True)
    calibrated_dem, calibration = calibrate_dem(dems["simple"], train_data[0])
    dems[CALIBRATED_DEM] = calibrated_dem
    dem_path = RESULTS / f"e2_{CALIBRATED_DEM}{suffix}.dem"
    dem_bytes = (str(calibrated_dem) + "\n").encode("utf-8")
    calibration["file"] = dem_path.name
    calibration["sha256"] = hashlib.sha256(dem_bytes).hexdigest()
    print(f"Calibration: alpha={calibration['alpha']:.6g}, "
          f"simple={calibration['original_mean_lit_detectors']:.5f}, "
          f"train={calibration['target_mean_lit_detectors']:.5f}, "
          f"calibrated={calibration['achieved_mean_lit_detectors']:.5f}", flush=True)
    baselines, baseline_wrong, comparators = score_baselines(dems, det, obs, google, test_idx)
    regimes = set(args.regimes)
    regimes.update(FINETUNE_FROM[regime] for regime in args.regimes if regime in FINETUNE_FROM)
    save_settings(RESULTS / f"e2_sycamore{suffix}.settings.json", args, hparams, layout, device,
                  comparators, calibration, regimes)
    pending_dem = dem_path.with_name(f".pending_{dem_path.name}")
    pending_dem.write_bytes(dem_bytes)
    pending_dem.replace(dem_path)

    np.savez_compressed(RESULTS / f"e2_split{suffix}.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    write_csv(RESULTS / f"e2_drift{suffix}.csv",
              ("shot_start", "shot_stop", "mean_lit_detectors", "undecoded_flip_rate"), drift_rows(det, obs))
    write_csv(RESULTS / f"e2_baselines{suffix}.csv", baselines[0].keys(), baselines)
    packed_baselines = {name: np.packbits(wrong) for name, wrong in baseline_wrong.items()}
    np.savez_compressed(pred_dir / f"baselines{suffix}.npz", truth=np.packbits(test_data[1]), **packed_baselines)

    nn_csv = RESULTS / f"e2_nn{suffix}.csv"
    fields = NN_FIELDS + tuple(f"{stat}_{tag}" for tag in comparators for stat in ("ratio", "b", "c", "p"))
    done = {}
    if nn_csv.exists():
        with nn_csv.open(newline="", encoding="utf-8") as fh:
            done = {run_stem(row["model"], row["regime"], row["seed"]): row for row in csv.DictReader(fh)}
        missing = [stem for stem, row in done.items() if not row.get(f"p_{CALIBRATED_COMPARATOR}")]
        if missing:
            reference = baseline_wrong[comparators[CALIBRATED_COMPARATOR]]
            for stem in missing:
                wrong = load_test_errors(pred_dir, stem, suffix)
                done[stem].update(comparison_stats(wrong, reference, CALIBRATED_COMPARATOR))
            write_csv(nn_csv, fields, done.values())
            print(f"Added calibrated comparator to {len(missing)} completed rows", flush=True)

    pretrain = {
        "pre0": ("simple", args.n_pretrain, PRETRAIN_CHECK_EVERY),
        "pcal": (CALIBRATED_DEM, args.n_pretrain_cal, PRETRAIN_CAL_CHECK_EVERY),
    }
    for kind in dict.fromkeys(args.models):
        for seed in dict.fromkeys(args.seeds):
            for regime in REGIMES:
                if regime not in regimes:
                    continue
                stem = run_stem(kind, regime, seed)
                if stem in done:
                    print(f"{stem}: already complete", flush=True)
                    continue
                model = make_model(kind, seed, hparams, stab, rnd, device)
                print(stem, flush=True)
                n_pretrain = 0
                if regime in pretrain:
                    dem, n_pretrain, check_every = pretrain[regime]
                    history = train_from_dem(model, dems[dem], val_data, n_shots=n_pretrain,
                                             check_every=check_every, lr=hparams[kind]["lr"], seed=seed, device=device)
                else:
                    lr, max_epochs = hparams[kind]["lr"], args.max_epochs_real
                    if regime in FINETUNE_FROM:
                        parent = run_stem(kind, FINETUNE_FROM[regime], seed)
                        checkpoint = torch.load(model_dir / f"{parent}{suffix}.pt", map_location="cpu", weights_only=True)
                        model.load_state_dict(checkpoint["state_dict"])
                        n_pretrain = checkpoint["n_pretrain"]
                        lr, max_epochs = FT_LR, args.max_epochs_ft
                    history = train_real_data(model, train_data, val_data, regime=regime, lr=lr,
                                              max_epochs=max_epochs, device=device)

                wrong = evaluate(model, test_data, device)
                ler, se = ler_and_se(wrong)
                row = {
                    "model": kind, "regime": regime, "seed": seed, "n_params": n_params(model),
                    "n_pretrain": n_pretrain, "epochs_run": history[-1].get("epoch", 0),
                    "best_val_ler": min(entry["val_ler"] for entry in history), "ler_test": ler,
                    "se_test": se, "train_time_s": history[-1]["time_s"],
                }
                for tag, name in comparators.items():
                    row.update(comparison_stats(wrong, baseline_wrong[name], tag))
                stem_with_suffix = f"{stem}{suffix}"
                torch.save({"state_dict": copy_weights(model), "kind": kind, "regime": regime,
                            "seed": seed, "hparams": hparams[kind], "n_pretrain": n_pretrain},
                           model_dir / f"{stem_with_suffix}.pt")
                (model_dir / f"{stem_with_suffix}.history.json").write_text(json.dumps(history, indent=2) + "\n")
                np.save(pred_dir / f"{stem_with_suffix}.npy", np.packbits(wrong))
                done[stem] = row
                write_csv(nn_csv, fields, done.values())
                print(f"  test_LER={ler:.5f}", flush=True)

    write_csv(RESULTS / f"e2_pairwise{suffix}.csv", ("a", "b", "n_a_wrong_b_right", "n_a_right_b_wrong", "p"),
              pairwise_rows(done, pred_dir, suffix))
    write_csv(RESULTS / f"e2_sim_check{suffix}.csv",
              ("dem", "decoder", "correlated", "model", "regime", "seed", "n_shots", "ler"),
              sim_check_rows(dems, done, model_dir, suffix, hparams, stab, rnd, device))
    print("done", flush=True)


def main():
    args = parse_args()
    model_dir = RESULTS / f"e2_models{args.out_suffix}"
    pred_dir = RESULTS / f"e2_preds{args.out_suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    # Serialize writes for the same suffix without creating a lock file.
    fd = os.open(model_dir, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        run(args, model_dir, pred_dir)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
