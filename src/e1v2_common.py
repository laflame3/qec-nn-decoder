"""Shared settings, paired metrics, and training for the E1 v2 experiments."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pymatching
import scipy
import stim
import torch
from scipy.stats import binomtest
from torch import nn
from torch.torch_version import TorchVersion

from decoders import build_pm_correlated, decode_pm_correlated
from model import detector_layout, predict
from noise import correlated_memory_circuit
from run_e2_sycamore import copy_weights, make_model, train_batches

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"
LEGACY_PREDS = RESULTS / "e1_preds"
LEGACY_MODELS = RESULTS / "e1_models"
MODELS = ("transformer", "mlp")
PM = "PM-correlated"
P_BG = 0.005
BATCH = 1024
EVAL_BATCH = 4096
WEIGHT_DECAY = 1e-4
TORCH_SEED = 0
POOL_SEED = 74000
TARGET_TEST_SEED = 20261125
TARGET_VAL_SEED = 30004
N_TEST = 1_000_000
N_VAL = 200_000
N_LEGACY_TRAIN = 10_000_000
STAT_FIELDS = (
    "pl", "pl_se", "pm_pl", "delta", "delta_se", "delta_ci_low", "delta_ci_high",
    "ratio", "ratio_ci_low", "ratio_ci_high", "b", "c", "mcnemar_p",
)


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Use smaller samples and separate outputs.")
    mode.add_argument("--verify", action="store_true", help="Validate the existing formal outputs without writing.")
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def best_hparams():
    """Select the lowest validation LER in the tuning CSV as in the E1 f-scan."""
    rows = read_rows(RESULTS / "e1_tune.csv")
    picked = {}
    for kind in MODELS:
        row = min((r for r in rows if r["kind"] == kind), key=lambda r: float(r["best_val_ler"]))
        picked[kind] = {"n_layers": int(row["n_layers"]), "lr": float(row["lr"])}
    return picked


def mcnemar(nn_wrong, pm_wrong):
    b = int((nn_wrong & ~pm_wrong).sum())
    c = int((~nn_wrong & pm_wrong).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def error_rate(wrong):
    pl = float(wrong.mean())
    return pl, math.sqrt(pl * (1 - pl) / wrong.size)


def paired_stats(nn_wrong, pm_wrong):
    """Compare two decoders on identical, ordered test shots."""
    nn_wrong, pm_wrong = np.asarray(nn_wrong, dtype=bool), np.asarray(pm_wrong, dtype=bool)
    if nn_wrong.ndim != 1 or nn_wrong.shape != pm_wrong.shape or nn_wrong.size < 2:
        raise ValueError("paired statistics require equally sized 1D masks with at least two shots")
    n = nn_wrong.size
    pl, pl_se = error_rate(nn_wrong)
    pm_pl, _ = error_rate(pm_wrong)
    b, c, p = mcnemar(nn_wrong, pm_wrong)
    delta = (b - c) / n
    delta_se = math.sqrt(b + c - (b - c) ** 2 / n) / n
    ratio = ci_low = ci_high = None
    if pm_pl:
        ratio = pl / pm_pl
        # e1_ratio_ci.py: estimate the variance of the per-shot ratio residual.
        residual = nn_wrong - ratio * pm_wrong
        se = residual.std(ddof=1) / (math.sqrt(n) * pm_pl)
        half_width = NormalDist().inv_cdf(0.975) * se
        ci_low, ci_high = float(ratio - half_width), float(ratio + half_width)
    return {
        "pl": pl, "pl_se": pl_se, "pm_pl": pm_pl,
        "delta": delta, "delta_se": delta_se,
        "delta_ci_low": delta - 1.96 * delta_se, "delta_ci_high": delta + 1.96 * delta_se,
        "ratio": ratio, "ratio_ci_low": ci_low, "ratio_ci_high": ci_high,
        "b": b, "c": c, "mcnemar_p": p,
    }


def sample(circuit, shots, seed):
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return det, obs[:, 0]


def target_circuit():
    return correlated_memory_circuit(P_BG, 1.0)[0]


def sample_pool(n_max, seed=POOL_SEED):
    """Sample the f=1 pool in one call so all prefixes share shot order."""
    return sample(target_circuit(), n_max, seed)


def prefix_sha256(pool, n):
    det, obs = pool
    if not 0 < n <= len(obs):
        raise ValueError(f"invalid training pool prefix size: {n}")
    digest = hashlib.sha256()
    digest.update(det[:n].tobytes(order="C"))
    digest.update(obs[:n].tobytes(order="C"))
    return digest.hexdigest()


def unpack_bits(packed, n):
    if packed.shape != (math.ceil(n / 8),) or packed.dtype != np.uint8:
        raise ValueError(f"packed mask has wrong shape or dtype for {n:,} shots")
    if n % 8 and int(packed[-1]) & ((1 << (8 - n % 8)) - 1):
        raise ValueError("packed mask has nonzero padding bits")
    return np.unpackbits(packed, count=n).astype(bool)


def legacy_pm(f):
    with np.load(LEGACY_PREDS / f"pm_f{f:.2f}.npz", allow_pickle=False) as file:
        return unpack_bits(file["truth"], N_TEST), unpack_bits(file[PM], N_TEST)


def legacy_wrong(kind, f):
    path = LEGACY_PREDS / f"{kind}_f{f:.2f}_s0.npy"
    return unpack_bits(np.load(path, allow_pickle=False), N_TEST)


def legacy_row(kind, f):
    return next(r for r in read_rows(RESULTS / "e1_nn_fscan.csv")
                if float(r["f"]) == f and r["model"] == kind and int(r["seed"]) == TORCH_SEED)


def load_checkpoint(path):
    """Read new checkpoints and older ones containing torch.__version__."""
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def legacy_model(kind, f, circuit, device):
    path = LEGACY_MODELS / f"{kind}_f{f:.2f}_s0.pt"
    checkpoint = load_checkpoint(path)
    if (checkpoint["kind"], checkpoint["f"], checkpoint["p"], checkpoint["seed"]) != (
        kind, f, P_BG, TORCH_SEED
    ):
        raise ValueError(f"unexpected checkpoint metadata in {path}")
    model = make_model(kind, TORCH_SEED, {kind: checkpoint["hparams"]}, *detector_layout(circuit), device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def target_data(circuit, smoke):
    n_test, n_val = test_sizes(smoke)
    test_det, truth = sample(circuit, n_test, TARGET_TEST_SEED)
    if smoke:
        pm_wrong = decode_pm_correlated(build_pm_correlated(circuit), test_det) != truth
    else:
        saved_truth, pm_wrong = legacy_pm(1.0)
        np.testing.assert_array_equal(truth, saved_truth, err_msg="f=1 legacy test truth differs")
    val_data = sample(circuit, n_val, TARGET_VAL_SEED)
    return (test_det, truth), val_data, pm_wrong


def test_sizes(smoke):
    return (20_000, 5_000) if smoke else (N_TEST, N_VAL)


def evaluate(model, data, device, batch=EVAL_BATCH):
    det, truth = data
    return predict(model, det, device, batch=batch) != truth


def train_fixed(model, train_data, val_data, *, lr, max_epochs, patience,
                warmup_fraction, device, evaluate_initial=False):
    """Shuffle a fixed prefix each epoch and restore the best validation weights."""
    torch.manual_seed(TORCH_SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train_data[1])
    total_steps = max_epochs * math.ceil(n / BATCH)
    warmup = max(1, int(warmup_fraction * total_steps))
    history, best_state = [], None
    best_pl, best_epoch, step = math.inf, 0, 0
    start = time.perf_counter()

    def validate(epoch, loss):
        nonlocal best_pl, best_epoch, best_state
        val_pl = float(evaluate(model, val_data, device).mean())
        history.append({"epoch": epoch, "seen": epoch * n, "train_loss": loss,
                        "val_ler": val_pl, "time_s": time.perf_counter() - start})
        print(f"  epoch={epoch} val_P_L={val_pl:.5f}", flush=True)
        if val_pl < best_pl:
            best_pl, best_epoch, best_state = val_pl, epoch, copy_weights(model)

    if evaluate_initial:
        validate(0, None)
    for epoch in range(1, max_epochs + 1):
        loss, step = train_batches(
            model, opt, loss_fn, train_data, BATCH, lr, warmup, total_steps, step, device,
        )
        validate(epoch, loss)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return history


def history_summary(history):
    best = min(history, key=lambda entry: entry["val_ler"])
    return {
        "epochs_run": history[-1]["epoch"], "best_epoch": best["epoch"],
        "best_val_pl": best["val_ler"], "train_time_s": history[-1]["time_s"],
    }


def common_settings(experiment, smoke, hparams, device):
    return {
        "script": f"src/run_e1_{experiment}.py", "smoke": smoke,
        "circuit": "surface_code:rotated_memory_z", "distance": 3,
        "rounds": 3, "basis": "Z", "n_detectors": 24, "p_bg": P_BG,
        "background_channels": ["after_clifford_depolarization",
                                "after_reset_flip_probability", "before_measure_flip_probability"],
        "pm": {"name": PM, "dem": "basis_decomposed_dem", "enable_correlations": True},
        "torch_seed": TORCH_SEED, "hyperparameters": hparams, "device": str(device),
        "batch": BATCH, "evaluation_batch": EVAL_BATCH, "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY, "loss": "BCEWithLogitsLoss",
        "validation_selection": "earliest minimum validation P_L",
        "prediction_rule": "logit > 0", "prediction_bits": "NumPy packbits with big bit order",
        "statistics": {"delta": "paired NN minus PM, 95% Wald CI",
                       "ratio": "paired delta method, 95% CI",
                       "mcnemar": "exact binomial, two-sided"},
        "versions": {name: str(version) for name, version in (
            ("python", platform.python_version()), ("stim", stim.__version__),
            ("torch", torch.__version__), ("numpy", np.__version__),
            ("pymatching", pymatching.__version__), ("scipy", scipy.__version__),
        )},
    }


def prefix_settings(smoke, sizes, n_max, pool):
    return {"n_pool": n_max, "pool_seed": POOL_SEED, "prefix_sizes": list(sizes),
            "prefix_sha256": {str(n): prefix_sha256(pool, n) for n in sizes},
            "prefix_hash_rule": "SHA-256 of detector bytes followed by observable bytes in C order",
            "target_f": 1.0, "test_seed": TARGET_TEST_SEED, "val_seed": TARGET_VAL_SEED,
            "n_test": test_sizes(smoke)[0], "n_val": test_sizes(smoke)[1],
            "pm_test_source": "fresh reduced sample" if smoke else "legacy f=1 with truth assertion"}


def atomic_write(path, write, *, exclusive=False):
    """Finish writing a new artifact before exposing it to a resumed run."""
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as file:
        temp = Path(file.name)
        try:
            write(file)
            file.flush()
            os.fsync(file.fileno())
            if exclusive:
                os.link(temp, path)
            else:
                temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


@dataclass
class ResultRow:
    columns: dict
    stem: str
    checkpoint: dict | None = None
    source_weights: Path | None = None
    reference_wrong: np.ndarray | None = None
    max_epochs: int | None = None


class Results:
    """One experiment's preregistration, append-only rows and artifacts."""

    def __init__(self, experiment, smoke, fields, key_fields):
        self.suffix = "_smoke" if smoke else ""
        stem = f"e1_{experiment}{self.suffix}"
        self.csv_path = RESULTS / f"{stem}.csv"
        self.settings_path = RESULTS / f"{stem}.settings.json"
        self.pred_dir = RESULTS / f"e1v2_preds{self.suffix}"
        self.model_dir = RESULTS / f"e1v2_models{self.suffix}"
        self.fields = fields
        self.key_fields = key_fields
        self.rows = {}

    def recorded_device(self):
        return json.loads(self.settings_path.read_text(encoding="utf-8"))["device"]

    def pred_path(self, stem):
        return self.pred_dir / f"{stem}{self.suffix}.npy"

    def model_path(self, stem):
        return self.model_dir / f"{stem}{self.suffix}.pt"

    def read_wrong(self, stem, n_test):
        path = self.pred_path(stem)
        try:
            return unpack_bits(np.load(path, allow_pickle=False), n_test)
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"invalid prediction mask {path}: {exc}") from exc

    @staticmethod
    def check_columns(row, expected):
        for name, value in expected.items():
            actual = row[name]
            if isinstance(value, float) and actual:
                equal = math.isclose(float(actual), value, rel_tol=1e-12)
            else:
                equal = actual == ("" if value is None else str(value))
            if not equal:
                raise ValueError(f"{name}: expected {value!r}, found {actual!r}")

    def validate_row(self, row, spec, settings):
        expected = dict(spec.columns)
        self.check_columns(row, expected)
        for name in STAT_FIELDS:
            if name in expected and expected[name] is None:
                continue
            if name.startswith("ratio") and row["pm_pl"] and float(row["pm_pl"]) == 0:
                continue
            if not row[name].strip():
                raise ValueError(f"required statistic {name} is empty")
        wrong = self.read_wrong(spec.stem, expected["n_test"])
        if spec.reference_wrong is not None and not np.array_equal(wrong, spec.reference_wrong):
            raise ValueError(f"prediction mask {self.pred_path(spec.stem)} differs from its reference bit-for-bit")
        pl, pl_se = error_rate(wrong)
        self.check_columns(row, {"pl": pl, "pl_se": pl_se})

        if spec.source_weights is not None and not spec.source_weights.is_file():
            raise ValueError(f"missing source weights: {spec.source_weights}")
        if spec.checkpoint is not None:
            path = self.model_path(spec.stem)
            if not path.is_file():
                raise ValueError(f"missing weights: {path}")
            checkpoint = load_checkpoint(path)
            for name, value in {**spec.checkpoint, "settings": settings}.items():
                if checkpoint.get(name) != value:
                    raise ValueError(f"checkpoint {name} differs from run configuration: {path}")
            if not checkpoint.get("state_dict"):
                raise ValueError(f"checkpoint is missing its state_dict: {path}")
            history = json.loads(path.with_suffix(".history.json").read_text(encoding="utf-8"))
            if spec.max_epochs is None:
                summary = {"n_train": history[-1]["seen"], "train_time_s": history[-1]["time_s"],
                           "best_val_pl": min(entry["val_ler"] for entry in history)}
            else:
                summary = history_summary(history)
                if not 1 <= summary["epochs_run"] <= spec.max_epochs:
                    raise ValueError(f"epochs_run exceeds configured range of 1–{spec.max_epochs}")
                if "samples_seen" in row:
                    summary["samples_seen"] = expected["n_train_unique"] * summary["epochs_run"]
            self.check_columns(row, summary)
            expected.update(summary)
        for name, actual in row.items():
            if name not in expected:
                if name.startswith("ratio") and float(row["pm_pl"]) == 0 and not actual:
                    continue
                if not actual.strip() or not math.isfinite(float(actual)):
                    raise ValueError(f"required numeric column {name} is empty or non-finite")

    @contextmanager
    def open(self, settings, specs=None, *, read_only=False, require_complete=False):
        # specs=None keeps the original behaviour for other callers (e.g. src/run_e2_v2.py):
        # rows are deduplicated by key but not validated against a plan.
        if specs is None and require_complete:
            raise ValueError("require_complete needs the planned specs")
        planned = None if specs is None else {self.key(spec.columns): spec for spec in specs}
        if planned is not None and len(planned) != len(specs):
            raise ValueError("duplicate planned result key")
        if not read_only and not self.settings_path.exists():
            if self.csv_path.exists() and not self.settings_path.exists():
                raise ValueError(f"missing settings for existing results: {self.csv_path}")
            payload = (json.dumps(settings, indent=2, sort_keys=True) + "\n").encode()
            try:
                atomic_write(self.settings_path, lambda file: file.write(payload), exclusive=True)
            except FileExistsError:
                pass  # Another runner published its settings first.
        # A read-only verifier shares the lock; a writer holds it until publication.
        with self.settings_path.open("r" if read_only else "r+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_SH if read_only else fcntl.LOCK_EX)
            previous = json.load(file)
            if previous != settings:
                different = sorted(k for k in previous.keys() | settings.keys()
                                   if previous.get(k) != settings.get(k))
                raise ValueError(f"preregistered settings differ in {', '.join(different)}: {self.settings_path}")

            self.rows.clear()
            if self.csv_path.exists():
                with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
                    reader = csv.DictReader(csv_file)
                    if tuple(reader.fieldnames or ()) != self.fields:
                        raise ValueError(f"unexpected CSV header: {self.csv_path}")
                    for row in reader:
                        try:
                            if None in row or None in row.values():
                                raise ValueError("incomplete row")
                            key = self.key(row)
                            if key in self.rows:
                                raise ValueError(f"duplicate row {key}")
                            if planned is not None:
                                if key not in planned:
                                    raise ValueError(f"unexpected configuration key {key}")
                                self.validate_row(row, planned[key], settings)
                            self.rows[key] = row
                        except (ValueError, OSError, TypeError, IndexError, KeyError) as exc:
                            raise ValueError(f"{self.csv_path}:{reader.line_num}: {exc}") from exc
            if require_complete:
                if self.rows.keys() != planned.keys():
                    missing = sorted(planned.keys() - self.rows.keys())
                    raise ValueError(f"missing result rows in {self.csv_path}: {missing}")
                checkpoints = sum(spec.checkpoint is not None for spec in specs)
                print(f"PASS {self.csv_path.name}: {len(self.rows)} rows, {len(self.rows)} prediction masks, "
                      f"{checkpoints} new checkpoints validated", flush=True)
            yield self

    def key(self, row):
        return tuple("" if row[field] is None else str(row[field]) for field in self.key_fields)

    def pending(self, **key_fields):
        key = self.key(key_fields)
        if key in self.rows:
            print(f"skip completed {' / '.join(key)}", flush=True)
            return False
        return True

    def save_wrong(self, stem, wrong):
        path = self.pred_path(stem)
        atomic_write(path, lambda file: np.save(file, np.packbits(wrong), allow_pickle=False))

    def save_model(self, stem, model, kind, hparams, history, settings, condition):
        path = self.model_path(stem)
        checkpoint = {"state_dict": copy_weights(model), "kind": kind, "hparams": hparams,
                      "settings": json.loads(json.dumps(settings)), "condition": condition}
        atomic_write(path, lambda file: torch.save(checkpoint, file))
        payload = (json.dumps(history, indent=2) + "\n").encode()
        atomic_write(path.with_suffix(".history.json"), lambda file: file.write(payload))

    def append(self, row):
        key = self.key(row)
        if key in self.rows:
            raise ValueError(f"duplicate result: {key}")
        header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            if header:
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
        self.rows[key] = row
        print(f"saved {' / '.join(key)} P_L={float(row['pl']):.5f}", flush=True)
