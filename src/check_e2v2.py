"""Verify the E2 v2 exports, split provenance, baseline and smoke preregistration."""

import hashlib
import json
import sys

sys.dont_write_bytecode = True

import numpy as np

from e1v2_common import read_rows
from rescore_e2_v2 import PM, load_references, sha256
from run_e2_sycamore import N_DETECTORS, N_SHOTS, N_TRAIN, RESULTS, load_packet, predict_pm
from run_e2_v2 import A_SEED, B_SEED, MODELS, POOL, POOL_SIZE, SAMPLE14, SIZES, SMOKE_POOL_SIZE, SMOKE_SIZES

SAMPLES = tuple(sample for sample in range(53) if sample != SAMPLE14)
STAT_COLUMNS = ("mean_lit_detectors", "undecoded_flip_rate")


def mean_values(row):
    return [float(row[key]) for key in STAT_COLUMNS]


def check_sample_stats(packed, obs, samples, det14, obs14):
    manifest = json.loads((POOL / "manifest.json").read_text(encoding="utf-8"))
    for name in ("pool.npz", "samples.csv"):
        assert sha256(POOL / name) == manifest["output_files"][name], name
    rows = read_rows(POOL / "samples.csv")
    stats = {int(row["sample"]): row for row in rows}
    assert len(rows) == len(stats) == len(SAMPLES) + 1 and set(stats) == set(range(53))

    for offset, sample in enumerate(SAMPLES):
        indices = slice(offset * N_SHOTS, (offset + 1) * N_SHOTS)
        assert np.all(samples[indices] == sample), sample
        bits = np.unpackbits(packed[indices], axis=1, count=N_DETECTORS, bitorder="big")
        np.testing.assert_allclose([bits.sum(axis=1).mean(), obs[indices].mean()],
                                   mean_values(stats[sample]), rtol=0, atol=1e-12, err_msg=str(sample))

    drift = read_rows(RESULTS / "e2_drift.csv")
    starts = [int(row["shot_start"]) for row in drift]
    stops = [int(row["shot_stop"]) for row in drift]
    assert starts[0] == 0 and stops[-1] == N_SHOTS and starts[1:] == stops[:-1]
    sizes = np.subtract(stops, starts)
    assert np.all(sizes > 0)
    drift_means = np.average([mean_values(row) for row in drift], axis=0, weights=sizes)
    lit, flip = mean_values(stats[SAMPLE14])
    assert stats[SAMPLE14]["role"] == "test_sample"
    np.testing.assert_allclose([lit, flip], drift_means, rtol=0, atol=1e-12)
    np.testing.assert_allclose([lit, flip], [det14.sum(axis=1).mean(), obs14.mean()], rtol=0, atol=1e-12)
    print(f"PASS 1: 52 sample summaries agree; sample_14 drift means {lit:.5f}, {flip:.8f}")


def check_pool_order(packed, obs, samples, shots):
    assert packed.shape == (POOL_SIZE, N_DETECTORS // 8) and packed.dtype == np.uint8
    assert obs.shape == samples.shape == shots.shape == (POOL_SIZE,)
    assert obs.dtype == bool and samples.dtype == np.int16 and shots.dtype == np.int32
    expected_shots = np.arange(N_SHOTS, dtype=np.int32)
    np.testing.assert_array_equal(samples.reshape(-1, N_SHOTS)[:, 0], np.array(SAMPLES))
    for sample, sample_block, shot_block in zip(
        SAMPLES, samples.reshape(-1, N_SHOTS), shots.reshape(-1, N_SHOTS), strict=True
    ):
        assert np.all(sample_block == sample), sample
        np.testing.assert_array_equal(shot_block, expected_shots, err_msg=str(sample))
    print(f"PASS 2: {len(SAMPLES)} samples × {N_SHOTS:,} ordered shots; sample_14 absent")


def provenance(samples, shots, split, n_external):
    train, val, test = split
    ext_ids = samples[:n_external].astype(np.int32) * N_SHOTS + shots[:n_external]
    target_ids = SAMPLE14 * N_SHOTS + train
    a_ids = np.concatenate((target_ids, ext_ids))
    order_a = np.random.default_rng(A_SEED).permutation(len(a_ids))
    order_b = np.random.default_rng(B_SEED).permutation(len(ext_ids))
    present_a, present_b = (np.zeros(53 * N_SHOTS, dtype=bool) for _ in range(2))
    present_a[a_ids[order_a]] = True
    present_b[ext_ids[order_b]] = True
    assert present_a.sum() == len(a_ids) and present_b.sum() == len(ext_ids)
    no_leakage = not present_a[SAMPLE14 * N_SHOTS + np.concatenate((val, test))].any()
    present_b[target_ids] = True
    return order_a, order_b, no_leakage, np.array_equal(present_a, present_b)


def check_rescore(reference):
    rows = read_rows(RESULTS / "e2v2_rescore.csv")
    assert len(rows) == 30
    scores = {(row["model"], row["regime"], int(row["seed"])): row for row in rows}
    for kind in MODELS:
        for regime in ("real", "pre0", "ft", "pcal", "ftcal"):
            current = [scores[kind, regime, seed] for seed in (0, 1, 2)]
            values = [float(row["pl"]) for row in current]
            stage = "post_hoc" if regime in ("pcal", "ftcal") else "preregistered"
            for row in current:
                assert row["protocol_stage"] == stage
                assert float(row["pl_seed_min"]) == min(values) and float(row["pl_seed_max"]) == max(values)
    ftcal, real = (scores["transformer", regime, 0] for regime in ("ftcal", "real"))
    ratio, p, real_ratio = float(ftcal["ratio"]), float(ftcal["mcnemar_p"]), float(real["ratio"])
    assert abs(ratio - 0.992) < 0.001 and abs(p - 0.6) < 0.01 and abs(real_ratio - 1.232) < 0.001
    settings = json.loads((RESULTS / "e2v2_rescore.settings.json").read_text(encoding="utf-8"))
    assert settings["google_reference"] == reference
    print(f"PASS 4: TF ftcal s0 ratio={ratio:.6f}, p={p:.6f}; TF real s0 ratio={real_ratio:.6f}")


def ordered_hash(packed, obs, indices):
    digest = hashlib.sha256()
    for start in range(0, len(indices), 75_000):
        bits = np.unpackbits(packed[indices[start:start + 75_000]], axis=1, count=N_DETECTORS, bitorder="big")
        digest.update(bits)
    digest.update(obs[indices])
    return digest.hexdigest()


def check_settings(packed, obs, det14, obs14, train_idx, smoke, orders):
    order_a, order_b = orders
    suffix = "_smoke" if smoke else ""
    a_path = RESULTS / f"e2v2_datasize{suffix}.settings.json"
    b_path = RESULTS / f"e2v2_transfer{suffix}.settings.json"
    if smoke:
        assert a_path.is_file() and b_path.is_file(), "Both smoke preregistrations are required"
    if a_path.exists():
        settings = json.loads(a_path.read_text(encoding="utf-8"))
        n_external = len(order_b)
        a_det = np.concatenate((np.packbits(det14[train_idx], axis=1, bitorder="big"), packed[:n_external]))
        a_obs = np.concatenate((obs14[train_idx], obs[:n_external]))
        sizes = SMOKE_SIZES if smoke else SIZES["mlp"]
        assert settings["pool_seed"] == A_SEED and settings["n_pool"] == len(order_a)
        assert set(settings["prefix_sha256"]) == set(map(str, sizes))
        assert set(settings["n_from_sample14"]) == set(map(str, sizes))
        for n in sizes:
            prefix = order_a[:n]
            assert settings["prefix_sha256"][str(n)] == ordered_hash(a_det, a_obs, prefix), n
            assert settings["n_from_sample14"][str(n)] == int((prefix < len(train_idx)).sum()), n
    if b_path.exists():
        settings = json.loads(b_path.read_text(encoding="utf-8"))
        assert settings["pool_seed"] == B_SEED and settings["external_pool_size"] == len(order_b)
        assert settings["pool_sha256"] == ordered_hash(packed, obs, order_b)


def main():
    with np.load(POOL / "pool.npz", allow_pickle=False) as file:
        assert set(file.files) == {"detectors", "obs", "sample", "shot"}
        packed, obs, samples, shots = (file[name] for name in ("detectors", "obs", "sample", "shot"))
    _, dems, det14, obs14, google = load_packet()
    split, masks, _, reference = load_references(obs14, google)
    train_idx, val_idx, test_idx = split

    check_sample_stats(packed, obs, samples, det14, obs14)
    check_pool_order(packed, obs, samples, shots)
    full_a, full_b, full_no_leakage, full_union = provenance(samples, shots, split, POOL_SIZE)
    smoke_a, smoke_b, smoke_no_leakage, smoke_union = provenance(samples, shots, split, SMOKE_POOL_SIZE)
    assert (len(train_idx), len(val_idx), len(test_idx)) == (N_TRAIN, 7_500, 22_500)
    assert full_no_leakage and smoke_no_leakage
    print(f"PASS 3: A full ({len(full_a):,}) and smoke ({len(smoke_a):,}) exclude sample_14 val/test shots")
    check_rescore(reference)

    fresh_wrong = predict_pm(dems["simple"], det14[test_idx], correlated=True) != obs14[test_idx]
    np.testing.assert_array_equal(fresh_wrong, masks[PM], err_msg=PM)
    print(f"PASS 5: fresh {PM} matches all {len(fresh_wrong):,} cached test bits")

    assert full_union and smoke_union
    check_settings(packed, obs, det14, obs14, train_idx, True, (smoke_a, smoke_b))
    check_settings(packed, obs, det14, obs14, train_idx, False, (full_a, full_b))
    print("PASS 6: smoke A prefixes 5,000/10,000 and B pool hash agree; A = B ∪ sample_14 train (full and smoke)")
    print("PASS: all six E2 v2 checks")


if __name__ == "__main__":
    main()
