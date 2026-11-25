"""Check the E3 v2 split, majority, Stim matching, and paired counts offline."""

import numpy as np

from analyze_e3 import FIT_FRACTION, SPLIT_SEED, analyze, fit_edge_probs, matching
from e1v2_common import paired_stats
from run_e3_v2 import (
    BLOCKS, EDGES, N_DET, PUB_SHOTS, ROUNDS, SIM_SEEDS, TAUS,
    graph_pm_wrong, load_raw, majority_wrong, make_dem, real_splits,
    sim_size, train_pool,
)


def check_splits(job, metadata, pubs):
    order = [(row["block"], row["tau_us"], row["logical"]) for row in job["pub_order"]]
    np.testing.assert_array_equal(metadata, order)
    print("PASS: 30 PUBs match job pub_order")
    all_splits = {}
    for tau in TAUS:
        splits = all_splits[tau] = real_splits(metadata, pubs, tau)
        seen = set()
        for name, blocks in BLOCKS.items():
            split = splits[name]
            ids = set(split.pub_ids.tolist())
            expected = {i for i, (block, t, _) in enumerate(order) if t == tau and block in blocks}
            assert ids == expected and not (ids & seen), (tau, name, ids)
            seen |= ids
            assert split.det.shape == (2 * len(blocks) * PUB_SHOTS, N_DET)
            np.testing.assert_array_equal(np.bincount(split.logical), [len(blocks) * PUB_SHOTS] * 2)
        assert len(seen) == 10
        print(f"PASS tau={tau}: disjoint train/val/test 60000/20000/20000, balanced L0/L1")
    return all_splits


def check_majority(metadata, pubs, splits):
    # Give legacy analyze a known fit/test partition. It selects the shuffled
    # second half, so place all block 4 shots there and block 0 in the fit half.
    old_input = {}
    rng = np.random.default_rng(SPLIT_SEED)
    for logical in (0, 1):
        name = f"rep3_r{ROUNDS}_L{logical}"
        perm = rng.permutation(2 * PUB_SHOTS)
        n_fit = int(FIT_FRACTION * len(perm))
        assert n_fit == PUB_SHOTS
        syn = np.empty((len(perm), ROUNDS, 2), dtype=np.uint8)
        data = np.empty((len(perm), 3), dtype=np.uint8)
        for block, idx in ((0, perm[:n_fit]), (4, perm[n_fit:])):
            pub, = np.flatnonzero((metadata[:, 0] == block) & (metadata[:, 1] == 0)
                                 & (metadata[:, 2] == logical))
            det, final_data = pubs[pub]
            data[idx] = final_data
            # Invert the syndrome differences for the legacy raw-input API.
            syn[idx] = np.bitwise_xor.accumulate(det.reshape(-1, ROUNDS + 1, 2)[:, :ROUNDS], axis=1)
        old_input[f"syn_{name}"], old_input[f"data_{name}"] = syn, data

    old_rows, _ = analyze(old_input, rounds=(ROUNDS,))
    test = splits[0]["test"]
    d0, d1, d2 = test.data.T
    direct = ((d0 & d1) | (d0 & d2) | (d1 & d2)) != test.logical
    np.testing.assert_array_equal(majority_wrong(test), direct)
    for row in old_rows:
        logical = row["logical"]
        rate = direct[test.logical == logical].mean()
        assert row["n_test"] == PUB_SHOTS and row["majority"] == rate
        print(f"PASS tau=0 L={logical}: test majority legacy=direct={rate:.4f}")

    for tau in TAUS:
        for logical in (0, 1):
            ids = np.flatnonzero((metadata[:, 1] == tau) & (metadata[:, 2] == logical))
            data = np.concatenate([pubs[i][1] for i in ids])
            assert len(data) == 5 * PUB_SHOTS
            retrieve_rate = ((data.sum(1) >= 2).astype(int) != logical).mean()
            d0, d1, d2 = data.T
            direct_rate = (((d0 & d1) | (d0 & d2) | (d1 & d2)) != logical).mean()
            assert retrieve_rate == direct_rate
            if (tau, logical) == (0, 1):
                assert f"{retrieve_rate:.4f}" == "0.1095", retrieve_rate
            print(f"PASS tau={tau:>2} L={logical}: all-block majority={retrieve_rate:.4f}")


def check_graph_and_pairing(splits):
    for tau in TAUS:
        data = splits[tau]
        probs = fit_edge_probs(train_pool(data["train"], tau, smoke=False)[0], EDGES)
        dem = make_dem(probs)
        n = sim_size("test", smoke=False)
        det, obs, faults = dem.compile_sampler(seed=SIM_SEEDS["test"] + tau).sample(
            n, return_errors=True,
        )
        logical_edges = [k for k, (i, j) in enumerate(EDGES) if j is None and i % 2 == 0]
        truth = np.logical_xor.reduce(faults[:, logical_edges], axis=1)
        np.testing.assert_array_equal(obs[:, 0], truth)
        wrong = graph_pm_wrong(dem, probs, (det, obs[:, 0]))
        direct = matching(EDGES, probs, N_DET).decode_batch(det)[:, 0] != truth
        np.testing.assert_array_equal(wrong, direct)
        print(f"PASS tau={tau}: DEM/direct PM identical on {n:,} shots, "
              f"PM P_L={np.count_nonzero(direct)}/{n}={wrong.mean():.5f}")

        test = data["test"]
        det, y = test.xy
        pm = matching(EDGES, probs, N_DET).decode_batch(det)[:, 0] != y
        maj = majority_wrong(test)
        for subset, mask in (("all", slice(None)), ("L0", test.logical == 0), ("L1", test.logical == 1)):
            a, b = maj[mask], pm[mask]
            stats = paired_stats(a, b)
            counts = np.bincount(2 * a.astype(np.uint8) + b, minlength=4)
            assert (stats["b"], stats["c"]) == (counts[2], counts[1])
            print(f"PASS tau={tau} {subset}: paired b={stats['b']}, c={stats['c']}")


def main():
    job, metadata, pubs = load_raw()
    splits = check_splits(job, metadata, pubs)
    check_majority(metadata, pubs, splits)
    check_graph_and_pairing(splits)
    print("PASS: all four E3 v2 checks")


if __name__ == "__main__":
    main()
