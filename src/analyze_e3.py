"""E3 analysis: decode the IBM repetition-code shots three ways.

Detectors (per shot), for r rounds, ancilla a0 = Z0Z1, a1 = Z1Z2:
  round 0:      s[0, j]                 (ideal parity is 0 for |000> and |111>)
  round k >= 1: s[k, j] xor s[k-1, j]
  final:        (data parity j) xor s[r-1, j]
Logical observable: data qubit d0 (any single data qubit works for this code).

Decoders
  majority:   majority vote of the final data bits, ignores all syndromes.
  PM-uniform: matching on the repetition-code graph with every edge equally
              likely (no noise information).
  PM-fitted:  same graph, edge probabilities estimated from detector
              correlations on a separate fitting half of the shots (both
              initial states pooled, so the decoder does not know the state).

Graph edges (from the circuit, CX order d0-a0, d1-a0, d1-a1, d2-a1):
  space   (0,k)-(1,k)      error on d1
  time    (j,k)-(j,k+1)    ancilla measurement error
  diag    (1,k)-(0,k+1)    d1 error between its two CX gates
  bound.  (0,k)            error on d0  -> flips the observable
  bound.  (1,k)            error on d2
"""

import json
import pathlib

import numpy as np
import pymatching

ROUNDS = [1, 3, 5, 9]
FIT_FRACTION = 0.5
SPLIT_SEED = 7
P_UNIFORM = 0.01


def detectors(syn, data):
    """syn: shots x r x 2, data: shots x 3  ->  shots x (r+1) x 2 detector events."""
    r = syn.shape[1]
    det = np.empty((syn.shape[0], r + 1, 2), dtype=np.uint8)
    det[:, 0] = syn[:, 0]
    if r > 1:
        det[:, 1:r] = syn[:, 1:] ^ syn[:, :-1]
    final_par = np.stack([data[:, 0] ^ data[:, 1], data[:, 1] ^ data[:, 2]], axis=1)
    det[:, r] = final_par ^ syn[:, r - 1]
    return det


def graph_edges(r):
    """List of (i, j) detector pairs (j = None for boundary) in the matching graph."""
    idx = lambda k, j: 2 * k + j  # noqa: E731
    n_layers = r + 1
    edges = []
    for k in range(n_layers):
        edges.append((idx(k, 0), idx(k, 1)))
        edges.append((idx(k, 0), None))
        edges.append((idx(k, 1), None))
        if k + 1 < n_layers:
            edges.append((idx(k, 0), idx(k + 1, 0)))
            edges.append((idx(k, 1), idx(k + 1, 1)))
            edges.append((idx(k, 1), idx(k + 1, 0)))
    return edges


def fit_edge_probs(det_flat, edges):
    """Pairwise-correlation estimate of edge probabilities.

    p_ij = 1/2 - 1/2 sqrt(1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>))
    Boundary: 1 - 2<x_i> = (1 - 2 p_i^b) * prod_j (1 - 2 p_ij) over the bulk edges at i.
    Values that come out negative or undefined are clipped to a small floor.
    """
    x = det_flat.astype(float)
    mean = x.mean(0)
    floor = 1e-5
    probs = {}
    for i, j in edges:
        if j is None:
            continue
        xij = (x[:, i] * x[:, j]).mean()
        num = 4 * (xij - mean[i] * mean[j])
        den = 1 - 2 * mean[i] - 2 * mean[j] + 4 * xij
        val = 1 - num / den if den > 0 else np.nan
        p = 0.5 - 0.5 * np.sqrt(val) if val is not None and val >= 0 else np.nan
        probs[(i, j)] = float(np.clip(np.nan_to_num(p, nan=floor), floor, 0.5 - 1e-6))
    for i, j in edges:
        if j is not None:
            continue
        prod = 1.0
        for (a, b), p in probs.items():
            if i in (a, b):
                prod *= 1 - 2 * p
        pb = 0.5 * (1 - (1 - 2 * mean[i]) / prod)
        probs[(i, None)] = float(np.clip(pb, floor, 0.5 - 1e-6))
    return probs


def matching(edges, probs, n_det):
    m = pymatching.Matching()
    for i, j in edges:
        p = probs[(i, j)]
        w = np.log((1 - p) / p)
        if j is None:
            obs = {0} if i % 2 == 0 else set()
            m.add_boundary_edge(i, weight=w, fault_ids=obs, error_probability=p)
        else:
            m.add_edge(i, j, weight=w, fault_ids=set(), error_probability=p)
    m.set_boundary_nodes(set())
    assert m.num_detectors == n_det
    return m


def analyze(raw, rounds=ROUNDS, split_seed=SPLIT_SEED):
    rows, fitted = [], {}
    rng = np.random.default_rng(split_seed)
    for r in rounds:
        per_state = {}
        for L in (0, 1):
            name = f"rep3_r{r}_L{L}"
            det = detectors(raw[f"syn_{name}"], raw[f"data_{name}"]).reshape(-1, 2 * (r + 1))
            data = raw[f"data_{name}"]
            perm = rng.permutation(len(det))
            n_fit = int(FIT_FRACTION * len(det))
            per_state[L] = (det, data, perm[:n_fit], perm[n_fit:])
        edges = graph_edges(r)
        n_det = 2 * (r + 1)
        fit_det = np.concatenate([per_state[L][0][per_state[L][2]] for L in (0, 1)])
        probs_fit = fit_edge_probs(fit_det, edges)
        fitted[r] = {f"{i}-{'B' if j is None else j}": p for (i, j), p in probs_fit.items()}
        decoders = {
            "PM-uniform": matching(edges, {e: P_UNIFORM for e in edges}, n_det),
            "PM-fitted": matching(edges, probs_fit, n_det),
        }
        for L in (0, 1):
            det, data, _, test = per_state[L]
            det_t, data_t = det[test], data[test]
            n = len(test)
            row = {"rounds": r, "logical": L, "n_test": n,
                   "detector_rate": float(det_t.mean()),
                   "final_layer_rate": float(det_t[:, -2:].mean())}
            wrong = {"majority": (data_t.sum(1) >= 2).astype(int) != L}
            for name, m in decoders.items():
                pred = m.decode_batch(det_t)[:, 0]
                wrong[name] = (data_t[:, 0] ^ pred) != L
            for name, w in wrong.items():
                ler = float(w.mean())
                row[name] = ler
                row[name + "_err"] = float(np.sqrt(max(ler * (1 - ler), 1 / n) / n))
            rows.append(row)
    return rows, fitted


if __name__ == "__main__":
    res = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"
    raw = np.load(res / "e3_ibm_raw.npz")
    rows, fitted = analyze(raw)
    cols = list(rows[0].keys())
    with open(res / "e3_ibm_analysis.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            fh.write(",".join(str(row[c]) for c in cols) + "\n")
    (res / "e3_ibm_fitted_edges.json").write_text(json.dumps(fitted, indent=1))
    (res / "e3_ibm_analysis.settings.json").write_text(json.dumps({
        "script": "src/analyze_e3.py", "input": "data/results/e3_ibm_raw.npz",
        "job": json.loads((res / "e3_ibm_job.json").read_text())["job_id"],
        "fit_fraction": FIT_FRACTION, "split_seed": SPLIT_SEED, "p_uniform": P_UNIFORM,
        "fit": "pairwise detector correlations, both initial states pooled",
        "observable": "data qubit d0", "pymatching": pymatching.__version__,
    }, indent=2))
    for row in rows:
        print(f"r={row['rounds']} L={row['logical']} det_rate={row['detector_rate']:.4f} "
              f"maj={row['majority']:.4f} PM-unif={row['PM-uniform']:.4f} "
              f"PM-fit={row['PM-fitted']:.4f} (±{row['PM-fitted_err']:.4f})")
