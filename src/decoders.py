"""PyMatching baselines for E1 (PLAN v1.2: three tiers).

PM-naive:      matching graph from the independent-noise circuit (f = 0) at the
               same p, i.e. the decoder assumes the wrong noise model.
PM-informed:   matching graph from the true data-generating circuit; hyperedges
               are split into graph edges.
PM-correlated: same DEM as informed, with PyMatching's correlated matching.

Why we decompose the DEM ourselves: Stim's heuristic decomposition can only
split a hyperedge into edges that already exist on their own. The ZZ part of
an adjacent-pair YY error links two X-type detectors that no single error
links, so Stim invents boundary edges whose observable flag contradicts the
existing ones (e.g. D0 alone both with and without L0), which corrupts the
matching graph. The surface code is CSS: X-type detectors only see Z errors
and Z-type detectors (plus the Z-basis logical L0) only see X errors. So each
error is split into its X-type part and its Z-type+L0 part. For d = 3 with
this noise every part has at most two detectors (checked in
basis_decomposed_dem).
"""

import numpy as np
import pymatching
import stim

from noise import correlated_memory_circuit


def x_type_detectors(circuit):
    """Detectors measured by X-type ancillas (the ones that get H gates)."""
    qc = circuit.get_final_qubit_coordinates()
    x_anc = {
        tuple(qc[t.value])
        for inst in circuit.flattened()
        if inst.name == "H"
        for t in inst.targets_copy()
    }
    return {k for k, v in circuit.get_detector_coordinates().items() if tuple(v[:2]) in x_anc}


def basis_decomposed_dem(circuit):
    """Undecomposed DEM, then split each error into X-type / Z-type(+L) parts."""
    xdet = x_type_detectors(circuit)
    raw = circuit.detector_error_model(decompose_errors=False)
    out = stim.DetectorErrorModel()
    for inst in raw.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs = [t.val for t in inst.targets_copy() if t.is_logical_observable_id()]
        x_part = [d for d in dets if d in xdet]
        z_part = [d for d in dets if d not in xdet]
        if len(x_part) > 2 or len(z_part) > 2:
            raise ValueError(f"basis part with >2 detectors: {inst}")
        if obs and not z_part:
            raise ValueError(f"logical flip with no Z-type detector: {inst}")
        comps = []
        if z_part:
            comps.append(
                [stim.target_relative_detector_id(d) for d in z_part]
                + [stim.target_logical_observable_id(o) for o in obs]
            )
        if x_part:
            comps.append([stim.target_relative_detector_id(d) for d in x_part])
        targets = []
        for i, comp in enumerate(comps):
            if i:
                targets.append(stim.target_separator())
            targets.extend(comp)
        if targets:
            out.append("error", inst.args_copy(), targets)
    return resolve_edge_observables(out)


def _components(inst):
    comps = [[]]
    for t in inst.targets_copy():
        if t.is_separator():
            comps.append([])
        else:
            comps[-1].append(t)
    return [
        (
            tuple(sorted(t.val for t in c if t.is_relative_detector_id())),
            tuple(sorted(t.val for t in c if t.is_logical_observable_id())),
        )
        for c in comps
    ]


def resolve_edge_observables(dem):
    """Give each graph edge a single observable flag: the most probable one.

    Correlated XX on adjacent boundary pairs creates edges (same detectors)
    that sometimes flip L0 and sometimes not. A matching graph edge can carry
    only one observable mask, and PyMatching does not choose by probability
    when merging such parallel edges. The maximum-likelihood choice for a
    matching decoder is the mask with the larger total probability; components
    of the minority mask are relabelled. This is an approximation the graph
    cannot avoid; the relabelled probability is reported by edge_conflicts().
    """
    mass = {}
    for inst in dem.flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            for d, o in _components(inst):
                mass.setdefault(d, {}).setdefault(o, 0.0)
                mass[d][o] += p
    best = {d: max(v, key=v.get) for d, v in mass.items()}
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        targets = []
        for i, (d, _) in enumerate(_components(inst)):
            if i:
                targets.append(stim.target_separator())
            targets += [stim.target_relative_detector_id(x) for x in d]
            targets += [stim.target_logical_observable_id(x) for x in best[d]]
        out.append("error", inst.args_copy(), targets)
    return out


def edge_conflicts(circuit):
    """Total probability of mechanisms whose edge observable had to be relabelled."""
    xdet = x_type_detectors(circuit)
    raw = circuit.detector_error_model(decompose_errors=False)
    mass = {}
    for inst in raw.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
        z_part = tuple(sorted(d for d in dets if d not in xdet))
        if z_part:
            mass.setdefault(z_part, {}).setdefault(obs, 0.0)
            mass[z_part][obs] += p
    return sum(sum(v.values()) - max(v.values()) for v in mass.values())


def build_pm_decoders(p, f, distance=3, rounds=3):
    true_circuit, _ = correlated_memory_circuit(p, f, distance, rounds)
    naive_circuit, _ = correlated_memory_circuit(p, 0.0, distance, rounds)
    dem_true = basis_decomposed_dem(true_circuit)
    dem_naive = basis_decomposed_dem(naive_circuit)
    return {
        "PM-naive": (pymatching.Matching.from_detector_error_model(dem_naive), False),
        "PM-informed": (pymatching.Matching.from_detector_error_model(dem_true), False),
        "PM-correlated": (
            pymatching.Matching.from_detector_error_model(dem_true, enable_correlations=True),
            True,
        ),
    }


def decode_all(decoders, detectors):
    """Return {name: predicted observable flips, shape (shots,)}."""
    out = {}
    for name, (matcher, corr) in decoders.items():
        pred = matcher.decode_batch(detectors, enable_correlations=corr)
        out[name] = np.asarray(pred[:, 0], dtype=bool)
    return out
