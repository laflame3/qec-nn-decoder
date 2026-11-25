"""Noisy d=3 rotated-surface-code memory circuits for E1 (PLAN v1.2 §1.2).

Gate, reset and measurement noise is Stim's standard circuit-level model at
strength p. Only the data-qubit idle noise at the start of each round is
changed: a fixed budget of n_data * p expected Pauli faults per round is split
between independent DEPOLARIZE1 (fraction 1 - f) and correlated XX / YY
errors on adjacent data-qubit pairs (fraction f).
"""

import itertools

import stim

# Placeholder strength for the before-round data noise. Stim emits it as a
# DEPOLARIZE1 instruction that we then replace; the value only has to be
# unique among the noise strengths used in the circuit.
_MARKER = 0.0314159


def _data_qubits_and_pairs(circuit):
    """Data qubits are the ones hit by the before-round marker noise.

    Adjacent pairs are data qubits whose (x, y) coordinates differ by 2 along
    one axis (nearest neighbours on the rotated lattice).
    """
    data = None
    for inst in circuit.flattened():
        if inst.name == "DEPOLARIZE1" and inst.gate_args_copy() == [_MARKER]:
            data = [t.value for t in inst.targets_copy()]
            break
    if data is None:
        raise ValueError("marker noise not found in circuit")
    coords = circuit.get_final_qubit_coordinates()
    pairs = []
    for a, b in itertools.combinations(data, 2):
        dx = abs(coords[a][0] - coords[b][0])
        dy = abs(coords[a][1] - coords[b][1])
        if sorted((dx, dy)) == [0, 2]:
            pairs.append((a, b))
    return data, pairs


def _replace_marker(circuit, data, pairs, p_ind, q_pair):
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = _replace_marker(inst.body_copy(), data, pairs, p_ind, q_pair)
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
        elif inst.name == "DEPOLARIZE1" and inst.gate_args_copy() == [_MARKER]:
            if p_ind > 0:
                out.append("DEPOLARIZE1", data, p_ind)
            if q_pair > 0:
                for a, b in pairs:
                    for pauli in ("X", "Y"):
                        out.append(
                            "CORRELATED_ERROR",
                            [stim.target_pauli(a, pauli), stim.target_pauli(b, pauli)],
                            q_pair / 2,
                        )
        else:
            out.append(inst)
    return out


def correlated_memory_circuit(p, f, distance=3, rounds=3):
    """Z-basis memory circuit with correlation fraction f in [0, 1].

    Budget per round: n_data * p expected data-qubit faults.
      independent part: DEPOLARIZE1(p * (1 - f)) on every data qubit
      correlated part:  per adjacent pair, XX and YY each with prob q/2,
                        q = n_data * p * f / (2 * n_pairs)  (each event = 2 faults)
    f = 0 reproduces Stim's standard circuit-level depolarising model.
    """
    if not 0.0 <= f <= 1.0:
        raise ValueError("f must be in [0, 1]")
    template = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=_MARKER,
    )
    data, pairs = _data_qubits_and_pairs(template)
    p_ind = p * (1.0 - f)
    q_pair = len(data) * p * f / (2 * len(pairs))
    circuit = _replace_marker(template, data, pairs, p_ind, q_pair)
    info = {
        "p": p,
        "f": f,
        "distance": distance,
        "rounds": rounds,
        "n_data": len(data),
        "n_pairs": len(pairs),
        "p_independent": p_ind,
        "q_pair": q_pair,
    }
    return circuit, info


def hyperedge_fraction(dem):
    """Share h of total error probability carried by non-graphlike mechanisms.

    A mechanism is non-graphlike if it flips more than two detectors (before
    decomposition). Needs an undecomposed DEM.
    """
    total = 0.0
    hyper = 0.0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        n_det = sum(1 for t in inst.targets_copy() if t.is_relative_detector_id())
        total += prob
        if n_det > 2:
            hyper += prob
    return hyper / total if total else 0.0


def structure_memory_circuit(kind, idle_prob, p_bg=0.005, distance=3, rounds=3):
    """Replace the before-round idle noise with one Pauli structure."""
    if kind not in ("DEP", "X", "Y", "XX", "YY"):
        raise ValueError(f"unknown idle noise structure: {kind}")
    if not 0 <= idle_prob <= 1:
        raise ValueError("idle_prob must be in [0, 1]")

    template = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_bg,
        after_reset_flip_probability=p_bg,
        before_measure_flip_probability=p_bg,
        before_round_data_depolarization=_MARKER,
    )
    data, pairs = _data_qubits_and_pairs(template)

    def replace_marker(block):
        out = stim.Circuit()
        for inst in block:
            if isinstance(inst, stim.CircuitRepeatBlock):
                out.append(stim.CircuitRepeatBlock(inst.repeat_count, replace_marker(inst.body_copy())))
            elif inst.name != "DEPOLARIZE1" or inst.gate_args_copy() != [_MARKER]:
                out.append(inst)
            elif idle_prob:
                if kind == "DEP":
                    out.append("DEPOLARIZE1", data, idle_prob)
                elif len(kind) == 1:
                    out.append(f"{kind}_ERROR", data, idle_prob)
                else:
                    for a, b in pairs:
                        targets = [stim.target_pauli(a, kind[0]), stim.target_pauli(b, kind[0])]
                        out.append("CORRELATED_ERROR", targets, idle_prob)
        return out

    return replace_marker(template), {
        "kind": kind, "idle_prob": idle_prob, "p_bg": p_bg,
        "n_data": len(data), "n_pairs": len(pairs),
    }


def mean_detection_density(circuit, shots, seed):
    """Mean number of lit detectors per shot."""
    det = circuit.compile_detector_sampler(seed=seed).sample(shots)
    return float(det.sum() / shots)


def calibrate_idle_prob(kind, target, shots, seed, p_bg=0.005):
    """Bisect the geometric midpoint with fresh samplers sharing one seed."""
    low, high = 1e-4, 5e-2
    trials = []
    for _ in range(40):
        prob = (low * high) ** 0.5
        circuit, _ = structure_memory_circuit(kind, prob, p_bg)
        density = mean_detection_density(circuit, shots, seed)
        error = abs(density - target) / target
        trials.append({"idle_prob": prob, "density": density, "relative_error": error})
        if error < 0.002:
            break
        if density < target:
            low = prob
        else:
            high = prob
    best = min(trials, key=lambda trial: trial["relative_error"])
    if best["relative_error"] >= 0.01:
        raise ValueError(f"{kind} calibration missed target {target:.6f}: best trial {best}")
    return best["idle_prob"], {**best, "target": target, "shots": shots, "seed": seed, "trials": trials}
