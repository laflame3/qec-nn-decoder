"""E3: distance-3 bit-flip repetition code for IBM hardware (PLAN §3).

3 data qubits + 2 ancillas. Each round the ancillas measure Z0Z1 and Z1Z2
(mid-circuit measurement, then reset); at the end all data qubits are
measured. Prepared in |000> or |111>. Nothing here talks to IBM; submission
is done separately and only after explicit authorisation.

Classical registers: syn{k} (2 bits) for round k = 0..r-1, and `data` (3 bits).
"""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def repetition_circuit(rounds, logical):
    """Repetition-code memory circuit; logical in {0, 1}."""
    d = QuantumRegister(3, "d")
    a = QuantumRegister(2, "a")
    syn = [ClassicalRegister(2, f"syn{k}") for k in range(rounds)]
    data = ClassicalRegister(3, "data")
    qc = QuantumCircuit(d, a, *syn, data, name=f"rep3_r{rounds}_L{logical}")
    if logical:
        qc.x(d)
    for k in range(rounds):
        qc.barrier()
        qc.cx(d[0], a[0])
        qc.cx(d[1], a[0])
        qc.cx(d[1], a[1])
        qc.cx(d[2], a[1])
        qc.measure(a, syn[k])
        if k < rounds - 1:
            qc.reset(a)
    qc.barrier()
    qc.measure(d, data)
    return qc


def split_shots(pub_data, rounds):
    """Bitstrings per register -> (syndromes: list of r lists, data: list).

    Qiskit bitstrings are little-endian: the rightmost character is bit 0.
    Returns per-shot integer arrays: syn[shot, round, ancilla], data[shot, qubit].
    """
    import numpy as np

    syn = np.stack(
        [
            np.array([[int(b) for b in s[::-1]] for s in getattr(pub_data, f"syn{k}").get_bitstrings()])
            for k in range(rounds)
        ],
        axis=1,
    )
    data = np.array([[int(b) for b in s[::-1]] for s in pub_data.data.get_bitstrings()])
    return syn, data
