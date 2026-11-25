"""Submit the E3 v2 repetition-code job (idle-delay knob) to ibm_kingston, once.

Authorised by the student on 2026-11-24 (chat: "授权，写脚本并提交") for exactly
this workload: d=3 repetition code, r=9, logical |000>/|111>, idle delay
tau in {0, 5, 15} us on the data qubits at the start of every round, fixed
chain d0,d1,d2,a0,a1 = 147,149,151,148,150, 6 conditions x 50,000 shots split
into 5 interleaved blocks of 10,000 (30 PUBs), dynamical decoupling off.

    ../../.venv/bin/python -I src/submit_e3_ibm_v2.py --dry-run   # no submission
    ../../.venv/bin/python -I src/submit_e3_ibm_v2.py             # submits once

Writes data/results/e3_ibm_v2_job.json and refuses to run if it exists.
Credentials are loaded by account name only; nothing secret is printed or saved.
"""

import datetime
import json
import pathlib
import sys

import qiskit
import qiskit_ibm_runtime
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

BACKEND = "ibm_kingston"
LAYOUT = [147, 149, 151, 148, 150]  # d0, d1, d2, a0, a1
ROUNDS = 9
TAUS_US = [0, 5, 15]
LOGICALS = [0, 1]
N_BLOCKS, BLOCK_SHOTS = 5, 10_000
MAX_QPU_SECONDS = 200  # refuse if the estimate is larger than this

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "results" / "e3_ibm_v2_job.json"


def rep_tau(rounds, logical, tau_dt):
    """Same registers as ibm_repetition.repetition_circuit, plus an idle delay per round."""
    d = QuantumRegister(3, "d")
    a = QuantumRegister(2, "a")
    syn = [ClassicalRegister(2, f"syn{k}") for k in range(rounds)]
    data = ClassicalRegister(3, "data")
    qc = QuantumCircuit(d, a, *syn, data)
    if logical:
        qc.x(d)
    for k in range(rounds):
        qc.barrier()
        if tau_dt:
            qc.delay(tau_dt, d, unit="dt")
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


def main(dry_run):
    if OUT.exists():
        sys.exit(f"{OUT.name} already exists; job was already submitted. Retrieve, do not resubmit.")
    service = QiskitRuntimeService(name="kcl-open-plan")
    backend = service.backend(BACKEND)
    status = backend.status()
    print("backend:", backend.name, "| operational:", status.operational, "| pending:", status.pending_jobs)
    assert status.operational
    usage = service.usage()
    print("usage remaining s:", usage.get("usage_remaining_seconds"))

    dt = backend.target.dt
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=LAYOUT)
    isa, info = {}, []
    for tau in TAUS_US:
        tau_dt = int(round(tau * 1e-6 / dt / 16)) * 16  # multiple of 16 dt
        for L in LOGICALS:
            qc = rep_tau(ROUNDS, L, tau_dt)
            qc.name = f"rep3_r{ROUNDS}_L{L}_tau{tau}"
            c = pm.run(qc)
            ops = {k: int(v) for k, v in c.count_ops().items()}
            assert c.layout.final_index_layout()[:5] == LAYOUT, c.layout.final_index_layout()[:5]
            assert ops.get("swap", 0) == 0, ops
            dur_us = c.estimate_duration(backend.target, unit="u")
            isa[(tau, L)] = c
            info.append({"name": qc.name, "tau_us": tau, "tau_dt": tau_dt, "logical": L,
                         "layout": c.layout.final_index_layout()[:5], "depth": c.depth(),
                         "ops": ops, "estimated_duration_us": round(dur_us, 2)})
            print(f"  {qc.name}: tau_dt={tau_dt} depth={c.depth()} dur~{dur_us:.0f}us ops={ops}")

    order = [(b, tau, L) for b in range(N_BLOCKS) for tau in TAUS_US for L in LOGICALS]
    pubs = [(isa[(tau, L)],) for _, tau, L in order]
    n_shots = len(pubs) * BLOCK_SHOTS
    # Job 1: 160,000 shots, 45 s execution, circuits ~45-110 us -> ~0.28 ms/shot incl. reset/rep delay.
    est_s = sum(BLOCK_SHOTS * (0.25e-3 + i["estimated_duration_us"] * 1e-6)
                for _, tau, L in order for i in info if (i["tau_us"], i["logical"]) == (tau, L))
    print(f"PUBs={len(pubs)} shots={n_shots:,} estimated QPU ~{est_s:.0f} s")
    assert est_s <= MAX_QPU_SECONDS, est_s
    assert est_s < usage.get("usage_remaining_seconds", 0), "not enough quota"

    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    assert sampler.options.dynamical_decoupling.enable is False
    if dry_run:
        print("dry run: nothing submitted")
        return

    job = sampler.run(pubs, shots=BLOCK_SHOTS)
    props = backend.properties()
    chain = {}
    for q in LAYOUT:
        chain[str(q)] = {k: props.qubit_property(q, k)[0] for k in ("T1", "T2", "readout_error")}
    record = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pending_jobs_at_submit": status.pending_jobs,
        "rounds": ROUNDS, "taus_us": TAUS_US, "logicals": LOGICALS,
        "n_blocks": N_BLOCKS, "block_shots": BLOCK_SHOTS, "total_shots": n_shots,
        "pub_order": [{"pub": i, "block": b, "tau_us": tau, "logical": L} for i, (b, tau, L) in enumerate(order)],
        "circuits": info,
        "qubit_order": "layout lists physical qubits for d0, d1, d2, a0, a1",
        "dt_s": dt,
        "transpile": "generate_preset_pass_manager(optimization_level=1, initial_layout=LAYOUT)",
        "options": {"dynamical_decoupling": False, "twirling_gates": False, "twirling_measure": False},
        "estimated_qpu_s": round(est_s, 1),
        "calibration_last_update": str(props.last_update_date),
        "chain_calibration": chain,
        "qiskit": qiskit.__version__, "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
        "authorisation": "student, chat, 2026-11-24: '授权，写脚本并提交'",
    }
    OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print("submitted job:", record["job_id"])


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
