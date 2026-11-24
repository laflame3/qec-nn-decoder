"""Submit the E3 repetition-code job to IBM hardware (one job, run once).

Authorised by the student on 2026-11-22 for exactly this workload:
8 circuits (rounds 1, 3, 5, 9 x logical 0, 1), 20,000 shots each, on the
least busy dynamic-circuit backend of the saved account `kcl-open-plan`.

Writes data/results/e3_ibm_job.json (job ID, backend, layouts, versions).
Refuses to run if that file already exists, so the job is never resubmitted.
Credentials are loaded by account name only; nothing secret is printed or saved.
"""

import datetime
import json
import pathlib
import sys

import qiskit
import qiskit_ibm_runtime
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ibm_repetition import repetition_circuit  # noqa: E402

ROUNDS = [1, 3, 5, 9]
LOGICALS = [0, 1]
SHOTS = 20_000

out = pathlib.Path(__file__).resolve().parent.parent / "data" / "results" / "e3_ibm_job.json"
if out.exists():
    sys.exit(f"{out.name} already exists; job was already submitted. Use retrieve, not resubmit.")

service = QiskitRuntimeService(name="kcl-open-plan")
backend = service.least_busy(operational=True, simulator=False, dynamic_circuits=True)
status = backend.status()
print("backend:", backend.name, "| pending jobs:", status.pending_jobs, "| operational:", status.operational)

circuits = [repetition_circuit(r, L) for r in ROUNDS for L in LOGICALS]
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa = [pm.run(c) for c in circuits]
for c in isa:
    ops = c.count_ops()
    print(f"  {c.name}: depth={c.depth()} ops={dict(ops)}")

job = SamplerV2(mode=backend).run(isa, shots=SHOTS)
record = {
    "job_id": job.job_id(),
    "backend": backend.name,
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "pending_jobs_at_submit": status.pending_jobs,
    "shots": SHOTS,
    "circuits": [
        {"name": c.name, "rounds": r, "logical": L,
         "layout": c.layout.final_index_layout()[:5], "depth": c.depth(),
         "ops": {k: int(v) for k, v in c.count_ops().items()}}
        for c, (r, L) in zip(isa, [(r, L) for r in ROUNDS for L in LOGICALS])
    ],
    "qubit_order": "layout lists physical qubits for d0, d1, d2, a0, a1",
    "transpile": "generate_preset_pass_manager(optimization_level=1)",
    "qiskit": qiskit.__version__, "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__,
    "authorisation": "student, chat, 2026-11-22: '授权提交'",
}
out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
print("submitted job:", record["job_id"])
