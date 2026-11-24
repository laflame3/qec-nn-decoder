"""Check / retrieve the E3 IBM job recorded in data/results/e3_ibm_job.json.

Never resubmits. If the job is done, saves per-shot data to
data/results/e3_ibm_raw.npz (syn_<name>: shots x rounds x 2, data_<name>:
shots x 3) and adds status and usage to the job record.
"""

import json
import pathlib
import sys

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ibm_repetition import split_shots  # noqa: E402

res = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"
rec_path = res / "e3_ibm_job.json"
rec = json.loads(rec_path.read_text())

service = QiskitRuntimeService(name="kcl-open-plan")
job = service.job(rec["job_id"])
status = str(job.status())
print("job", rec["job_id"], "on", rec["backend"], "status:", status)

if status in ("DONE", "JobStatus.DONE"):
    result = job.result()
    arrays = {}
    for c, pub in zip(rec["circuits"], result):
        syn, data = split_shots(pub.data, c["rounds"])
        arrays[f"syn_{c['name']}"] = syn.astype(np.uint8)
        arrays[f"data_{c['name']}"] = data.astype(np.uint8)
        maj = (data.sum(1) >= 2).astype(int)
        print(f"  {c['name']}: shots={len(data)} majority-vote error={np.mean(maj != c['logical']):.4f}")
    np.savez_compressed(res / "e3_ibm_raw.npz", **arrays)
    rec["status"] = "DONE"
    try:
        rec["usage"] = job.usage_estimation
        rec["metrics"] = job.metrics()
    except Exception as e:  # usage fields vary between runtime versions
        rec["usage_error"] = repr(e)
    rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    print("saved e3_ibm_raw.npz")
else:
    rec["last_status"] = status
    rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
