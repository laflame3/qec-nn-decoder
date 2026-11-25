"""Retrieve the E3 v2 job recorded in data/results/e3_ibm_v2_job.json. Never resubmits.

    ../../.venv/bin/python -I src/retrieve_e3_ibm_v2.py

When DONE, saves data/results/e3_ibm_v2_raw.npz with one entry per PUB, in
submission order: syn_p{i} (shots x 9 x 2), data_p{i} (shots x 3), plus
pub_block, pub_tau_us, pub_logical arrays. Adds status and usage to the record.
Refuses to overwrite an existing raw file.
"""

import json
import pathlib
import sys

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ibm_repetition import split_shots  # noqa: E402

RES = pathlib.Path(__file__).resolve().parent.parent / "data" / "results"
REC = RES / "e3_ibm_v2_job.json"
RAW = RES / "e3_ibm_v2_raw.npz"

rec = json.loads(REC.read_text())
job = QiskitRuntimeService(name="kcl-open-plan").job(rec["job_id"])
status = str(job.status())
print("job", rec["job_id"], "status:", status)
if "DONE" not in status:
    rec["last_status"] = status
    REC.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    sys.exit(0)
if RAW.exists():
    sys.exit(f"{RAW.name} already exists; not overwriting")

result = job.result()
order = rec["pub_order"]
assert len(result) == len(order) == 30
arrays = {"pub_block": np.array([o["block"] for o in order]),
          "pub_tau_us": np.array([o["tau_us"] for o in order]),
          "pub_logical": np.array([o["logical"] for o in order])}
for o, pub in zip(order, result):
    syn, data = split_shots(pub.data, rec["rounds"])
    assert syn.shape == (rec["block_shots"], rec["rounds"], 2) and data.shape == (rec["block_shots"], 3)
    arrays[f"syn_p{o['pub']}"] = syn.astype(np.uint8)
    arrays[f"data_p{o['pub']}"] = data.astype(np.uint8)
np.savez_compressed(RAW, **arrays)

for tau in rec["taus_us"]:
    for L in rec["logicals"]:
        idx = [o["pub"] for o in order if (o["tau_us"], o["logical"]) == (tau, L)]
        data = np.concatenate([arrays[f"data_p{i}"] for i in idx])
        syn = np.concatenate([arrays[f"syn_p{i}"] for i in idx])
        maj = ((data.sum(1) >= 2).astype(int) != L).mean()
        print(f"  tau={tau:>2} L={L}: shots={len(data):,} syndrome rate={syn.mean():.4f} majority error={maj:.4f}")

rec["status"] = "DONE"
try:
    rec["metrics"] = job.metrics()
except Exception as e:  # field names vary between runtime versions
    rec["metrics_error"] = repr(e)
rec["raw_file"] = RAW.name
REC.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
print("saved", RAW.name)
