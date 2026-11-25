"""Pack the files the notebook reads into its own metadata (single-file submission, 2026-11-24).

The submission is one .ipynb (KEATS, QPR003). The notebook's setup cell finds this
archive in its own file, checks the SHA-256, and unpacks it to a temporary folder.
Run from the project root after any result file changes:
    ../../.venv/bin/python src/embed_notebook_data.py
"""

import base64
import hashlib
import io
import json
import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "qec_nn_decoder.ipynb"
KEY = "qec_embedded_data"
RESULTS = [  # only what the notebook reads (2026-11-24: no data packet, no weights)
    "e1_pm_fscan.csv", "e1_pm_fscan.settings.json", "e1_tune.csv", "e1_tune.settings.json",
    "e1_nn_fscan.csv", "e1_nn_fscan.settings.json", "e1_tf_vs_mlp.csv",
    "e1_lut.csv", "e1_lut.settings.json", "e1_ratio_ci.csv",
    "e1_structure.csv", "e1_structure.settings.json", "e1_datasize.csv", "e1_datasize.settings.json",
    "e1_finetune.csv", "e1_finetune.settings.json", "e1_pm_prior.csv", "e1_pm_prior.settings.json",
    "e2_baselines.csv", "e2_nn.csv", "e2_pairwise.csv", "e2_sim_check.csv", "e2_drift.csv",
    "e2_sycamore.settings.json",
    "e2v2_datasize.csv", "e2v2_datasize.settings.json", "e2v2_transfer.csv", "e2v2_transfer.settings.json",
    "e3v2_main.csv", "e3v2_main.settings.json", "e3v2_graph.csv", "e3v2_graph.settings.json",
    "e3v2_datasize.csv", "e3v2_datasize.settings.json", "e3v2_transfer.csv", "e3v2_transfer.settings.json",
    "e3v2_edges.json", "e3v2_edges.settings.json", "e3v2_diag.csv", "e3v2_diag.settings.json",
]


def files():
    return [ROOT / "data" / "results" / name for name in RESULTS]


def main():
    buf = io.BytesIO()
    listing = {}
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tar:
        for path in files():
            rel = str(path.relative_to(ROOT))
            info = tar.gettarinfo(str(path), arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(path, "rb") as fh:
                tar.addfile(info, fh)
            listing[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = buf.getvalue()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nb["metadata"][KEY] = {
        "description": "Result and settings files read by this notebook, tar.xz, base64.",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "files": listing,
        "tar_xz_base64": base64.b64encode(raw).decode("ascii"),
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"embedded {len(listing)} files, {len(raw):,} bytes compressed, into {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
