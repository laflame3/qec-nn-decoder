"""Export the 52 non-target Sycamore samples. Run with Python -I."""

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

if not sys.flags.isolated:
    raise SystemExit("Use ../../.venv/bin/python -I src/export_e2_pool.py")

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from sycamore_data import instance_path, load

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "raw"
PACKET = DATA / "e2_r05_packet"
POOL = DATA / "e2_pool"
N_SHOTS, N_DETECTORS, TEST_SAMPLE = 75_000, 56, 14
SAMPLES = tuple(sample for sample in range(53) if sample != TEST_SAMPLE)
CIRCUIT_HASH = "a45f6fd7d9b2da7c4896783741baf5f5ab9684264c3382cbdc7a6466af4cc8a5"
SOURCE_FILES = ("circuit_ideal.stim", "metadata.json", "detection_events.b8",
                "obs_flips_actual.b8", "dems/simple.dem", "dems/correlations.dem")
FIELDS = ("sample", "mean_lit_detectors", "undecoded_flip_rate", "circuit_sha256",
          "simple_dem_sha256", "correlations_dem_sha256", "role")
OUTPUTS = ("pool.npz", "samples.csv")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def sample_row(sample, det, obs, hashes):
    return {
        "sample": sample, "mean_lit_detectors": float(det.sum() / len(obs)),
        "undecoded_flip_rate": float(obs.mean()),
        "circuit_sha256": hashes["circuit_ideal.stim"],
        "simple_dem_sha256": hashes["dems/simple.dem"],
        "correlations_dem_sha256": hashes["dems/correlations.dem"],
        "role": "test_sample" if sample == TEST_SAMPLE else "training_pool",
    }


def main():
    packet_hashes = {name: sha256(PACKET / name) for name in
                     ("circuit_ideal.stim", "dems/simple.dem", "dems/correlations.dem", "r05.npz")}
    if packet_hashes["circuit_ideal.stim"] != CIRCUIT_HASH:
        raise ValueError("sample_14 packet circuit differs from the expected SHA-256")

    sources = {sample: instance_path(sample, "d3_at_q4_5", 5, "X") for sample in SAMPLES}
    hashes_by_sample = {
        sample: {name: sha256(source / name) for name in SOURCE_FILES}
        for sample, source in sources.items()
    }
    manifest = {
        "schema_version": 1,
        "raw_files": {(sources[sample] / name).relative_to(RAW).as_posix(): digest
                      for sample, hashes in hashes_by_sample.items() for name, digest in hashes.items()},
        "packet_files": packet_hashes,
    }
    mismatches = [
        f"sample_{sample:02d}/{name}: {hashes[name]} (expected {packet_hashes[name]})"
        for sample, hashes in hashes_by_sample.items()
        for name in ("circuit_ideal.stim", "dems/simple.dem")
        if hashes[name] != packet_hashes[name]
    ]
    if mismatches:
        raise ValueError("Source SHA-256 mismatch:\n" + "\n".join(mismatches))

    manifest_path = POOL / "manifest.json"
    if any((POOL / name).exists() for name in (*OUTPUTS, "manifest.json")):
        if not all((POOL / name).is_file() for name in (*OUTPUTS, "manifest.json")):
            raise ValueError(f"Incomplete existing pool: {POOL}")
        manifest["output_files"] = {name: sha256(POOL / name) for name in OUTPUTS}
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Source or output hashes differ from {manifest_path}")
        print(f"Verified existing pool: {len(SAMPLES) * N_SHOTS:,} shots, {len(SAMPLES)} samples; skip")
        return

    n = len(SAMPLES) * N_SHOTS
    detectors = np.empty((n, N_DETECTORS // 8), dtype=np.uint8)
    obs = np.empty(n, dtype=bool)
    rows = []
    for i, (sample, source) in enumerate(sources.items()):
        instance = load(source)
        if (instance.circuit.num_detectors != N_DETECTORS
                or instance.detectors.shape != (N_SHOTS, N_DETECTORS)
                or instance.obs_actual.shape != (N_SHOTS,)):
            raise ValueError(f"Unexpected detector/observation shape: {source.relative_to(RAW)}")
        rows.append(sample_row(sample, instance.detectors, instance.obs_actual, hashes_by_sample[sample]))
        block = slice(i * N_SHOTS, (i + 1) * N_SHOTS)
        detectors[block] = np.packbits(instance.detectors, axis=1, bitorder="big")
        obs[block] = instance.obs_actual

    with np.load(PACKET / "r05.npz", allow_pickle=False) as file:
        test_det = np.unpackbits(file["detectors"], axis=1, count=N_DETECTORS, bitorder="big")
        test_obs = file["obs"]
    if test_det.shape != (N_SHOTS, N_DETECTORS) or test_obs.shape != (N_SHOTS,):
        raise ValueError("Unexpected packet detector/observation shape")
    rows.append(sample_row(TEST_SAMPLE, test_det, test_obs, packet_hashes))

    POOL.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=POOL) as temp:
        staging = Path(temp)
        np.savez_compressed(staging / "pool.npz", detectors=detectors, obs=obs,
                            sample=np.repeat(np.asarray(SAMPLES, dtype=np.int16), N_SHOTS),
                            shot=np.tile(np.arange(N_SHOTS, dtype=np.int32), len(SAMPLES)))
        with (staging / "samples.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["sample"]))
        manifest["output_files"] = {name: sha256(staging / name) for name in OUTPUTS}
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name in (*OUTPUTS, "manifest.json"):
            (staging / name).rename(POOL / name)
    print(f"Exported {n:,} shots from {len(SAMPLES)} samples; wrote {manifest_path.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
