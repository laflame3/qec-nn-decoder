"""Export and verify the Sycamore r05 packet. Run with Python -I."""

import hashlib
import sys
from pathlib import Path

import numpy as np

if not sys.flags.isolated:
    raise SystemExit("Use ../../.venv/bin/python -I src/export_e2_packet.py")

# Isolated mode excludes the project source directory from the import path.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sycamore_data import instance_path, load  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data" / "e2_r05_packet"
DECODERS = ("correlated_matching", "belief_matching", "harmony")


def export_packet():
    source = instance_path(14, "d3_at_q4_5", 5)
    instance = load(source)
    expected = {"detectors": instance.detectors, "obs": instance.obs_actual}
    assert instance.detectors.shape == (75_000, 56)
    assert instance.obs_actual.shape == (75_000,)

    dem_paths = sorted((source / "dems").glob("*.dem"))
    assert len(dem_paths) == 8, len(dem_paths)
    copies = [source / name for name in ("circuit_ideal.stim", "metadata.json")] + dem_paths
    originals = [source / name for name in ("detection_events.b8", "obs_flips_actual.b8")] + copies
    for dem_path in dem_paths:
        dem = dem_path.stem
        for decoder in DECODERS:
            prediction = instance.predicted(dem, decoder)
            assert prediction.shape == instance.obs_actual.shape, (dem, decoder, prediction.shape)
            expected[f"google_pred_{dem}_{decoder}"] = prediction
            originals.append(source / "obs_flips_predicted" / f"dem_{dem}" / f"{decoder}.b8")

    (PACKET / "dems").mkdir(parents=True, exist_ok=True)
    for path in copies:
        (PACKET / path.relative_to(source)).write_bytes(path.read_bytes())
    packed = {key: np.packbits(values, axis=-1) for key, values in expected.items() if key != "obs"}
    np.savez_compressed(PACKET / "r05.npz", **packed, obs=expected["obs"])

    with np.load(PACKET / "r05.npz", allow_pickle=False) as packet:
        assert set(packet.files) == set(expected)
        for key, values in expected.items():
            if key == "obs":
                np.testing.assert_array_equal(packet[key], values, err_msg=key)
            else:
                np.testing.assert_array_equal(packet[key], packed[key], err_msg=key)
                unpacked = np.unpackbits(packet[key], axis=-1, count=values.shape[-1])
                np.testing.assert_array_equal(unpacked, values, err_msg=key)
    for path in copies:
        assert (PACKET / path.relative_to(source)).read_bytes() == path.read_bytes(), path

    manifest = [
        "# Sycamore r05 data packet",
        "",
        "Source: Google Sycamore surface-code dataset, Zenodo record 11403595 (v1.0.0),",
        "DOI [10.5281/zenodo.11403595](https://doi.org/10.5281/zenodo.11403595).",
        "License: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).",
        "",
        "This packet selects `sample_14/d3_at_q4_5/X/r05` and preserves the original shot order.",
        "The circuit, metadata and DEMs are byte-for-byte copies. The original Stim b8 arrays",
        "are decoded and repackaged in `r05.npz` as follows:",
        "",
        "- `detectors`: `uint8` of shape `(75000, 7)`. Decode with",
        "  `np.unpackbits(detectors, axis=1, count=56)` to get `(75000, 56)`.",
        "- `obs`: Boolean ground truth of shape `(75000,)`.",
        "- `google_pred_<dem>_<decoder>`: 24 packed shot vectors. Use",
        "  `np.unpackbits(prediction, count=75000)` for each vector.",
        "- Both packed formats use NumPy's default `bitorder='big'`.",
        "",
        "## Original files",
        "",
        "Paths are relative to the project root. Hashes are of the original bytes.",
        "",
        "| Original path | SHA-256 |",
        "| --- | --- |",
    ]
    for path in sorted(originals):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"| `{path.relative_to(ROOT).as_posix()}` | `{digest}` |")
    (PACKET / "DATA_README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Verified {len(expected)} arrays and {len(copies)} byte-for-byte copies in {PACKET}")


if __name__ == "__main__":
    export_packet()
