"""Load one decoding instance of the Zenodo 11403595 Sycamore surface-code data.

An instance is one (sample, patch, basis, rounds) directory, e.g.
    data/raw/zenodo_11403595/sycamore_surface_code_d3_d5/sample_00/d3_at_q4_5/X/r05
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

DATA_ROOT = Path(__file__).resolve().parents[1] / "data/raw/zenodo_11403595/sycamore_surface_code_d3_d5"


@dataclass
class Instance:
    path: Path
    meta: dict
    circuit: stim.Circuit
    detectors: np.ndarray        # (shots, num_detectors) bool
    obs_actual: np.ndarray       # (shots,) bool

    def dem(self, method: str = "simple") -> stim.DetectorErrorModel:
        return stim.DetectorErrorModel.from_file(self.path / "dems" / f"{method}.dem")

    def predicted(self, dem_method: str, decoder: str) -> np.ndarray:
        f = self.path / "obs_flips_predicted" / f"dem_{dem_method}" / f"{decoder}.b8"
        return stim.read_shot_data_file(path=str(f), format="b8", num_observables=1)[:, 0]

    def logical_error_rate(self, pred: np.ndarray) -> float:
        return float(np.mean(pred != self.obs_actual))


def instance_path(sample: int, patch: str, rounds: int, basis: str = "X") -> Path:
    return DATA_ROOT / f"sample_{sample:02d}" / patch / basis / f"r{rounds:02d}"


def load(path: Path) -> Instance:
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    circuit = stim.Circuit.from_file(str(path / "circuit_ideal.stim"))
    det = stim.read_shot_data_file(path=str(path / "detection_events.b8"), format="b8",
                                   num_detectors=circuit.num_detectors)
    obs = stim.read_shot_data_file(path=str(path / "obs_flips_actual.b8"), format="b8",
                                   num_observables=circuit.num_observables)[:, 0]
    assert det.shape[0] == obs.shape[0] == meta["shots"], (det.shape, obs.shape, meta["shots"])
    return Instance(path, meta, circuit, det, obs)
