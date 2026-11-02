"""Generate the labelled LogiEdge dataset and the normalisation stats.

Runs the simulator generator in accelerated (simulated) time - the same code path the
live simulator publishes over MQTT - and passes it through the streaming preprocessor.

Outputs:
  data_pipeline/training_stats.npy   mean/std of 6 features from 10 min of clean Normal data
  training/data/dataset.npz          X (raw features), y (labels)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
from preprocessing import FEATURES, STATS_PATH, windows_from_stream  # noqa: E402
from simulator import generate  # noqa: E402

# (label, simulator mode, minutes, seed)
PLAN = [(0, "none", 20, 1), (1, "temp_drift", 15, 2), (2, "combined", 15, 3)]


def main():
    clean = windows_from_stream(generate("none", duration_s=10 * 60, seed=100))
    stats = np.stack([clean.mean(axis=0), clean.std(axis=0)]).astype(np.float32)
    np.save(STATS_PATH, stats)
    print(f"training_stats.npy from {len(clean)} clean windows")
    for name, m, s in zip(FEATURES, *stats):
        print(f"  {name:<20} mean={m:9.4f} std={s:8.4f}")

    X, y = [], []
    for label, mode, minutes, seed in PLAN:
        w = windows_from_stream(generate(mode, duration_s=minutes * 60, seed=seed))
        X.append(w)
        y.append(np.full(len(w), label))
        print(f"class {label} ({mode:<10}) {minutes} min -> {len(w)} windows")
    out = os.path.join(ROOT, "training", "data", "dataset.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, X=np.concatenate(X), y=np.concatenate(y))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
