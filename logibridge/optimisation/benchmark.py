"""Five-metric benchmark of M1/M2/M3 (all run through the TFLite interpreter, same harness).

Metrics: mean latency, p95 latency (200 runs after 10 warm-up), size, validation accuracy,
energy per inference E = P x t with P = TDP x (process CPU% / cores). Also class-2 recall.
Outputs results/benchmark_results.csv and results/pareto_chart.png.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "inference"))
from inference_service import load_model, predict  # noqa: E402
from preprocessing import load_stats, normalise  # noqa: E402

MODELS = os.path.join(ROOT, "training", "models")
VARIANTS = [("M1 FP32", "m1_fp32.tflite"), ("M2 PTQ INT8", "m2_ptq_int8.tflite"), ("M3 Pruned+INT8", "m3_pruned_int8.tflite")]


def bench(path, X, y, tdp_w, runs=200, warmup=10):
    interp = load_model(path)
    pred = np.array([predict(interp, x).argmax() for x in X])
    for i in range(warmup):
        predict(interp, X[i % len(X)])
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        predict(interp, X[i % len(X)])
        times.append((time.perf_counter() - t0) * 1000)
    # 200 runs last only a few ms, below psutil's sampling resolution: sample CPU% over 1 s of back-to-back inference
    proc = psutil.Process()
    proc.cpu_percent(None)
    end, i = time.perf_counter() + 1.0, 0
    while time.perf_counter() < end:
        predict(interp, X[i % len(X)])
        i += 1
    cpu_frac = proc.cpu_percent(None) / 100 / psutil.cpu_count()
    mean_ms = float(np.mean(times))
    return {
        "mean_latency_ms": round(mean_ms, 4),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 4),
        "size_kb": round(os.path.getsize(path) / 1024, 2),
        "accuracy_pct": round(100 * float(np.mean(pred == y)), 2),
        "class2_recall_pct": round(100 * float(np.mean(pred[y == 2] == 2)), 2),
        "energy_mj": round(tdp_w * cpu_frac * mean_ms, 6),  # W x ms = mJ
        "cpu_pct_of_package": round(100 * cpu_frac, 2),
    }


def pareto(rows):
    """Not dominated on (latency low, size low, accuracy high)."""
    key = lambda r: (r["mean_latency_ms"], r["size_kb"], -r["accuracy_pct"])  # noqa: E731
    return [all(not (all(a <= b for a, b in zip(key(o), key(r))) and key(o) != key(r)) for o in rows) for r in rows]


def plot(rows, flags, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, p in zip(rows, flags):
        ax.scatter(r["size_kb"], r["mean_latency_ms"], s=120, c="tab:green" if p else "tab:gray", zorder=3)
        ax.annotate(f"{r['variant']}\nacc {r['accuracy_pct']}%  C2 recall {r['class2_recall_pct']}%",
                    (r["size_kb"], r["mean_latency_ms"]), textcoords="offset points", xytext=(8, 6), fontsize=8)
    front = sorted((r for r, p in zip(rows, flags) if p), key=lambda r: r["size_kb"])
    ax.plot([r["size_kb"] for r in front], [r["mean_latency_ms"] for r in front], "g--", label="Pareto frontier")
    ax.set_xlabel("Model size (KB)")
    ax.set_ylabel("Mean inference latency (ms)")
    ax.set_title("LogiEdge variants - lower-left is better (green = Pareto-optimal)")
    ax.margins(0.35)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdp", type=float, default=15.0, help="laptop CPU TDP estimate in watts")
    args = ap.parse_args()
    d = np.load(os.path.join(ROOT, "training", "data", "split.npz"))
    X, y = normalise(d["X_val"], load_stats()), d["y_val"]

    rows = [{"variant": name, "file": f, **bench(os.path.join(MODELS, f), X, y, args.tdp)} for name, f in VARIANTS]
    flags = pareto(rows)
    for r, p in zip(rows, flags):
        r["pareto_optimal"] = p
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, flags, os.path.join(out_dir, "pareto_chart.png"))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
