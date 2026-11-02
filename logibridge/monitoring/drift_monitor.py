"""PSI drift monitoring on the model's output confidence distribution.

Confidence score = P(Normal) from the softmax. Max-class probability would not work here:
under a fault the model is *confidently* Critical, so max-prob stays in [0.75, 1.0] and PSI
never moves. P(Normal) collapses towards 0, which is exactly the shift we want to see.

  python drift_monitor.py --build-reference [--model ../inference/model.tflite]
      300 clean Normal windows -> 4-bin distribution -> reference_dist.json
  python drift_monitor.py [--host localhost]
      subscribe to logibridge/trucks/+/inference, PSI over last 100 inferences every 60 s
"""
import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "inference"))
from inference_service import load_model, predict  # noqa: E402
from preprocessing import load_stats, normalise, windows_from_stream  # noqa: E402
from simulator import generate  # noqa: E402

BINS = [0.0, 0.25, 0.50, 0.75, 1.0]
REF_PATH = os.path.join(HERE, "reference_dist.json")
EPS = 1e-4  # floor for empty bins so ln(a/r) stays finite


def distribution(scores):
    counts, _ = np.histogram(np.clip(scores, 0, 1), bins=BINS)  # last bin is closed: [0.75, 1.0]
    return counts / max(len(scores), 1)


def psi(ref, cur):
    r, c = np.maximum(ref, EPS), np.maximum(cur, EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def build_reference(model_path):
    feats = windows_from_stream(generate("none", duration_s=3030, seed=7))[:300]
    interp, stats = load_model(model_path), load_stats()
    scores = [predict(interp, normalise(f, stats))[0] for f in feats]
    ref = {"bins": BINS, "score": "p_normal", "n_windows": len(scores), "model": os.path.basename(model_path),
           "distribution": distribution(scores).round(4).tolist()}
    with open(REF_PATH, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"reference from {len(scores)} clean windows: {ref['distribution']} -> {REF_PATH}")


def monitor(host, port, window, interval):
    import paho.mqtt.client as mqtt

    with open(REF_PATH) as f:
        ref = np.array(json.load(f)["distribution"])
    scores = deque(maxlen=window)

    def on_message(client, userdata, msg):
        scores.append(json.loads(msg.payload)["p_normal"])

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="drift-monitor")
    client.on_message = on_message
    client.connect(host, port)
    client.subscribe("logibridge/trucks/+/inference", qos=1)
    client.loop_start()
    print(f"[DRIFT] reference={ref.tolist()} window={window} every {interval}s", flush=True)
    try:
        while True:
            time.sleep(interval)
            if not scores:
                print("[DRIFT] no inferences yet", flush=True)
                continue
            cur = distribution(list(scores))
            value = psi(ref, cur)
            print(f"[DRIFT] {time.strftime('%H:%M:%S')} n={len(scores):>3} dist={cur.round(2).tolist()} PSI={value:.3f}",
                  flush=True)
            if value > 0.25:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
    except KeyboardInterrupt:
        client.loop_stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--model", default=os.path.join(ROOT, "inference", "model.tflite"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    if args.build_reference:
        build_reference(args.model)
    else:
        monitor(args.host, args.port, args.window, args.interval)


if __name__ == "__main__":
    main()
