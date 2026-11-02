"""LogiEdge on-truck inference service: MQTT sensors -> preprocessing -> TFLite -> MQTT result.

Environment:
  MODEL_PATH  model to load (switch variants without rebuilding the image)   default ./model.tflite
  TRUCK_ID    default TRK001
  MQTT_HOST / MQTT_PORT   local broker on the truck                         default localhost:1883
  ALERT_LOG   local append-only alert log (survives cellular outages)        default ./alerts.jsonl

Publishes  logibridge/trucks/{truck_id}/inference  every 10 s (QoS 1)
           logibridge/trucks/{truck_id}/alerts     on Warning/Critical (QoS 1)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_pipeline"))  # repo layout; in Docker it's alongside
from preprocessing import Preprocessor, load_stats, normalise  # noqa: E402

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

LABELS = ["Normal", "Warning", "Critical"]


def load_model(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def predict(interp, x):
    """x: normalised float32 feature vector (6,). Returns class probabilities (3,)."""
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    x = x.reshape(1, -1).astype(np.float32)
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"])[0]
    if out["dtype"] == np.int8:
        scale, zp = out["quantization"]
        y = (y.astype(np.float32) - zp) * scale
    return y


def main():
    import paho.mqtt.client as mqtt

    model_path = os.environ.get("MODEL_PATH", os.path.join(HERE, "model.tflite"))
    truck = os.environ.get("TRUCK_ID", "TRK001")
    alert_log = os.environ.get("ALERT_LOG", os.path.join(HERE, "alerts.jsonl"))
    interp, stats, pre = load_model(model_path), load_stats(), Preprocessor()
    door = {"state": "CLOSE"}
    base = f"logibridge/trucks/{truck}"

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe(f"{base}/sensors/#", qos=1)
        print(f"[INFER] connected, model={model_path}, truck={truck}", flush=True)

    def on_message(client, userdata, msg):
        stream = msg.topic.rsplit("/", 1)[-1]
        m = json.loads(msg.payload)
        if stream == "door_event":
            door["state"] = m["value"]
            return
        feats = pre.add(stream, m["ts"], m["value"])
        if feats is None:
            return
        t0 = time.perf_counter()
        probs = predict(interp, normalise(feats, stats))
        latency_ms = (time.perf_counter() - t0) * 1000
        cls = int(np.argmax(probs))
        result = {
            "truck_id": truck, "ts": m["ts"], "class": cls, "label": LABELS[cls],
            "probs": [round(float(p), 4) for p in probs], "p_normal": round(float(probs[0]), 4),
            "latency_ms": round(latency_ms, 3), "model": os.path.basename(model_path), "door": door["state"],
            "features": [round(float(f), 4) for f in feats],
        }
        client.publish(f"{base}/inference", json.dumps(result), qos=1)
        print(f"[INFER] {LABELS[cls]:<8} p={result['probs']} temp_mean={feats[0]:.2f} vib_rms={feats[3]:.2f} "
              f"{latency_ms:.2f} ms", flush=True)
        if cls > 0:
            # written locally first: the alert exists even if the uplink is down
            with open(alert_log, "a") as f:
                f.write(json.dumps(result) + "\n")
            client.publish(f"{base}/alerts", json.dumps(result), qos=1)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"infer-{truck}")
    client.on_connect, client.on_message = on_connect, on_message
    client.connect(os.environ.get("MQTT_HOST", "localhost"), int(os.environ.get("MQTT_PORT", 1883)))
    client.loop_forever()


if __name__ == "__main__":
    main()
