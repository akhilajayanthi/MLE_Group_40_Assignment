"""LogiEdge cold-chain truck sensor simulator.

Streams (published to the local Mosquitto broker):
  temperature    1 Hz    N(4.0, 0.3) degC, setpoint 4 degC
  vibration_rms  0.5 Hz  N(0.45, 0.05) g compressor RMS
  door_event     discrete 'OPEN' / 'CLOSE'

Usage:
  python simulator.py --anomaly none|temp_drift|vibration|combined [--truck-id TRK001] [--speed 1]
"""
import argparse
import json
import time

import numpy as np

SETPOINT_C = 4.0
TOPIC = "logibridge/trucks/{truck_id}/sensors/{stream}"
# QoS per stream: periodic readings tolerate a lost sample, door events are discrete and must arrive
QOS = {"temperature": 0, "vibration_rms": 0, "door_event": 1}


def generate(anomaly="none", duration_s=None, start_ts=0.0, seed=None):
    """Yield (stream, ts, value) in simulated time, one tick per second."""
    rng = np.random.default_rng(seed)
    drift = anomaly in ("temp_drift", "combined")
    vib_fault = anomaly in ("vibration", "combined")
    door_open_until = None
    k = 0
    while duration_s is None or k < duration_s:
        ts = start_ts + k
        temp = rng.normal(SETPOINT_C, 0.3) + (0.08 * k if drift else 0.0)
        yield "temperature", ts, round(float(temp), 3)
        if k % 2 == 0:
            vib = rng.normal(1.2, 0.15) if vib_fault else rng.normal(0.45, 0.05)
            yield "vibration_rms", ts, round(float(vib), 4)
        # roughly one delivery stop every 10 minutes, door open 20-60 s
        if door_open_until is None and rng.random() < 1 / 600:
            door_open_until = k + int(rng.integers(20, 61))
            yield "door_event", ts, "OPEN"
        elif door_open_until is not None and k >= door_open_until:
            door_open_until = None
            yield "door_event", ts, "CLOSE"
        k += 1


def main():
    import paho.mqtt.client as mqtt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anomaly", choices=["none", "temp_drift", "vibration", "combined"], default="none")
    ap.add_argument("--truck-id", default="TRK001")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--duration", type=int, default=None, help="simulated seconds (default: run forever)")
    ap.add_argument("--speed", type=float, default=1.0, help="time acceleration for demos (1 = real time)")
    args = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{args.truck_id}")
    client.connect(args.host, args.port)
    client.loop_start()
    print(f"[SIM] truck={args.truck_id} anomaly={args.anomaly} speed={args.speed}x -> {args.host}:{args.port}")

    start_wall = time.time()
    try:
        for stream, ts, value in generate(args.anomaly, args.duration, start_ts=start_wall):
            # pace on simulated time so --speed only compresses the clock, not the data
            delay = start_wall + (ts - start_wall) / args.speed - time.time()
            if delay > 0:
                time.sleep(delay)
            payload = json.dumps({"truck_id": args.truck_id, "ts": ts, "value": value})
            client.publish(TOPIC.format(truck_id=args.truck_id, stream=stream), payload, qos=QOS[stream])
            if stream == "temperature" and int(ts - start_wall) % 30 == 0:
                print(f"[SIM] t={int(ts - start_wall):>5}s temp={value:.2f}C")
            elif stream == "door_event":
                print(f"[SIM] door {value}")
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
