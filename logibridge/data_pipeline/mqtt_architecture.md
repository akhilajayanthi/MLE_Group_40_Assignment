# MQTT architecture (Component C)

Broker: Mosquitto on the truck edge node (`localhost:1883`), config in [`deployment/mosquitto.conf`](../deployment/mosquitto.conf).
Everything on the truck talks to the local broker, so sensing → inference → alerting works with zero cellular coverage.
A Mosquitto bridge forwards only `alerts` and `inference` to the operations centre and queues them during coverage gaps.

## Topic tree

```
logibridge/
└── trucks/
    └── {truck_id}/                    e.g. TRK001
        ├── sensors/
        │   ├── temperature            1 Hz      QoS 0   {"truck_id","ts","value"}  degC
        │   ├── vibration_rms          0.5 Hz    QoS 0   {"truck_id","ts","value"}  g
        │   └── door_event             discrete  QoS 1   {"truck_id","ts","value": "OPEN"|"CLOSE"}
        ├── inference                  every 10 s QoS 1  class, label, probs, p_normal, latency_ms, features, door
        ├── alerts                     on class 1/2 QoS 1  same payload; bridged to ops centre
        └── ota/                       (in)      QoS 1   reserved for model-update notifications
```

## QoS per topic

| Topic | QoS | Why |
|---|---|---|
| `sensors/temperature` | 0 | 1 Hz and 5-sample smoothed: one lost reading changes nothing; QoS 0 has no ack overhead on the local loopback. |
| `sensors/vibration_rms` | 0 | Same reasoning: periodic, redundant within the 30 s window. |
| `sensors/door_event` | 1 | Discrete event that is never repeated; losing an OPEN breaks chain-of-custody evidence. |
| `inference` | 1 | Feeds the PSI monitor and the ops-centre dashboard; at-least-once, duplicates are harmless (keyed by `ts`). |
| `alerts` | 1 | Must survive the cellular gap: QoS 1 + persistent session (`cleansession false`) makes the bridge queue them and deliver when coverage returns. QoS 2 was rejected: exactly-once costs a 4-way handshake on a flaky link, and the consumer can de-duplicate on `(truck_id, ts)`. |

Alerts are additionally appended to a local `alerts.jsonl` before publishing (on the host volume `/opt/logibridge`, so it survives container updates) — this is the chain-of-custody record.

## Preprocessing (implemented in `preprocessing.py`)

1. **Filtering** – 5-sample moving average per stream.
2. **Windowing** – 30 s window, 10 s step; a feature vector is emitted every 10 s once 30 s of data exist.
3. **Features (feature-level fusion)** – temperature: mean, std, rate-of-change (°C/min, least-squares slope); vibration: RMS, peak, kurtosis. Concatenated into one 6-value vector.
4. **Normalisation** – z-score with `training_stats.npy` (computed once from 10 min of clean Normal data by `training/generate_dataset.py`, loaded at runtime, never recomputed from live data), clipped to ±10σ so runaway drift does not destroy the INT8 input scale.

## Normalisation experiment (measured, M1 FP32, 57 validation windows)

| Stats used | Accuracy | Recall Normal / Warning / Critical |
|---|---|---|
| Correct `training_stats.npy` | 100.00 % | 100 / 100 / 100 % |
| Means shifted by +3σ | 98.25 % | 100 / 94.1 / 100 % |

Source: `training/models/normalisation_experiment.json`. Re-run `training/train_model.py` and use your own numbers.
