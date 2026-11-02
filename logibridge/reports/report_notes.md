# Report notes – measured data and calculations

> Raw material for `final_report.pdf` (2,500–3,000 words, your own words). Numbers marked *measured* come from
> this repo's scripts on the dev laptop — re-run them and use your own values. Rubric: the report must contain
> "substantial original analysis and measured data".

## Section 3 – 10-stage Edge ML pipeline mapping (rename stages to match your Module 4 slides)

| # | Stage | LogiEdge |
|---|---|---|
| 1 | Data collection | `simulator.py` emits temperature (1 Hz), compressor vibration RMS (0.5 Hz) and door events per truck over MQTT. |
| 2 | Data ingestion | Local Mosquitto broker on the truck; topic tree `logibridge/trucks/{id}/sensors/*`, QoS 0/1 per stream. |
| 3 | Preprocessing | 5-sample moving average, 30 s / 10 s sliding window (`preprocessing.py`). |
| 4 | Feature engineering & fusion | 3 temperature + 3 vibration features, feature-level fusion into a 6-vector, z-scored with frozen `training_stats.npy`. |
| 5 | Labelling & dataset | `generate_dataset.py`: none / temp_drift / combined runs → 117 / 87 / 87 windows, 80/20 stratified split. |
| 6 | Training & validation | 6-32-16-3 ReLU MLP, 100 % validation accuracy (gate: > 88 %). |
| 7 | Optimisation & conversion | Full-INT8 PTQ (234 calibration windows); 35 % neuron pruning + INT8. |
| 8 | Packaging | `python:3.11-slim` image, deps layer before `COPY model.tflite .`, model selectable via `MODEL_PATH`. |
| 9 | Deployment / OTA | Ansible playbook pushes model + reference distribution and (re)starts the container from the local registry. |
| 10 | Monitoring & feedback | PSI on P(Normal) every 60 s over the last 100 inferences; drift alert > 0.25 triggers retraining/rollback. |

## Section 3 – OTA (D2, E3)

Model 280 KB = 0.28 MB, ₹0.10/MB, 85 trucks, update every 6 weeks (~8.7 cycles/yr).

| Strategy | Data per cycle | Cost per cycle | Cost per year |
|---|---|---|---|
| Full replacement (all 85 at once) | 85 × 0.28 = 23.8 MB | ₹2.38 | ₹20.7 |
| Canary (10, then 75) | 23.8 MB (+2.8 MB if the 10 roll back) | ₹2.38 – ₹2.66 | ₹20.7 – ₹23.1 |
| Shadow (new model runs alongside old for 14 days, predictions uplinked) | 23.8 MB + 85 × (8,640/day × 20 B × 14 d = 2.42 MB) = 229.4 MB | ₹22.94 | ₹199.6 |

Bandwidth cost is negligible in every case — the decision is about **safety and connectivity**:
full replacement exposes all 85 cold-chain trucks to an untested model at once; shadow mode is safest per decision
but needs 10× the uplink on routes with hour-long gaps and delays fixes by weeks. Canary (10 trucks on well-covered
routes, watch PSI + Critical recall, then roll out) is the suggested middle ground — your call to argue.

**Docker layer-cache saving (measure in demo):** `docker history logibridge-inference` → note total image size S MB vs.
model layer size m MB. Full-image push per cycle = 85 × S MB; layer-only = 85 × m MB. Example with S ≈ 150 MB:
85 × 150 = 12,750 MB = ₹1,275 vs. 85 × 0.003 MB ≈ ₹0.03 → saving ≈ ₹1,275 per cycle. In this repo the Ansible path
goes further: the model is bind-mounted from `/opt/logibridge`, so a model update ships only the file (no image pull at all).

## Section 4 – Benchmark (measured, laptop CPU, TDP estimate 15 W)

From `optimisation/results/benchmark_results.csv`:

| Variant | Mean ms | p95 ms | Size KB | Acc % | Energy mJ | C2 recall % |
|---|---|---|---|---|---|---|
| M1 FP32 | 0.0151 | 0.0188 | 5.19 | 100.0 | 0.0142 | 100.0 |
| M2 PTQ INT8 | 0.0183 | 0.0185 | 3.38 | 100.0 | 0.0171 | 100.0 |
| M3 Pruned + INT8 | 0.0454 | 0.0559 | 2.94 | 100.0 | 0.0432 | 100.0 |

Notes for the analysis:
- Latencies are tens of microseconds and dominated by Python/interpreter call overhead, not the MACs — differences between variants are within run-to-run noise. Re-run 3× and report the spread.
- On x86, INT8 is not faster for a model this small (quantise/dequantise overhead); the INT8 benefit is size and NPU/MCU compatibility.
- M3 hidden layers 32/16 → 21/10 (17/48 neurons = 35.4 % removed).

**F3 – latency budget:** 90 s SLA − 30 s window − 10 s step − ~10 s smoothing lag − MQTT/alerting (< 1 s) ≈ **40 s per inference budget**; measured
< 0.1 ms → ~10^5× headroom. Flash/SRAM: RPi 5 has 8 GB RAM / SD storage, model ≈ 3 KB; even an STM32H7 (2 MB flash / 1 MB SRAM) would fit it.
Class-2 recall on the validation set is 100 % for all three (> 95 % requirement).

## Section 5 – PSI drift test (measured, `--speed 20`, monitor `--interval 5`)

| Phase | PSI |
|---|---|
| Clean (before injection) | 0.000 |
| ~5 anomalous windows into `--anomaly combined` | 0.313 → `[LOGIBRIDGE DRIFT ALERT]` |
| Peak during injection | 2.719 |
| After restoring clean data (window fully refreshed) | 0.106 → 0.000 |

Drift type: sudden (step) covariate/concept shift in the input → abrupt change in the output-confidence distribution.
For the video, run at real time (`--speed 1`, `--interval 60`): PSI crosses 0.25 within ~1–2 min of injection; recovery below 0.10 needs ~100 clean inferences (≈ 17 min real time, or use `--speed 10` ≈ 1.7 min and say so).

**Genuine technical difficulty (example from building this):** tfmot `prune_low_magnitude` with a block covering a whole kernel column
crashes (the pooled mask is squeezed to rank 1), so neuron-level pruning was implemented with tfmot's `PolynomialDecay` schedule in a
Keras callback. Also: using max-softmax as the PSI score never drifts, because faults are predicted *confidently* — P(Normal) had to be used.
