# LogiEdge – Edge AI cold-chain monitoring (FreightBridge pilot)

Edge pipeline for refrigerated trucks: sensor simulator → local MQTT → preprocessing (feature-level fusion) →
TFLite INT8 classifier (Normal / Warning / Critical) → MQTT alerts + local alert log, with PSI drift monitoring,
Docker packaging and Ansible deployment. Works fully offline; alerts are queued and bridged when coverage returns.

| Component | Files |
|---|---|
| A – Architecture | `scenario_architecture/constraint_analysis.md` (+ diagram source) |
| B – Hardware | `hardware/hardware_justification.md` |
| C – Sensors / MQTT | `data_pipeline/simulator.py`, `preprocessing.py`, `training_stats.npy`, `mqtt_architecture.md`, `deployment/mosquitto.conf` |
| D – Model / Docker | `training/*.py`, `training/models/`, `inference/Dockerfile`, `inference/inference_service.py`, `inference/model.tflite` |
| E – MLOps | `monitoring/drift_monitor.py`, `monitoring/reference_dist.json`, `deployment/logibridge_deploy.yml` |
| F – Optimisation | `optimisation/benchmark.py`, `optimisation/results/` |
| Report data | `reports/report_notes.md` |

## 1. Setup

Python 3.11 (matches the Docker image):

```bash
pip install -r requirements.txt
```

Mosquitto, Docker and Ansible are Linux tools here — run sections 3–5 in WSL Ubuntu (or any Linux box):

```bash
sudo apt update && sudo apt install -y mosquitto mosquitto-clients docker.io docker-buildx ansible python3-docker python3-pip
```

```bash
sudo usermod -aG docker $USER && ansible-galaxy collection install community.docker
```

## 2. Train and optimise (offline, ~2 min)

```bash
python training/generate_dataset.py      # training_stats.npy + dataset (117/87/87 windows)
python training/train_model.py           # M1 FP32, >88% gate, 3-sigma normalisation experiment
python training/convert_ptq.py           # M2 full INT8 -> also copied to inference/model.tflite
python training/prune_quantise.py        # M3 35% neuron pruning + INT8
python monitoring/drift_monitor.py --build-reference   # reference_dist.json (300 clean windows)
python optimisation/benchmark.py --tdp 15              # CSV + Pareto chart (set your laptop's TDP)
```

## 3. Run live (3 terminals)

The apt package already runs Mosquitto on `localhost:1883` (`systemctl status mosquitto`).
`deployment/mosquitto.conf` is the production edge-node config (persistence + ops-centre bridge) — copy it to
`/etc/mosquitto/conf.d/` only after removing the duplicate `persistence*` lines from the default config.

```bash
python inference/inference_service.py
```

```bash
python monitoring/drift_monitor.py
```

```bash
python data_pipeline/simulator.py --anomaly none
```

Watch the topics: `mosquitto_sub -t 'logibridge/#' -v`. Switch the simulator to `--anomaly temp_drift | vibration | combined` (Ctrl+C and restart).
`--speed N` compresses simulated time for demos (the pipeline uses message timestamps, so results are identical).

## 4. Docker + OTA layer-cache demo

```bash
docker run -d -p 5000:5000 --name registry registry:2
docker build --build-context pipeline=data_pipeline -t localhost:5000/logibridge-inference:latest inference
docker run --rm --network host -e MODEL_PATH=/app/model.tflite -e TRUCK_ID=TRK001 localhost:5000/logibridge-inference:latest
```

OTA demo: replace only the model and rebuild — every layer except `COPY model.tflite .` shows `CACHED`:

```bash
cp training/models/m3_pruned_int8.tflite inference/model.tflite
docker build --build-context pipeline=data_pipeline -t localhost:5000/logibridge-inference:latest inference
docker history localhost:5000/logibridge-inference:latest
```

Switch variant without rebuilding: `-e MODEL_PATH=...` plus a volume, e.g. `-v $PWD/training/models:/models -e MODEL_PATH=/models/m1_fp32.tflite`.

## 5. Ansible (idempotency)

```bash
docker push localhost:5000/logibridge-inference:latest
cd deployment && ansible-playbook -i inventory.ini logibridge_deploy.yml -K
```

Run the same command again → recap shows `changed=0`.

## Demo video script (15–20 min)

1. Repo tour + architecture diagram (1 min)
2. Run training scripts: dataset counts, >88% accuracy, 3σ experiment, PTQ, pruning 32/16 → 21/10 (3 min)
3. Benchmark + Pareto chart, deployment recommendation (2 min)
4. Live: mosquitto + inference + simulator `none` → `temp_drift` → `combined`; show `mosquitto_sub` and `alerts.jsonl` (3 min)
5. Offline proof: stop the bridge / pull the network; alerts still logged locally (1 min)
6. Docker build, OTA rebuild showing CACHED layers, bandwidth saving for 85 trucks, `MODEL_PATH` switch (3 min)
7. PSI: clean → inject `combined` → `[LOGIBRIDGE DRIFT ALERT]` → restore → PSI < 0.10 (4 min)
8. Ansible run twice, second `changed=0` (2 min)
