# Hardware justification – worked numbers (Component B)

> Calculation notes. The argument in your report must be in your own words.

## B1 – Constraint Triangle (Performance / Power / Cost)

| | RPi 5 + AI HAT+ | Jetson Orin Nano Super | STM32H7 MCU |
|---|---|---|---|
| Compute | 13 TOPS NPU + 4× A76 | 67 TOPS | ~0.5 GFLOP/s-class Cortex-M7 |
| TDP vs 10 W AI budget | 7.5 W ✅ | 15 W ❌ (exceeds budget) | 0.4 W ✅ |
| Pilot cost (85) | ₹12.75 lakh | ₹38.25 lakh | ₹2.98 lakh |
| Fleet cost (265) | ₹39.75 lakh | ₹1.19 crore | ₹9.28 lakh |
| Runs Linux / Docker / Mosquitto / Ansible | yes | yes | no |
| Our model (≈3 KB INT8, < 0.1 ms) | huge headroom | huge headroom | fits (2 MB flash / 1 MB SRAM) |

- The model is tiny, so **Performance is not the binding vertex** — all three meet the 90 s SLA by orders of magnitude.
- **Power is the hard gate**: 10 W budget from the 12 V supply via DC-DC → Jetson (15 W) is eliminated.
- Between the remaining two the trade is **cost vs. operability**: the MCU is ~₹11,500/truck cheaper (₹30.5 lakh at 265 trucks)
  but cannot host the required stack (Docker OTA, Mosquitto broker + store-and-forward, Ansible, PSI monitor),
  and has no headroom for richer models (e.g. raw 500 Hz vibration spectra) after the pilot.
- Suggested position: **RPi 5 + AI HAT+** (dominant vertex: Power, then Cost at fleet scale). Decide and argue it yourself.

## B2 – Arithmetic Intensity and Roofline (RPi 5 CPU)

- AI = 45 MFLOP / 18 MB = **2.5 FLOP/byte**
- Ridge point = 16 GFLOP/s / 12 GB/s = **1.33 FLOP/byte**
- AI (2.5) > ridge (1.33) → **compute-bound**
- Attainable = min(16, 2.5 × 12 = 30) = 16 GFLOP/s → t ≈ 45e6 / 16e9 = **2.8 ms/inference** (memory time 18 MB / 12 GB/s = 1.5 ms)
- Because it is compute-bound, reduce FLOPs or raise compute throughput: INT8 quantisation (NEON int8 dot-product ≈ 4× FLOP/s
  and 4× fewer bytes), structured pruning (35% fewer neurons → fewer MACs), or offload to the Hailo-8L NPU.
  Increasing memory bandwidth would not help.
