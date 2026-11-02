"""LogiEdge preprocessing: filtering -> windowed feature extraction -> normalisation.

Feature-level fusion: temperature and vibration features are extracted independently
and concatenated into one 6-value vector:
  [temp_mean, temp_std, temp_rate_c_per_min, vib_rms, vib_peak, vib_kurtosis]
"""
from collections import deque
import os

import numpy as np

WINDOW_S = 30
STEP_S = 10
MA_SAMPLES = 5
FEATURES = ["temp_mean", "temp_std", "temp_rate_c_per_min", "vib_rms", "vib_peak", "vib_kurtosis"]
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_stats.npy")
# z-scores of a runaway drift reach the hundreds; clipping keeps the INT8 input scale usable
Z_CLIP = 10.0


class Preprocessor:
    """Streaming pipeline. Feed samples with add(); it returns a raw feature vector every STEP_S."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.raw = {"temperature": deque(maxlen=MA_SAMPLES), "vibration_rms": deque(maxlen=MA_SAMPLES)}
        self.buf = {"temperature": deque(), "vibration_rms": deque()}
        self.t0 = None
        self.next_emit = None

    def add(self, stream, ts, value):
        if stream not in self.buf:
            return None
        buf = self.buf[stream]
        if buf and ts < buf[-1][0]:  # simulator restarted with an older clock
            self.reset()
            buf = self.buf[stream]
        self.raw[stream].append(value)
        buf.append((ts, float(np.mean(self.raw[stream]))))  # 5-sample moving average
        if self.t0 is None:
            self.t0, self.next_emit = ts, ts + WINDOW_S
        if stream == "temperature" and ts >= self.next_emit:
            self.next_emit = ts + STEP_S
            return self.features(ts)
        return None

    def features(self, now):
        for b in self.buf.values():
            while b and b[0][0] <= now - WINDOW_S:
                b.popleft()
        tt, temp = np.array(self.buf["temperature"]).T
        vib = np.array([v for _, v in self.buf["vibration_rms"]])
        if len(temp) < 2 or len(vib) < 2:
            return None
        rate = np.polyfit(tt - tt[0], temp, 1)[0] * 60.0
        var = vib.var()
        kurt = float(np.mean((vib - vib.mean()) ** 4) / var**2 - 3.0) if var > 0 else 0.0
        return np.array([temp.mean(), temp.std(), rate, np.sqrt(np.mean(vib**2)), vib.max(), kurt], dtype=np.float32)


def windows_from_stream(samples):
    """Run the streaming preprocessor over (stream, ts, value) tuples; return raw feature matrix."""
    pre = Preprocessor()
    out = [f for s, t, v in samples if (f := pre.add(s, t, v)) is not None]
    return np.array(out, dtype=np.float32)


def load_stats(path=STATS_PATH):
    """training_stats.npy = [[mean x6], [std x6]] computed once from clean Normal data."""
    return np.load(path)


def normalise(x, stats):
    mean, std = stats
    return np.clip((x - mean) / np.where(std > 0, std, 1.0), -Z_CLIP, Z_CLIP).astype(np.float32)
