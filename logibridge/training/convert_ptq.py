"""M2 - Post-Training Quantisation, full INT8 (int8 weights, activations, input and output).

Calibration: all training windows (>= 200 samples required).
Output: training/models/m2_ptq_int8.tflite (also copied to inference/model.tflite as the default deployed model).
"""
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
from preprocessing import load_stats, normalise  # noqa: E402

MODELS = os.path.join(ROOT, "training", "models")


def calibration_set():
    X = normalise(np.load(os.path.join(ROOT, "training", "data", "split.npz"))["X_train"], load_stats())
    assert len(X) >= 200, f"need >= 200 calibration samples, have {len(X)}"
    return X


def convert_int8(model, X_cal, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[None, :]] for x in X_cal)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    with open(out_path, "wb") as f:
        f.write(conv.convert())
    print(f"saved {out_path} ({os.path.getsize(out_path) / 1024:.2f} KB, {len(X_cal)} calibration samples)")


def main():
    model = tf.keras.models.load_model(os.path.join(MODELS, "m1_fp32.keras"))
    out = os.path.join(MODELS, "m2_ptq_int8.tflite")
    convert_int8(model, calibration_set(), out)
    shutil.copy(out, os.path.join(ROOT, "inference", "model.tflite"))


if __name__ == "__main__":
    main()
