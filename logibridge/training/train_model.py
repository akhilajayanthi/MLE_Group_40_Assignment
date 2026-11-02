"""M1 - FP32 baseline: 2-hidden-layer MLP (32, 16, ReLU) on the 6-feature vector.

Also runs the mandatory normalisation experiment (correct stats vs stats shifted by 3 sigma).
Outputs in training/models/: m1_fp32.keras, m1_fp32.tflite, normalisation_experiment.json
and training/data/split.npz (fixed train/val split reused by every later script).
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
from preprocessing import load_stats, normalise  # noqa: E402

MODELS = os.path.join(ROOT, "training", "models")
DATA = os.path.join(ROOT, "training", "data")
SEED = 42


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])


def stratified_split(y, val_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    val = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: round(val_frac * np.sum(y == c))] for c in np.unique(y)])
    mask = np.zeros(len(y), bool)
    mask[val] = True
    return ~mask, mask


def main():
    tf.keras.utils.set_random_seed(SEED)
    d = np.load(os.path.join(DATA, "dataset.npz"))
    X, y = d["X"], d["y"]
    tr, va = stratified_split(y)
    np.savez(os.path.join(DATA, "split.npz"), X_train=X[tr], y_train=y[tr], X_val=X[va], y_val=y[va])

    stats = load_stats()
    Xtr, Xva = normalise(X[tr], stats), normalise(X[va], stats)
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, y[tr], validation_data=(Xva, y[va]), epochs=150, batch_size=16, verbose=0)

    pred = model.predict(Xva, verbose=0).argmax(1)
    acc = float(np.mean(pred == y[va]))
    print(f"train={tr.sum()} val={va.sum()}  validation accuracy = {acc:.2%}")
    for c in range(3):
        print(f"  class {c} recall = {np.mean(pred[y[va] == c] == c):.2%}")
    assert acc > 0.88, "validation accuracy must exceed 88% - revisit features/architecture"

    os.makedirs(MODELS, exist_ok=True)
    model.save(os.path.join(MODELS, "m1_fp32.keras"))
    with open(os.path.join(MODELS, "m1_fp32.tflite"), "wb") as f:
        f.write(tf.lite.TFLiteConverter.from_keras_model(model).convert())

    # Mandatory experiment: same model, normalisation stats shifted by +3 sigma
    shifted = np.stack([stats[0] + 3 * stats[1], stats[1]])
    pred_shift = model.predict(normalise(X[va], shifted), verbose=0).argmax(1)
    acc_shift = float(np.mean(pred_shift == y[va]))
    result = {
        "accuracy_correct_stats": acc,
        "accuracy_shifted_3sigma": acc_shift,
        "per_class_recall_shifted": [float(np.mean(pred_shift[y[va] == c] == c)) for c in range(3)],
    }
    print(f"normalisation experiment: correct={acc:.2%}  shifted 3 sigma={acc_shift:.2%}")
    with open(os.path.join(MODELS, "normalisation_experiment.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
