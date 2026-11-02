"""M3 - 35% structured pruning (tfmot PolynomialDecay schedule) + full INT8 PTQ.

Structured = whole hidden neurons (the MLP equivalent of conv filters). At each pruning step
the tfmot PolynomialDecay schedule gives the target sparsity; the neurons with the lowest
L2 kernel norm are masked (kernel column + bias). tfmot's own block_size pruning cannot do
this: a block spanning a full kernel column gets squeezed to rank 1 and crashes
(pruning_impl._maybe_update_block_mask), hence the small callback below.
After fine-tuning, the zeroed neurons are physically removed, then INT8-quantised.

Output: training/models/m3_pruned_int8.tflite (+ m3_pruned_fp32.keras)
"""
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_pipeline"))
sys.path.insert(0, os.path.join(ROOT, "training"))
from convert_ptq import MODELS, calibration_set, convert_int8  # noqa: E402
from preprocessing import load_stats, normalise  # noqa: E402

SPARSITY = 0.35
EPOCHS = 60
BATCH = 24


class NeuronPruning(tf.keras.callbacks.Callback):
    def __init__(self, layers, schedule):
        super().__init__()
        self.layers_, self.schedule, self.step = layers, schedule, 0
        self.masks = [np.ones(l.units, bool) for l in layers]

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        should_prune, sparsity = self.schedule(self.step)
        for i, layer in enumerate(self.layers_):
            W, b = layer.get_weights()
            if should_prune:
                n = int(round(float(sparsity) * layer.units))
                self.masks[i][np.argsort(np.linalg.norm(W, axis=0) * self.masks[i])[:n]] = False
            layer.set_weights([W * self.masks[i], b * self.masks[i]])


def remove_dead_neurons(model):
    (W1, b1), (W2, b2), (W3, b3) = [l.get_weights() for l in model.layers]
    k1 = np.any(W1 != 0, axis=0) | (b1 != 0)
    k2 = np.any(W2 != 0, axis=0) | (b2 != 0)
    small = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(int(k1.sum()), activation="relu"),
        tf.keras.layers.Dense(int(k2.sum()), activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    small.set_weights([W1[:, k1], b1[k1], W2[k1][:, k2], b2[k2], W3[k2], b3])
    return small


def main():
    tf.keras.utils.set_random_seed(42)
    d = np.load(os.path.join(ROOT, "training", "data", "split.npz"))
    stats = load_stats()
    Xtr, Xva = normalise(d["X_train"], stats), normalise(d["X_val"], stats)

    model = tf.keras.models.load_model(os.path.join(MODELS, "m1_fp32.keras"))
    steps = EPOCHS * int(np.ceil(len(Xtr) / BATCH))
    # reach 35% at 2/3 of training, fine-tune the remaining third with the mask held
    schedule = tfmot.sparsity.keras.PolynomialDecay(0.0, SPARSITY, begin_step=0, end_step=steps * 2 // 3, frequency=10)
    pruner = NeuronPruning(model.layers[:2], schedule)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, d["y_train"], epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[pruner])

    small = remove_dead_neurons(model)
    assert np.allclose(small.predict(Xva, verbose=0), model.predict(Xva, verbose=0), atol=1e-5)
    pruned = sum((~m).sum() for m in pruner.masks)
    print(f"hidden neurons 32/16 -> {small.layers[0].units}/{small.layers[1].units} ({pruned}/48 = {pruned / 48:.1%} pruned)")
    acc = np.mean(small.predict(Xva, verbose=0).argmax(1) == d["y_val"])
    print(f"pruned FP32 validation accuracy = {acc:.2%}")

    small.save(os.path.join(MODELS, "m3_pruned_fp32.keras"))
    convert_int8(small, calibration_set(), os.path.join(MODELS, "m3_pruned_int8.tflite"))


if __name__ == "__main__":
    main()
