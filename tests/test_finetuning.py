import numpy as np
import tensorflow as tf

from adapt.utils import make_classification_da
from adapt.parameter_based import FineTuning

np.random.seed(0)
tf.random.set_seed(0)

encoder = tf.keras.Sequential()
encoder.add(tf.keras.layers.Dense(50, activation="relu"))
encoder.add(tf.keras.layers.Dense(50, activation="relu"))

task = tf.keras.Sequential()
task.add(tf.keras.layers.Dense(1, activation="sigmoid"))

ind = np.random.choice(100, 10)
Xs, ys, Xt, yt = make_classification_da()


def test_finetune():
    model = FineTuning(encoder=encoder, task=task, loss="bce", optimizer="adam", random_state=0)
    model.fit(Xs, ys, epochs=100, verbose=0)

    assert np.mean((model.predict(Xt).ravel()>0.5) == yt) < 0.7

    fine_tuned = FineTuning(encoder=model.encoder_, task=model.task_,
                            training=False,
                            loss="bce", optimizer="adam", random_state=0)
    fine_tuned.fit(Xt[ind], yt[ind], epochs=100, verbose=0)

    assert np.abs(fine_tuned.encoder_.get_weights()[0] - model.encoder_.get_weights()[0]).sum() == 0.
    assert np.mean((fine_tuned.predict(Xt).ravel()>0.5) == yt) > 0.7
    assert np.mean((fine_tuned.predict(Xt).ravel()>0.5) == yt) < 0.8

    fine_tuned = FineTuning(encoder=model.encoder_, task=model.task_,
                            training=True,
                            loss="bce", optimizer="adam", random_state=0)
    fine_tuned.fit(Xt[ind], yt[ind], epochs=100, verbose=0)

    assert np.abs(fine_tuned.encoder_.get_weights()[0] - model.encoder_.get_weights()[0]).sum() > 1.
    assert np.mean((fine_tuned.predict(Xt).ravel()>0.5) == yt) > 0.9

    fine_tuned = FineTuning(encoder=model.encoder_, task=model.task_,
                            training=[True, False],
                            loss="bce", optimizer="adam", random_state=0)
    fine_tuned.fit(Xt[ind], yt[ind], epochs=100, verbose=0)

    assert np.abs(fine_tuned.encoder_.get_weights()[0] - model.encoder_.get_weights()[0]).sum() == 0.
    assert np.abs(fine_tuned.encoder_.get_weights()[-1] - model.encoder_.get_weights()[-1]).sum() > 1.

    fine_tuned = FineTuning(encoder=model.encoder_, task=model.task_,
                            training=[False],
                            loss="bce", optimizer="adam", random_state=0)
    fine_tuned.fit(Xt[ind], yt[ind], epochs=100, verbose=0)

    assert np.abs(fine_tuned.encoder_.get_weights()[0] - model.encoder_.get_weights()[0]).sum() == 0.
    assert np.abs(fine_tuned.encoder_.get_weights()[-1] - model.encoder_.get_weights()[-1]).sum() == 0
