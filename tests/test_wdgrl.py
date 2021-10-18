"""
Test functions for dann module.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from adapt.feature_based import WDGRL
from adapt.feature_based._wdgrl import _Interpolation

Xs = np.concatenate((
    np.linspace(0, 1, 100).reshape(-1, 1),
    np.zeros((100, 1))
    ), axis=1)
Xt = np.concatenate((
    np.linspace(0, 1, 100).reshape(-1, 1),
    np.ones((100, 1))
    ), axis=1)
ys = 0.2 * Xs[:, 0].ravel()
yt = 0.2 * Xt[:, 0].ravel()


def _get_encoder(input_shape=Xs.shape[1:]):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape,
                    kernel_initializer="ones",
                    use_bias=False))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_discriminator(input_shape=(1,)):
    model = Sequential()
    model.add(Dense(10,
                    input_shape=input_shape,
                    activation="relu"))
    model.add(Dense(1,
                    activation=None))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_task(input_shape=(1,), output_shape=(1,)):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    use_bias=False,
                    input_shape=input_shape))
    model.compile(loss="mse", optimizer=Adam(0.1))
    return model


def test_interpolation():
    np.random.seed(0)
    tf.random.set_seed(0)

    zeros = tf.identity(np.zeros((3, 1), dtype=np.float32))
    ones= tf.identity(np.ones((3, 1), dtype=np.float32))

    inter, dist = _Interpolation().call([zeros, ones])
    assert np.all(np.round(dist, 3) == np.round(inter, 3))
    assert np.all(inter >= zeros)
    assert np.all(inter <= ones)


def test_fit_lambda_zero():
    tf.random.set_seed(1)
    np.random.seed(1)
    model = WDGRL(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=0, loss="mse", optimizer=Adam(0.01), metrics=["mse"],
                 random_state=0)
    model.fit(Xs, ys, Xt, yt,
              epochs=300, verbose=0)
    assert isinstance(model.model_, Model)
    assert model.encoder_.get_weights()[0][1][0] == 1.0
    assert np.sum(np.abs(model.predict(Xs).ravel() - ys)) < 0.01
    assert np.sum(np.abs(model.predict(Xt).ravel() - yt)) > 10


def test_fit_lambda_one():
    tf.random.set_seed(1)
    np.random.seed(1)
    model = WDGRL(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=1, gamma=0, loss="mse", optimizer=Adam(0.01),
                  metrics=["mse"], random_state=0)
    model.fit(Xs, ys, Xt, yt,
              epochs=300, verbose=0)
    assert isinstance(model.model_, Model)
    assert np.abs(model.encoder_.get_weights()[0][1][0] / 
            model.encoder_.get_weights()[0][0][0]) < 0.05
    assert np.sum(np.abs(model.predict(Xs).ravel() - ys)) < 2
    assert np.sum(np.abs(model.predict(Xt).ravel() - yt)) < 2
