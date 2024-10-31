"""
Test functions for dann module.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

from adapt.feature_based import MDD

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
    model.add(Input(shape=input_shape))
    model.add(Dense(1, kernel_initializer="ones",
                    use_bias=False))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_discriminator(input_shape=(1,)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(10,
                    kernel_initializer=GlorotUniform(seed=0),
                    activation="relu"))
    model.add(Dense(1,
                    kernel_initializer=GlorotUniform(seed=0),
                    activation="sigmoid"))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_task(input_shape=(1,), output_shape=(1,)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(np.prod(output_shape),
                    use_bias=False,
                    kernel_initializer=GlorotUniform(seed=0)))
    model.compile(loss="mse", optimizer=Adam(0.1))
    return model


def test_fit():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = MDD(_get_encoder(), _get_task(), lambda_=1.,
                loss="mse", optimizer=Adam(0.01), metrics=["mse"])
    model.fit(Xs, ys, Xt, yt,
              epochs=100, batch_size=34, verbose=0)
    assert isinstance(model, Model)
    assert np.abs(model.encoder_.get_weights()[0][1][0] /
                  model.encoder_.get_weights()[0][0][0]) < 0.3
    assert np.sum(np.abs(model.predict(Xs).ravel() - ys)) < 0.1
    assert np.sum(np.abs(model.predict(Xt).ravel() - yt)) < 7.
    
    
def test_not_same_weights():
    tf.random.set_seed(0)
    np.random.seed(0)
    task = _get_task()
    encoder = _get_encoder()
    X_enc = encoder.predict(Xs)
    task.predict(X_enc)
    model = MDD(encoder, task, copy=False,
                loss="mse", optimizer=Adam(0.01), metrics=["mse"])
    model.fit(Xs, ys, Xt, yt,
              epochs=0, batch_size=34, verbose=0)
    assert np.any(model.task_.get_weights()[0] !=
                  model.discriminator_.get_weights()[0])
    assert np.all(model.task_.get_weights()[0] ==
                  task.get_weights()[0])
    
    
def test_cce():
    tf.random.set_seed(0)
    np.random.seed(0)
    task = _get_task(output_shape=(2,))
    encoder = _get_encoder()
    ys_2 = np.zeros((len(Xs), 2))
    ys_2[Xs[:, 0]<0.5, 0] = 1
    ys_2[Xs[:, 0]>=0.5, 1] = 1
    yt_2 = np.zeros((len(Xt), 2))
    yt_2[Xt[:, 0]<0.5, 0] = 1
    yt_2[Xt[:, 0]>=0.5, 1] = 1
    model = MDD(encoder, task, copy=False,
                loss="categorical_crossentropy", optimizer=Adam(0.01), metrics=["acc"])
    model.fit(Xs, ys_2, Xt, yt_2,
              epochs=10, batch_size=34, verbose=0)
