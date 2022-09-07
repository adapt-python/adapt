"""
Test functions for dann module.
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from adapt.feature_based import DANN
from adapt.utils import UpdateLambda
from tensorflow.keras.initializers import GlorotUniform

Xs = np.concatenate((
    np.linspace(0, 1, 100).reshape(-1, 1),
    np.zeros((100, 1))
    ), axis=1)
Xt = np.concatenate((
    np.linspace(0, 1, 100).reshape(-1, 1),
    np.ones((100, 1))
    ), axis=1)
ys = 0.2 * Xs[:, 0].reshape(-1, 1)
yt = 0.2 * Xt[:, 0].reshape(-1, 1)


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
                    kernel_initializer=GlorotUniform(seed=0),
                    activation="elu"))
    model.add(Dense(1,
                    kernel_initializer=GlorotUniform(seed=0),
                    activation="sigmoid"))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_task(input_shape=(1,), output_shape=(1,)):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    kernel_initializer=GlorotUniform(seed=0),
                    use_bias=False,
                    input_shape=input_shape))
    model.compile(loss="mse", optimizer=Adam(0.1))
    return model


def test_fit_lambda_zero():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = DANN(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=0, loss="mse", optimizer=Adam(0.01), metrics=["mae"],
                random_state=0)
    model.fit(Xs, ys, Xt=Xt, yt=yt,
              epochs=200, batch_size=32, verbose=0)
    assert isinstance(model, Model)
    assert model.encoder_.get_weights()[0][1][0] == 1.0
    assert np.sum(np.abs(model.predict(Xs) - ys)) < 0.01
    assert np.sum(np.abs(model.predict(Xt) - yt)) > 10


def test_fit_lambda_one():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = DANN(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=1, loss="mse", optimizer=Adam(0.01), random_state=0)
    model.fit(Xs, ys, Xt, yt,
              epochs=100, batch_size=32, verbose=0)
    assert isinstance(model, Model)
    assert np.abs(model.encoder_.get_weights()[0][1][0] / 
            model.encoder_.get_weights()[0][0][0]) < 0.15
    assert np.sum(np.abs(model.predict(Xs) - ys)) < 1
    assert np.sum(np.abs(model.predict(Xt) - yt)) < 2
    
    
def test_fit_lambda_update():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = DANN(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=tf.Variable(0.), loss="mse", optimizer=Adam(0.01), random_state=0)
    model.fit(Xs, ys, Xt=Xt, yt=yt,
              epochs=100, batch_size=32, verbose=0,
              callbacks=UpdateLambda(max_steps=400, gamma=10.))
    assert isinstance(model, Model)
    assert np.abs(model.encoder_.get_weights()[0][1][0] / 
            model.encoder_.get_weights()[0][0][0]) < 0.2
    assert np.sum(np.abs(model.predict(Xs) - ys)) < 1
    assert np.sum(np.abs(model.predict(Xt) - yt)) < 5
    assert model.lambda_.numpy() == 1
    
    
def test_optimizer_enc_disc():
    tf.random.set_seed(0)
    np.random.seed(0)
    encoder = _get_encoder()
    task = _get_task()
    disc = _get_discriminator()
    X_enc = encoder.predict(Xs)
    task.predict(X_enc)
    disc.predict(X_enc)
    model = DANN(encoder, task, disc, copy=True,
                 optimizer_enc=Adam(0.0), optimizer_disc=Adam(0.001),
                 lambda_=tf.Variable(0.), loss="mse", optimizer=Adam(0.01), random_state=0)
    model.fit(Xs, ys, Xt=Xt, yt=yt,
              epochs=10, batch_size=32, verbose=0)
    assert np.all(model.encoder_.get_weights()[0] == encoder.get_weights()[0])
    assert np.any(model.task_.get_weights()[0] != task.get_weights()[0])
    assert np.any(model.discriminator_.get_weights()[0] != disc.get_weights()[0])
    
    
def test_warnings():
    with pytest.warns() as record:
        model = DANN(gamma=10.)
    assert len(record) == 1