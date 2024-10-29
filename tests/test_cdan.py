"""
Test functions for cdan module.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

from adapt.feature_based import CDAN

from adapt.utils import make_classification_da

Xs, ys, Xt, yt = make_classification_da()
yss = np.zeros((len(ys), 2))
yss[ys==0, 0] = 1
yss[ys==1, 1] = 1

ytt = np.zeros((len(yt), 2))
ytt[yt==0, 0] = 1
ytt[yt==1, 1] = 1

def _entropy(x):
    return -np.sum(x * np.log(x), 1)

def _get_encoder(input_shape=Xs.shape[1:], units=10):
    model = Sequential()
    model.add(Dense(units, input_shape=input_shape,
                    kernel_initializer=GlorotUniform(seed=0),))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_discriminator(input_shape=(10*2,)):
    model = Sequential()
    model.add(Dense(10,
                    input_shape=input_shape,
                    kernel_initializer=GlorotUniform(seed=0),
                    activation="relu"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer=GlorotUniform(seed=0)))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_task(input_shape=(10,)):
    model = Sequential()
    model.add(Dense(2,
                    kernel_initializer=GlorotUniform(seed=0),
                    input_shape=input_shape,
                    activation="softmax"))
    model.compile(loss="mse", optimizer=Adam(0.1))
    return model


def test_fit_lambda_zero():
    tf.random.set_seed(1)
    np.random.seed(1)
    model = CDAN(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=0, loss="categorical_crossentropy",
                 optimizer=Adam(0.001), metrics=["acc"],
                 random_state=0, validation_data=(Xt, ytt))
    model.fit(Xs, yss, Xt, ytt,
              epochs=300, verbose=0)
    assert model.history_['accuracy'][-1] > 0.9
    assert model.history_['val_accuracy'][-1] < 0.9


def test_fit_lambda_one_no_entropy():
    tf.random.set_seed(1)
    np.random.seed(1)
    model = CDAN(_get_encoder(), _get_task(), _get_discriminator(),
                 lambda_=1., entropy=False, loss="categorical_crossentropy",
                 optimizer=Adam(0.001), metrics=["acc"],
                 random_state=0, validation_data=(Xt, ytt))
    model.fit(Xs, yss, Xt, ytt,
              epochs=300, verbose=0)
    assert model.history_['accuracy'][-1] > 0.8
    assert model.history_['val_accuracy'][-1] > 0.8
    
    
def test_fit_lambda_entropy():
    tf.random.set_seed(1)
    np.random.seed(1)
    encoder = _get_encoder()
    encoder.trainable = False
    model = CDAN(encoder, _get_task(), _get_discriminator(),
                 lambda_=1., entropy=True, loss="categorical_crossentropy",
                 optimizer=Adam(0.01), metrics=["acc"],
                 random_state=0)
    model.fit(Xs, yss, Xt, ytt,
              epochs=40, verbose=0)
    
    ys_disc = model.predict_disc(Xs).ravel()
    ys_ent = _entropy(model.predict(Xs))
    yt_disc = model.predict_disc(Xt).ravel()
    yt_ent = _entropy(model.predict(Xt))
    assert np.corrcoef(yt_ent, yt_disc)[0, 1] > 0.
    assert np.corrcoef(ys_ent, ys_disc)[0, 1] < 0.
    
    
def test_fit_max_features():
    tf.random.set_seed(1)
    np.random.seed(1)
    model = CDAN(_get_encoder(), _get_task(), _get_discriminator((10,)), max_features=10,
                 lambda_=0., entropy=False, loss="categorical_crossentropy",
                 optimizer=Adam(0.01), metrics=["acc"],
                 random_state=0)
    model.fit(Xs, yss, Xt, ytt,
              epochs=30, verbose=0)
    assert model._random_task.shape == (2, 10)
    assert model._random_enc.shape == (10, 10)
    assert model.predict_disc(Xt).mean() < 0.5
    assert model.predict_disc(Xs).mean() > 0.5