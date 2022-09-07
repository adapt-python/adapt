"""
Test functions for coral module.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import linalg
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform

from adapt.feature_based import CORAL, DeepCORAL


np.random.seed(0)
Xs = np.random.multivariate_normal(
     np.array([0, 0]),
     np.array([[0.001, 0], [0, 1]]),
     1000)
Xt = np.random.multivariate_normal(
     np.array([0, 0]),
     np.array([[0.1, 0.2], [0.2, 0.5]]),
     1000)
ys = np.zeros(1000)
yt = np.zeros(1000)

ys[Xs[:, 1]>0] = 1
yt[(Xt[:, 1]-0.5*Xt[:, 0])>0] = 1


def _get_encoder(input_shape=Xs.shape[1:]):
    model = Sequential()
    model.add(Dense(2, input_shape=input_shape,
                    kernel_initializer=GlorotUniform(seed=0),
                    use_bias=False))
    model.compile(loss="mse", optimizer="adam")
    return model


def _get_task(input_shape=(2,), output_shape=(1,)):
    model = Sequential()
    model.add(Dense(np.prod(output_shape),
                    kernel_initializer=GlorotUniform(seed=0),
                    input_shape=input_shape,
                    use_bias=False,
                    activation="sigmoid"))
    model.compile(loss="mse", optimizer="adam")
    return model


def test_setup():
    model = LogisticRegression()
    model.fit(Xs, ys)
    assert model.coef_[0][0] < 0.1 * model.coef_[0][1]
    assert (model.predict(Xs) == ys).sum() / len(Xs) >= 0.99
    assert (model.predict(Xt) == yt).sum() / len(Xt) < 0.97


def test_fit_coral():
    np.random.seed(0)
    model = CORAL(LogisticRegression(), lambda_=0.)
    model.fit(Xs, ys, Xt=Xt)
    assert isinstance(model.estimator_, LogisticRegression)
    assert len(model.estimator_.coef_[0]) == Xs.shape[1]
    assert (model.predict(Xt) == yt).sum() / len(Xt) >= 0.99
    
    
def test_fit_coral_complex():
    np.random.seed(0)
    model = CORAL(LogisticRegression(), lambda_=0.)
    Xs_ = np.random.randn(10, 100)
    Xt_ = np.random.randn(10, 100)
    model.fit(Xs_, ys[:10], Xt=Xt_)
    assert np.iscomplexobj(linalg.inv(linalg.sqrtm(model.Cs_)))
    assert np.iscomplexobj(linalg.sqrtm(model.Ct_))
    model.predict(Xs_, domain="src")


def test_fit_deepcoral():
    tf.random.set_seed(0)
    np.random.seed(0)
    model = DeepCORAL(_get_encoder(), _get_task(), metrics=["mse"])
    model.fit(Xs, ys, Xt,
              epochs=100, batch_size=64, verbose=0)
    assert isinstance(model.encoder_, Model)
    assert isinstance(model.task_, Model)
    assert len(model.encoder_.get_weights()[0]) == Xs.shape[1]
    assert np.abs(np.cov(Xs, rowvar=False) -
            np.cov(Xt, rowvar=False)).sum() > 0.5
    assert np.abs(np.cov(model.encoder_.predict(Xs), rowvar=False) -
            np.cov(model.encoder_.predict(Xt), rowvar=False)).sum() < 0.3
    assert (np.abs(model.predict(Xt) - yt) < 0.5).sum() / len(Xt) >= 0.99