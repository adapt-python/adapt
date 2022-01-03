"""
Test base
"""

import shutil
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import clone

from adapt.base import BaseAdaptEstimator, BaseAdaptDeep

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


def test_base_adapt_estimator():
    base_adapt = BaseAdaptEstimator(Xt=Xt)
    for check in check_estimator(base_adapt, generate_only=True):
        try:
            check[1](base_adapt)
        except Exception as e:
            if "The Adapt model should implement a transform or predict_weights methods" in str(e):
                print(str(e))
            else:
                raise
                
                
def test_base_adapt_deep():
    model = BaseAdaptDeep(Xt=Xt, loss="mse", optimizer=Adam(), learning_rate=0.1)
    model.fit(Xs, ys)
    model.predict(Xt)
    model.score(Xt, yt)
    model.transform(Xs)
    model.predict_task(np.random.randn(10, 10))
    model.predict_disc(np.random.randn(10, 10))
    
    assert isinstance(model.opimizer, Adam)
    assert model.optimizer.learning_rate.numpy() == 0.1
    assert hasattr(model, "encoder_")
    assert hasattr(model, "task_")
    assert hasattr(model, "discriminator_")
    
    
def test_base_adapt_deep():
    model = BaseAdaptDeep(Xt=Xt, loss="mse",
                          epochs=2,
                          optimizer=Adam(),
                          learning_rate=0.1,
                          random_state=0)
    model.fit(Xs, ys)
    yp = model.predict(Xt)
    score = model.score(Xt, yt)
    X_enc = model.transform(Xs)
    ypt = model.predict_task(Xt)
    ypd = model.predict_disc(Xt)
    
    new_model = clone(model)
    new_model.fit(Xs, ys)
    yp2 = new_model.predict(Xt)
    score2 = new_model.score(Xt, yt)
    X_enc2 = new_model.transform(Xs)
    ypt2 = new_model.predict_task(Xt)
    ypd2 = new_model.predict_disc(Xt)
    
    model.save("model.tf", save_format="tf")
    new_model = tf.keras.models.load_model("model.tf")
    shutil.rmtree("model.tf")
    yp3 = new_model.predict(Xt)
    
    assert isinstance(model.optimizer, Adam)
    assert np.abs(model.optimizer.learning_rate.numpy() - 0.1) < 1e-6
    assert hasattr(model, "encoder_")
    assert hasattr(model, "task_")
    assert hasattr(model, "discriminator_")
    
    assert np.all(yp == yp2)
    assert score == score2
    assert np.all(ypt == ypt2)
    assert np.all(ypd == ypd2)
    assert np.all(X_enc == X_enc2)
    assert np.mean(np.abs(yp - yp3)) < 1e-6