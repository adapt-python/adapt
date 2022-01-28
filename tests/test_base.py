"""
Test base
"""

import copy
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


class DummyFeatureBased(BaseAdaptEstimator):
    
    def fit_transform(self, Xs, **kwargs):
        return Xs
    
    def transform(self, Xs):
        return Xs
    
    
class DummyInstanceBased(BaseAdaptEstimator):
    
    def fit_weights(self, Xs, **kwargs):
        return np.ones(len(Xs))
    
    def predict_weights(self):
        return np.ones(100)
    
    
class DummyParameterBased(BaseAdaptEstimator):
    
    def fit(self, Xs, ys):
        return self.fit_estimator(Xs, ys)


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
                

def test_base_adapt_score():
    model = DummyParameterBased(Xt=Xt, random_state=0)
    model.fit(Xs, ys)
    model.score(Xt, yt)
    
    model = DummyFeatureBased(Xt=Xt, random_state=0)
    model.fit(Xs, ys)
    s1 = model.score(Xt, yt)
    s2 = model._score_adapt(Xs, Xt)
    assert s1 == s2
    
    model = DummyInstanceBased(Xt=Xt, random_state=0)
    model.fit(Xs, ys)
    model.score(Xt, yt)
    s1 = model.score(Xt, yt)
    s2 = model._score_adapt(Xs, Xt)
    assert s1 == s2
    
    
def test_base_adapt_val_sample_size():
    model = DummyFeatureBased(Xt=Xt, random_state=0, val_sample_size=10)
    model.fit(Xs, ys)
    model.score(Xt, yt)
    assert len(model.Xs_) == 10
    assert len(model.Xt_) == 10
    assert np.all(model.Xs_ == Xs[model.src_index_])
    

def test_base_adapt_keras_estimator():
    est = Sequential()
    est.add(Dense(1, input_shape=Xs.shape[1:]))
    est.compile(loss="mse", optimizer=Adam(0.01))
    model = BaseAdaptEstimator(est, Xt=Xt)
    model.fit(Xs, ys)
    assert model.estimator_.loss == "mse"
    assert isinstance(model.estimator_.optimizer, Adam)
    assert model.estimator_.optimizer.learning_rate == 0.01
    
    model = BaseAdaptEstimator(est, Xt=Xt, loss="mae",
                               optimizer=Adam(0.01, beta_1=0.5),
                               learning_rate=0.1)
    model.fit(Xs, ys)
    assert model.estimator_.loss == "mae"
    assert isinstance(model.estimator_.optimizer, Adam)
    assert model.estimator_.optimizer.learning_rate == 0.1
    assert model.estimator_.optimizer.beta_1 == 0.5
    
    model = BaseAdaptEstimator(est, Xt=Xt, optimizer="sgd")
    model.fit(Xs, ys)
    assert not isinstance(model.estimator_.optimizer, Adam)
    
    est = Sequential()
    est.add(Dense(1, input_shape=Xs.shape[1:]))
    model = BaseAdaptEstimator(est, Xt=Xt, loss="mae",
                               optimizer=Adam(0.01, beta_1=0.5),
                               learning_rate=0.1)
    model.fit(Xs, ys)
    assert model.estimator_.loss == "mae"
    assert isinstance(model.estimator_.optimizer, Adam)
    assert model.estimator_.optimizer.learning_rate == 0.1
    assert model.estimator_.optimizer.beta_1 == 0.5
    
    s1 = model.score_estimator(Xt[:10], yt[:10])
    s2 = model.estimator_.evaluate(Xt[:10], yt[:10])
    assert s1 == s2
    
    copy_model = copy.deepcopy(model)
    assert s1 == copy_model.score_estimator(Xt[:10], yt[:10])
    assert hex(id(model)) != hex(id(copy_model))


def test_base_adapt_deep():
    model = BaseAdaptDeep(Xt=Xt, loss="mse",
                          epochs=2,
                          optimizer=Adam(),
                          learning_rate=0.1,
                          random_state=0)
    model.fit(Xs, ys)
    yp = model.predict(Xt)
    score = model.score(Xt, yt)
    score_est = model.score_estimator(Xt, yt)
    X_enc = model.transform(Xs)
    ypt = model.predict_task(Xt)
    ypd = model.predict_disc(Xt)
    
    new_model = clone(model)
    new_model.fit(Xs, ys)
    yp2 = new_model.predict(Xt)
    score2 = new_model.score(Xt, yt)
    score_est2 = new_model.score_estimator(Xt, yt)
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
    assert score_est == score_est2
    assert np.all(ypt == ypt2)
    assert np.all(ypd == ypd2)
    assert np.all(X_enc == X_enc2)
    assert np.mean(np.abs(yp - yp3)) < 1e-6
    
    
def test_base_deep_validation_data():
    model = BaseAdaptDeep(Xt=Xt)
    model.fit(Xs, ys, validation_data=(Xt, yt))
    model.fit(Xs, ys, validation_split=0.1)
    
    model = BaseAdaptDeep(Xt=Xt, yt=yt)
    model.fit(Xs, ys, validation_data=(Xt, yt))
    model.fit(Xs, ys, validation_split=0.1)
    
    
def test_base_deep_dataset():
    model = BaseAdaptDeep()
    model.fit(Xs, ys, Xt=Xt, validation_data=(Xs, ys))
    model.predict(Xs)
    model.evaluate(Xs, ys)
    
    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(Xs),
                                   tf.data.Dataset.from_tensor_slices(ys.reshape(-1,1))
                                  ))
    model = BaseAdaptDeep()
    model.fit(dataset, Xt=dataset, validation_data=dataset.batch(10))
    model.predict(tf.data.Dataset.from_tensor_slices(Xs).batch(32))
    model.evaluate(dataset.batch(32))
    
    def gens():
        for i in range(40):
            yield Xs[i], ys[i]
            
    dataset = tf.data.Dataset.from_generator(gens,
                                             output_shapes=([2], []),
                                             output_types=("float32", "float32"))
    model = BaseAdaptDeep()
    model.fit(dataset, Xt=Xt, validation_data=dataset.batch(10))
    model.predict(tf.data.Dataset.from_tensor_slices(Xs).batch(32))
    model.evaluate(dataset.batch(32))
    
    
def _unpack_data_ms(self, data):
    data_src = data[0]
    data_tgt = data[1]
    Xs = data_src[0][0]
    ys = data_src[1][0]
    if isinstance(data_tgt, tuple):
        Xt = data_tgt[0]
        yt = data_tgt[1]
        return Xs, Xt, ys, yt
    else:
        Xt = data_tgt
        return Xs, Xt, ys, None
    
    
def test_multisource():
    np.random.seed(0)
    model = BaseAdaptDeep()
    model._unpack_data = _unpack_data_ms.__get__(model)
    model.fit(Xs, ys, Xt=Xt, domains=np.random.choice(2, len(Xs)))
    model.predict(Xs)
    model.evaluate(Xs, ys)
    assert model.n_sources_ == 2