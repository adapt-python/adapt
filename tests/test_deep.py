"""
Test functions for BaseDeepFeature object
"""


import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

from adapt.feature_based._deep import accuracy, UpdateLambda, BaseDeepFeature
from adapt.utils import (GradientHandler,
                         get_default_encoder,
                         get_default_task,
                         get_default_discriminator,
                         make_regression_da)


class _RecordLambda(Callback):
    
    def __init__(self):
        self.lambdas = []
    
    def on_train_batch_end(self, batch, logs=None):
        self.lambdas.append(
            self.model.get_layer("gh").lambda_.numpy())
        
        
class CustomDeep(BaseDeepFeature):
    
    def create_model(self, inputs_Xs, inputs_Xt):
        encoded_s = self.encoder_(inputs_Xs)
        encoded_t = self.encoder_(inputs_Xt)
        task_s = self.task_(encoded_s)
        task_t = self.task_(encoded_t)
        disc_s = self.discriminator_(encoded_s)
        disc_t = self.discriminator_(encoded_t)
        return dict(task_s=task_s, task_t=task_t,
                    disc_s=disc_s, disc_t=disc_t)
    
    def get_loss(self, inputs_ys,
                 task_s, task_t,
                 disc_s, disc_t):
        
        loss = self.loss_(inputs_ys, task_s)
        return loss


def test_accuracy():
    y_true = tf.Variable([[0, 1, 0],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]],
            dtype="float32")
    y_pred = tf.Variable([[0.5, 0.3, 0.2],
             [0.9, 0.1, 0.],
             [0.6, 0.1, 0.3],
             [0.1, 0.7, 0.2]],
            dtype="float32")
    acc = accuracy(y_true, y_pred)
    assert np.all(np.array([0, 1, 1, 0]) == acc.numpy())
    
    y_true = tf.Variable([[0], [1], [0]],
                         dtype="float32")
    y_pred = tf.Variable([[0.6], [0.3], [0.2]],
                         dtype="float32")
    acc = accuracy(y_true, y_pred)
    assert np.all(np.array([0, 0, 1]) == acc.numpy())

       
def test_update_lambda():
    model = Sequential()
    model.add(Dense(1, use_bias=False,
                    kernel_initializer="zeros"))
    model.add(GradientHandler(lambda_init=0., name="gh"))
    model.compile(loss="mse", optimizer="sgd")
    update_lambda = UpdateLambda(lambda_name="gh", gamma=1.)
    record_lambda = _RecordLambda()
    X = np.ones((10, 1)); y = np.ones((10, 1));
    model.fit(X, y, epochs=3, batch_size=2, verbose=0,
              callbacks=[update_lambda, record_lambda])
    
    x = np.array([float(i) for i in range(15)])
    x /= 15.
    z = -(2 / (1 + np.exp(-1 * x)) - 1)
    
    assert update_lambda.total_steps == 15
    assert update_lambda.steps == 15
    assert np.abs(model.layers[1].lambda_.numpy() - z[-1]) < 1e-5
    assert np.abs(np.array(record_lambda.lambdas) - z).mean() < 1e-5
    
    
def test_basedeep():
    model = CustomDeep()
    assert isinstance(model.optimizer, Adam)
    assert model.loss_  == mean_squared_error
    assert model.metrics_task_ == []
    assert model.metrics_disc_ == []
    assert model.copy
    
    Xs, ys, Xt, yt = make_regression_da()
    model._fit(Xs, ys, Xt, yt, epochs=1, verbose=0,
               batch_size=100)
    
    