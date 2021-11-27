"""
Weighting Adversarial Neural Network (WANN)
"""
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, multiply
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import MaxNorm

from adapt.utils import (GradientHandler,
                         check_arrays,
                         check_one_array,
                         check_network,
                         get_default_task)
from adapt.feature_based import BaseDeepFeature


class StopTraining(Callback):
    
    def on_train_batch_end(self, batch, logs={}):
        if logs.get('loss') < 0.01:
            print("Weights initialization succeeded !")
            self.model.stop_training = True


class WANN(BaseDeepFeature):
    """
    WANN: Weighting Adversarial Neural Network is an instance-based domain adaptation
    method suited for regression tasks. It supposes the supervised setting where some
    labeled target data are available.
    
    The goal of WANN is to compute a source instances reweighting which correct
    "shifts" between source and target domain. This is done by minimizing the
    Y-discrepancy distance between source and target distributions
    
    WANN involves three networks:
        - the weighting network which learns the source weights.
        - the task network which learns the task.
        - the discrepancy network which is used to estimate a distance 
          between the reweighted source and target distributions: the Y-discrepancy
    
    Parameters
    ----------        
    task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        
    weighter : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as
        weighter network.
        
    C : float (default=1.)
        Clipping constant for the weighting networks
        regularization. Low value of ``C`` produce smoother
        weighting map. If ``C<=0``, No regularization is added.
        
    init_weights : bool (default=True)
        If True a pretraining of ``weighter`` is made such
        that all predicted weights start close to one.
        
    loss : string or tensorflow loss (default="mse")
        Loss function used for the task.
        
    metrics : dict or list of string or tensorflow metrics (default=None)
        Metrics given to the model. If a list is provided,
        metrics are used on both ``task`` and ``discriminator``
        outputs. To give seperated metrics, please provide a
        dict of metrics list with ``"task"`` and ``"disc"`` as keys.
        
    optimizer : string or tensorflow optimizer (default=None)
        Optimizer of the model. If ``None``, the
        optimizer is set to tf.keras.optimizers.Adam(0.001)
        
    copy : boolean (default=True)
        Whether to make a copy of ``encoder``, ``task`` and
        ``discriminator`` or not.
        
    random_state : int (default=None)
        Seed of random generator.
    """
    
    def __init__(self,
                 task=None,
                 weighter=None,
                 C=1.,
                 init_weights=True,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 copy=True,
                 random_state=None):
        
        super().__init__(weighter, task, None,
                         loss, metrics, optimizer, copy,
                         random_state)
        
        self.init_weights = init_weights
        self.init_weights_ = init_weights
        self.C = C
        
        if weighter is None:
            self.weighter_ = get_default_task() #activation="relu"
        else:
            self.weighter_ = self.encoder_
        
        if self.C > 0.:
            self._add_regularization()
            
        self.discriminator_ = check_network(self.task_, 
                                            copy=True,
                                            display_name="task",
                                            force_copy=True)
        self.discriminator_._name = self.discriminator_._name + "_2"


    def _add_regularization(self):
        for layer in self.weighter_.layers:
            if hasattr(self.weighter_, "kernel_constraint"):
                self.weighter_.kernel_constraint = MaxNorm(self.C)
            if hasattr(self.weighter_, "bias_constraint"):
                self.weighter_.bias_constraint = MaxNorm(self.C)
        
    
    def fit(self, Xs, ys, Xt, yt, **fit_params):
        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)
        
        if self.init_weights_:
            self._init_weighter(Xs)
            self.init_weights_ = False
        self._fit(Xs, ys, Xt, yt, **fit_params)
        return self
    
    
    def _init_weighter(self, Xs):
        self.weighter_.compile(optimizer=deepcopy(self.optimizer), loss="mse")
        batch_size = 64
        epochs = max(1, int(64*1000/len(Xs)))
        callback = StopTraining()
        self.weighter_.fit(Xs, np.ones(len(Xs)),
                          epochs=epochs, batch_size=batch_size,
                          callbacks=[callback], verbose=0)
        
    
    def _initialize_networks(self, shape_Xt):
        self.weighter_.predict(np.zeros((1,) + shape_Xt));
        self.task_.predict(np.zeros((1,) + shape_Xt));
        self.discriminator_.predict(np.zeros((1,) + shape_Xt));

            
    def create_model(self, inputs_Xs, inputs_Xt):
        
        Flip = GradientHandler(-1.)
        
        # Get networks output for both source and target
        weights_s = self.weighter_(inputs_Xs)
        weights_s = tf.math.abs(weights_s)
        task_s = self.task_(inputs_Xs)
        task_t = self.task_(inputs_Xt)
        disc_s = self.discriminator_(inputs_Xs)
        disc_t = self.discriminator_(inputs_Xt)
        
        # Reversal layer at the end of discriminator
        disc_s = Flip(disc_s)
        disc_t = Flip(disc_t)

        return dict(task_s=task_s, task_t=task_t,
                    disc_s=disc_s, disc_t=disc_t,
                    weights_s=weights_s)
            
    
    def get_loss(self, inputs_ys, inputs_yt, task_s,
                 task_t, disc_s, disc_t, weights_s):
        
        loss_task_s = self.loss_(inputs_ys, task_s)
        loss_task_s = multiply([weights_s, loss_task_s])
        
        loss_disc_s = self.loss_(inputs_ys, disc_s)
        loss_disc_s = multiply([weights_s, loss_disc_s])
        
        loss_disc_t = self.loss_(inputs_yt, disc_t)
        
        loss_disc = (tf.reduce_mean(loss_disc_t) - 
                     tf.reduce_mean(loss_disc_s))
                         
        loss = tf.reduce_mean(loss_task_s) + loss_disc
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt, task_s,
                 task_t, disc_s, disc_t, weights_s):
        
        metrics = {}
        
        loss_s = self.loss_(inputs_ys, task_s)        
        loss_t = self.loss_(inputs_yt, task_t) 
        
        metrics["task_s"] = tf.reduce_mean(loss_s)
        metrics["task_t"] = tf.reduce_mean(loss_t)
        
        names_task, names_disc = self._get_metric_names()
        
        for metric, name in zip(self.metrics_task_, names_task):
            metrics[name + "_s"] = metric(inputs_ys, task_s)
            metrics[name + "_t"] = metric(inputs_yt, task_t)
        return metrics
    
    
    def predict(self, X):
        """
        Predict method: return the prediction of task network
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of task network
        """
        X = check_one_array(X)
        return self.task_.predict(X)
    
    
    def predict_weights(self, X):
        """
        Return the predictions of weighting network
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        array:
            weights
        """
        return np.abs(self.weighter_.predict(X))
    
    
    def predict_disc(self, X):
        """
        Return predictions of the discriminator.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_disc : array
            predictions of discriminator network
        """
        X = check_one_array(X)
        return self.discriminator_.predict(X)