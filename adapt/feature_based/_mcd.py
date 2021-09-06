"""
DANN
"""

import warnings
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Input, subtract
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from adapt.utils import (GradientHandler,
                         check_arrays,
                         check_network)
from adapt.feature_based import BaseDeepFeature

EPS = K.epsilon()


class MCD(BaseDeepFeature):
    """
    MCD: Maximum Classifier Discrepancy is a feature-based domain adaptation
    method originally introduced for unsupervised classification DA.
    
    The goal of MCD is to find a new representation of the input features which
    minimizes the discrepancy between the source and target domains 
    
    The discrepancy is estimated through adversarial training of three networks:
    An encoder and two classifiers. These two learn the task on the source domains
    and are used to compute the discrepancy. A reversal layer is placed between
    the encoder and the two classifiers to perform adversarial training.
    
    Parameters
    ----------
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
        
    get_task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        
    is_pretrained : boolean (default=False)
        Specify if the `encoder` and `task` networks are already
        pretrained on source or not.

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
        
    optimizer_src : string or tensorflow optimizer (default=None)
        Optimizer for the pretraining on source. If ``None``,
        ``optimizer_src`` is a copy of ``optimizer``.
        
    copy : boolean (default=True)
        Whether to make a copy of ``encoder``, ``task`` and
        ``discriminator`` or not.
        
    random_state : int (default=None)
        Seed of random generator.
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        Principal task network.
        
    task_sec_ : tensorflow Model
        Secondary task network.
    
    model_ : tensorflow Model
        Fitted model: the union of ``encoder_``, ``task_``,
        ``task_sec_`` and ``discriminator_`` networks.
        
    model_src_ : tensorflow Model
        Fitted model: the union of ``encoder_``, ``task_``
        and ``task_sec_`` networks.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    history_src_ : dict
        Source model history of the losses and metrics
        across the epochs. If ``yt`` is given in ``fit``
        method, target metrics and losses are recorded too.
        
    is_pretrained_ : boolean
        Specify if the encoder is already pretrained on
        source or not. If True, the ``fit`` method will
        avoid the pretraining step.
        
    Examples
    --------
    >>> Xs = np.concatenate((np.random.random((100, 1)),
    ...                      np.zeros((100, 1))), 1)
    >>> Xt = np.concatenate((np.random.random((100, 1)),
    ...                      np.ones((100, 1))), 1)
    >>> ys = 0.2 * Xs[:, 0]
    >>> yt = 0.2 * Xt[:, 0]
    >>> model = MCD(random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_src_["task_t"][-1]
    0.0234...
    >>> model.history_["task_t"][-1]
    0.0009...
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1702.05464.pdf>`_ K. Saito, K. Watanabe, \
Y. Ushiku, and T. Harada. "Maximum  classifier  discrepancy  for  unsupervised  \
domain adaptation". In CVPR, 2018.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 is_pretrained=False,
                 lambda_=1.,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 optimizer_src=None,
                 copy=True,
                 random_state=None):

        super().__init__(encoder, task, None,
                         loss, metrics, optimizer, copy,
                         random_state)
        self.is_pretrained_ = is_pretrained
        self.lambda_ = lambda_
        
        if optimizer_src is None:
            self.optimizer_src = deepcopy(self.optimizer)
        else:
            self.optimizer_src = optimizer_src
            
        self.task_sec_ = check_network(self.task_, 
                                       copy=True,
                                       display_name="task",
                                       force_copy=True)
        self.task_sec_._name = self.task_sec_._name + "_2"

        
    def fit_source(self, Xs, ys, Xt=None, yt=None, **fit_params):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
                
        # Call predict to avoid strange behaviour with
        # Sequential model whith unspecified input_shape
        zeros_enc_ = self.encoder_.predict(np.zeros((1,) + Xs.shape[1:]));
        self.task_.predict(zeros_enc_);
        
        self.model_src_ = Sequential()
        self.model_src_.add(self.encoder_)
        self.model_src_.add(self.task_)
        self.model_src_.compile(loss=self.loss_,
                                metrics=self.metrics_task_,
                                optimizer=self.optimizer_src)
        
        if (Xt is not None and yt is not None and 
            not "validation_data" in fit_params):
            hist = self.model_src_.fit(Xs, ys,
                                       validation_data=(Xt, yt),
                                       **fit_params)
        else:
            hist = self.model_src_.fit(Xs, ys, **fit_params)
        
        for k, v in hist.history.items():
            if not hasattr(self, "history_src_"):
                self.history_src_ = {}
            self.history_src_[k] = self.history_src_.get(k, []) + v            
            
        self.task_sec_.predict(zeros_enc_);
#         Add a small noise on weights to avoid task and task_sec
#         being identical.
        weights = self.task_.get_weights()
        for i in range(len(weights)):
            weights[i] += (0.01 * weights[i] *
                           np.random.standard_normal(weights[i].shape))
        self.task_sec_.set_weights(weights)
        return self


    def fit_target(self, Xs, ys, Xt, yt=None, **fit_params):
        self._fit(Xs, ys, Xt, yt, **fit_params)
        return self
        
    
    def fit(self, Xs, ys, Xt, yt=None, fit_params_src=None, **fit_params):  
        if fit_params_src is None:
            fit_params_src = fit_params
        
        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)
        
        if not self.is_pretrained_:
            self.fit_source(Xs, ys, Xt, yt, **fit_params_src)
            self.is_pretrained_ = True
        
        self.fit_target(Xs, ys, Xt, yt, **fit_params)
        return self
    
    
    def create_model(self, inputs_Xs, inputs_Xt):

        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)
        
        task_src = GradientHandler(0., name="gh_0")(encoded_src)
        task_sec_src = GradientHandler(0., name="gh_1")(encoded_src)
        task_src = self.task_(task_src)
        task_sec_src = self.task_sec_(task_sec_src)
        
        task_tgt = GradientHandler(-self.lambda_, name="gh_2")(encoded_tgt)
        task_sec_tgt = GradientHandler(-self.lambda_, name="gh_3")(encoded_tgt)
        task_tgt = self.task_(task_tgt)
        task_sec_tgt = self.task_sec_(task_sec_tgt)
        
        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       task_sec_src=task_sec_src,
                       task_sec_tgt=task_sec_tgt)
        return outputs


    def get_loss(self, inputs_ys, task_src,
                  task_tgt, task_sec_src, task_sec_tgt):
        
        loss_task = 0.5 * (self.loss_(inputs_ys, task_src) + self.loss_(inputs_ys, task_sec_src))

        loss_disc = K.mean(K.abs(subtract([task_tgt, task_sec_tgt])), axis=-1)
        
        loss = K.mean(loss_task) - K.mean(loss_disc)
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt,
                     task_src, task_tgt,
                     task_sec_src, task_sec_tgt):
        metrics = {}
        
        disc = K.abs(K.mean(subtract([task_tgt, task_sec_tgt]), axis=-1))
        task_s = self.loss_(inputs_ys, task_src)
        
        metrics["task_s"] = K.mean(task_s)
        metrics["disc"] = K.mean(disc)
        if inputs_yt is not None:
            task_t = self.loss_(inputs_yt, task_tgt)
            metrics["task_t"] = K.mean(task_t)
        
        names_task, names_disc = self._get_metric_names()
        
        for metric, name in zip(self.metrics_task_, names_task):
            metrics[name + "_s"] = metric(inputs_ys, task_src)
            if inputs_yt is not None:
                metrics[name + "_t"] = metric(inputs_yt, task_tgt)
                
        for metric, name in zip(self.metrics_disc_, names_disc):
            metrics[name] = metric(task_tgt, task_sec_tgt)
        return metrics