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


class MDD(BaseDeepFeature):
    """
    MDD: Margin Disparity Discrepancy is a feature-based domain adaptation
    method originally introduced for unsupervised classification DA.
    
    The goal of MDD is to find a new representation of the input features which
    minimizes the disparity discrepancy between the source and target domains 
    
    The discrepancy is estimated through adversarial training of three networks:
    An encoder a task network and a discriminator.
    
    Parameters
    ----------
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
        
    get_task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        
    gamma : float (default=4.)
        Margin parameter.

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
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        Principal task network.
        
    discriminator_ : tensorflow Model
        Adversarial task network.
    
    model_ : tensorflow Model
        Fitted model: the union of ``encoder_``, ``task_``,
        and ``discriminator_`` networks.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> Xs = np.concatenate((np.random.random((100, 1)),
    ...                      np.zeros((100, 1))), 1)
    >>> Xt = np.concatenate((np.random.random((100, 1)),
    ...                      np.ones((100, 1))), 1)
    >>> ys = 0.2 * Xs[:, 0]
    >>> yt = 0.2 * Xt[:, 0]
    >>> model = MDD(random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_["task_t"][-1]
    0.0009...
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1904.05801.pdf>`_ Y. Zhang, \
T. Liu, M. Long, and M. Jordan. "Bridging theory and algorithm for \
domain adaptation". ICML, 2019.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 is_pretrained=False,
                 lambda_=1.,
                 gamma=4.,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 optimizer_src=None,
                 copy=True,
                 random_state=None):

        super().__init__(encoder, task, None,
                         loss, metrics, optimizer, copy,
                         random_state)
        self.lambda_ = lambda_
        self.gamma = gamma
        
        if optimizer_src is None:
            self.optimizer_src = deepcopy(self.optimizer)
        else:
            self.optimizer_src = optimizer_src

        self.discriminator_ = check_network(self.task_, 
                                       copy=True,
                                       display_name="task",
                                       force_copy=True)
        self.discriminator_._name = self.discriminator_._name + "_2"
        
        if hasattr(self.loss_, "__name__"):
            self.loss_name_ = self.loss_.__name__
        elif hasattr(self.loss_, "__class__"):
            self.loss_name_ = self.loss_.__class__.__name__
        else:
            self.loss_name_ = ""

      
    def create_model(self, inputs_Xs, inputs_Xt):

        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)

        task_src = self.task_(encoded_src)
        task_tgt = self.task_(encoded_tgt)
        
        task_src_nograd = GradientHandler(0., name="gh_2")(task_src)
        task_tgt_nograd = GradientHandler(0., name="gh_3")(task_tgt)
        
        # TODO, add condition for bce and cce     
#         if self.loss_name_ in ["categorical_crossentropy",
#                                "CategoricalCrossentropy"]:

        disc_src = GradientHandler(-self.lambda_, name="gh_0")(encoded_src)
        disc_tgt = GradientHandler(-self.lambda_, name="gh_1")(encoded_tgt)
        disc_src = self.discriminator_(disc_src)
        disc_tgt = self.discriminator_(disc_tgt)

        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       task_src_nograd=task_src_nograd,
                       task_tgt_nograd=task_tgt_nograd,
                       disc_src=disc_src,
                       disc_tgt=disc_tgt)
        return outputs


    def get_loss(self, inputs_ys, inputs_yt, task_src,
                 task_src_nograd, task_tgt_nograd,
                 task_tgt, disc_src, disc_tgt):
        
        task_loss = self.loss_(inputs_ys, task_src)
        
        disc_loss_src = self.loss_(task_src_nograd, disc_src)
        disc_loss_tgt = self.loss_(task_tgt_nograd, disc_tgt)
        
        disc_loss = disc_loss_tgt - self.gamma * disc_loss_src
        
        loss = K.mean(task_loss) - K.mean(disc_loss)
        return loss


    def get_metrics(self, inputs_ys, inputs_yt,
                    task_src, task_tgt,
                    task_src_nograd, task_tgt_nograd,
                    disc_src, disc_tgt):
        metrics = {}
        
        task_s = self.loss_(inputs_ys, task_src)
        disc = (self.loss_(task_tgt_nograd, disc_tgt) -
                self.gamma * self.loss_(task_src_nograd, disc_src))
        
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
            metrics[name] = (metric(task_tgt_nograd, disc_tgt) -
                self.gamma * metric(task_src_nograd, disc_src))
        return metrics
