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
                         check_arrays)
from adapt.feature_based import BaseDeepFeature

EPS = K.epsilon()


class DeepCORAL(BaseDeepFeature):
    """
    DeepCORAL: Deep CORrelation ALignment
    
    DeepCORAL is an extension of CORAL method. It learns a nonlinear
    transformation which aligns correlations of layer activations in
    deep neural networks.
    
    The method consists in training both an **encoder** and a **task**
    network. The **encoder** network maps input features into new
    encoded ones on which the **task** network is trained.
    
    The parameters of the two networks are optimized in order to
    minimize the following loss function:
    
    .. math::
    
        \mathcal{L} = \mathcal{L}_{task} + \\lambda ||C_S - C_T||_F^2
        
    Where:
    
    - :math:`\mathcal{L}_{task}` is the task loss computed with
      source labeled data.
    - :math:`C_S` is the correlation matrix of source data in the
      encoded feature space.
    - :math:`C_T` is the correlation matrix of target data in the
      encoded feature space.
    - :math:`||.||_F` is the Frobenius norm.
    - :math:`\\lambda` is a trade-off parameter.
    
    Thus the **encoder** network learns a new feature representation
    on wich the correlation matrixes of source and target data are
    "close" and where a **task** network is able to learn the task
    with source labeled data.
    
    Notice that DeepCORAL only uses labeled source and unlabeled target
    data. It belongs then to "unsupervised" domain adaptation methods.
    
    .. figure:: ../_static/images/deepcoral.png
        :align: center
        
        DeepCORAL architecture (source: [1])
    
    Parameters
    ----------
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
        
    task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        
    lambda_ : float (default=1.)
        Trade-off parameter.

    loss : string or tensorflow loss (default="mse")
        Loss function used for the task.
        
    metrics : dict or list of string or tensorflow metrics (default=None)
        Metrics given to the model. Metrics are used
        on ``task`` outputs.
        
    optimizer : string or tensorflow optimizer (default=None)
        Optimizer of the model. If ``None``, the
        optimizer is set to tf.keras.optimizers.Adam(0.001)
        
    copy : boolean (default=True)
        Whether to make a copy of ``encoder``
        and ``task`` or not.
        
    random_state : int (default=None)
        Seed of random generator.
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        task network.
    
    model_ : tensorflow Model
        Fitted model: the union of ``encoder_``
        and ``task_`` networks.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.feature_based import DeepCORAL
    >>> np.random.seed(0)
    >>> Xs = np.random.multivariate_normal(
    ...      np.array([0, 0]), np.array([[0.001, 0], [0, 1]]), 100)
    >>> Xt = np.random.multivariate_normal(
    ...      np.array([0, 0]), np.array([[0.1, 0.2], [0.2, 0.5]]), 100)
    >>> ys = np.zeros(100)
    >>> yt = np.zeros(100)
    >>> ys[Xs[:, 1]>0] = 1
    >>> yt[(Xt[:, 1]-0.5*Xt[:, 0])>0] = 1
    >>> model = DeepCORAL(lambda_=0., random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=500, batch_size=100, verbose=0)
    >>> model.history_["task_t"][-1]
    1.30188e-05
    >>> model = DeepCORAL(lambda_=1., random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=500, batch_size=100, verbose=0)
    >>> model.history_["task_t"][-1]
    5.4704474e-06
        
    See also
    --------
    CORAL
    DANN
    ADDA

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1607.01719.pdf>`_ Sun B. and Saenko K. \
"Deep CORAL: correlation alignment for deep domain adaptation." In ICCV, 2016.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 lambda_=1.,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 copy=True,
                 random_state=None):
        
        self.lambda_ = lambda_    
        super().__init__(encoder, task, None,
                         loss, metrics, optimizer, copy,
                         random_state)

        
    def fit(self, Xs, ys, Xt, yt=None, **fit_params):
        self._fit(Xs, ys, Xt, yt, **fit_params)
        return self
    
    
    def create_model(self, inputs_Xs, inputs_Xt):
               
        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)
        
        batch_size = K.mean(K.sum(K.ones_like(inputs_Xs), 0))
        dim = len(encoded_src.shape)
        
        if dim != 2:
            raise ValueError("Encoded space should "
                             "be 2 dimensional, got, "
                             "%s"%encoded_src.shape)
            
        task_src = self.task_(encoded_src)
        task_tgt = self.task_(encoded_tgt)
        
        factor_1 = 1 / (batch_size - 1 + EPS)
        factor_2 = 1 / batch_size
        
        sum_src = K.sum(encoded_src, axis=0)
        sum_src_row = K.reshape(sum_src, (1, -1))
        sum_src_col = K.reshape(sum_src, (-1, 1))
        
        cov_src = factor_1 * (
            K.dot(K.transpose(encoded_src), encoded_src) -
            factor_2 * K.dot(sum_src_col, sum_src_row)
        )
        
        sum_tgt = K.sum(encoded_tgt, axis=0)
        sum_tgt_row = K.reshape(sum_tgt, (1, -1))
        sum_tgt_col = K.reshape(sum_tgt, (-1, 1))
        
        cov_tgt = factor_1 * (
            K.dot(K.transpose(encoded_tgt), encoded_tgt) -
            factor_2 * K.dot(sum_tgt_col, sum_tgt_row)
        )
        
        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       cov_src=cov_src,
                       cov_tgt=cov_tgt)
        return outputs


    def get_loss(self, inputs_ys,
                  task_src, task_tgt,
                  cov_src, cov_tgt):
        
        loss_task = self.loss_(inputs_ys, task_src)
        loss_disc = 0.25 * K.mean(K.square(subtract([cov_src, cov_tgt])))
        loss_disc_lambda = self.lambda_ * loss_disc
        
        loss = K.mean(loss_task) + loss_disc_lambda
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt,
                     task_src, task_tgt,
                     cov_src, cov_tgt):
        metrics = {}
        
        task_s = self.loss_(inputs_ys, task_src)
        disc = 0.25 * K.mean(K.square(subtract([cov_src, cov_tgt])))
        
        metrics["task_s"] = K.mean(task_s)
        metrics["disc"] = self.lambda_ * K.mean(disc)
        if inputs_yt is not None:
            task_t = self.loss_(inputs_yt, task_tgt)
            metrics["task_t"] = K.mean(task_t)
        
        names_task, names_disc = self._get_metric_names()
        
        for metric, name in zip(self.metrics_task_, names_task):
            metrics[name + "_s"] = metric(inputs_ys, task_src)
            if inputs_yt is not None:
                metrics[name + "_t"] = metric(inputs_yt, task_tgt)
        return metrics