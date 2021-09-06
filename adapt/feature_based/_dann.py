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
from adapt.feature_based._deep import UpdateLambda

EPS = K.epsilon()


class DANN(BaseDeepFeature):
    """
    DANN: Discriminative Adversarial Neural Network
    
    DANN is a feature-based domain adaptation method.
    
    The goal of DANN is to find a new representation of the input features
    in which source and target data could not be distinguished by any
    **discriminator** network. This new representation is learned by an
    **encoder** network in an adversarial fashion. A **task** network is
    learned on the encoded space in parallel to the **encoder** and 
    **discriminator** networks.
    
    The three network paremeters are optimized according to the
    following objectives:
    
    .. math::
    
        \min_{\phi, F} & \; \mathcal{L}_{task}(F(\phi(X_S)), y_S) -
        \lambda \\left(
        \log(1 - D(\phi(X_S))) + \log(D(\phi(X_T))) \\right) \\\\
        \max_{D} & \; \log(1 - D(\phi(X_S))) + \log(D(\phi(X_T)))
        
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi, F, D` are respectively the **encoder**, the **task**
      and the **discriminator** networks
    - :math:`\lambda` is the trade-off parameter.
    
    The adversarial training is done through a **reversal gradient layer**
    placed between the **encoder** and the **discriminator** networks.
    This layer inverses the gradient sign in backpropagation, thus the
    two networks are optimized according to two opposite objective functions.
    
    The method has been originally introduced for **unsupervised**
    classification DA but it could be widen to other task in
    **supervised** DA straightforwardly.
    
    .. figure:: ../_static/images/dann.png
        :align: center
        
        DANN architecture (source: [1])
    
    Parameters
    ----------
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
        
    task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        
    discriminator : tensorflow Model (default=None)
        Discriminator netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as discriminator
        network. Note that the output shape of the discriminator should
        be ``(None, 1)`` and a ``sigmoid`` activation should be used.
        
    lambda_ : float or None (default=0.1)
        Trade-off parameter.
        If ``None``, ``lambda_`` increases gradually
        according to the following formula:
        ``lambda_`` = 2/(1 + exp(-``gamma`` * p)) - 1.
        With p growing from 0 to 1 during training.
        
    gamma : float (default=10.0)
        Increase rate parameter.
        Give the increase rate of the trade-off parameter if
        ``lambda_`` is set to ``None``.

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
        task network.
        
    discriminator_ : tensorflow Model
        discriminator network.
    
    model_ : tensorflow Model
        Fitted model: the union of ``encoder_``,
        ``task_`` and ``discriminator_`` networks.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.feature_based import DANN
    >>> np.random.seed(0)
    >>> Xs = np.concatenate((np.random.random((100, 1)),
    ...                      np.zeros((100, 1))), 1)
    >>> Xt = np.concatenate((np.random.random((100, 1)),
    ...                      np.ones((100, 1))), 1)
    >>> ys = 0.2 * Xs[:, 0]
    >>> yt = 0.2 * Xt[:, 0]
    >>> model = DANN(lambda_=0., random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_["task_t"][-1]
    0.0240...
    >>> model = DANN(lambda_=0.1, random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_["task_t"][-1]
    0.0022...
    
    See also
    --------
    ADDA
    DeepCORAL
        
    References
    ----------
    .. [1] `[1] <http://jmlr.org/papers/volume17/15-239/15-239.pdf>`_ Y. Ganin, \
E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, \
and V. Lempitsky. "Domain-adversarial training of neural networks". In JMLR, 2016.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 lambda_=0.1,
                 gamma=10.,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 copy=True,
                 random_state=None):
        
        self.lambda_ = lambda_
        if self.lambda_ is None:
            self.lambda_init_ = 0.
        else:
            self.lambda_init_ = self.lambda_
        self.gamma = gamma        
        super().__init__(encoder, task, discriminator,
                         loss, metrics, optimizer, copy,
                         random_state)

        
    def fit(self, Xs, ys, Xt, yt=None, **fit_params):  
        Xs, ys, Xt, yt = check_arrays(Xs, ys, Xt, yt)
        
        # Define callback for incresing lambda_, if model_ is
        # already built, do not reinitialized lambda_
        if self.lambda_ is None and not hasattr(self, "model_"):
            callback = UpdateLambda(gamma=self.gamma)
            if "callbacks" in fit_params:
                fit_params["callbacks"].append(callback)
            else:
                fit_params["callbacks"] = [callback]

        self._fit(Xs, ys, Xt, yt, **fit_params)
        return self
    
    
    def create_model(self, inputs_Xs, inputs_Xt):

        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)
        task_src = self.task_(encoded_src)
        task_tgt = self.task_(encoded_tgt)
        
        flip = GradientHandler(-self.lambda_init_)
        
        disc_src = flip(encoded_src)
        disc_src = self.discriminator_(disc_src)
        disc_tgt = flip(encoded_tgt)
        disc_tgt = self.discriminator_(disc_tgt)
        
        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       disc_src=disc_src,
                       disc_tgt=disc_tgt)
        return outputs


    def get_loss(self, inputs_ys,
                  task_src, task_tgt,
                  disc_src, disc_tgt):
        
        loss_task = self.loss_(inputs_ys, task_src)
        loss_disc = (-K.log(1-disc_src + EPS)
                     -K.log(disc_tgt + EPS))
        
        loss = K.mean(loss_task) + K.mean(loss_disc)
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt,
                     task_src, task_tgt,
                     disc_src, disc_tgt):
        metrics = {}
        
        task_s = self.loss_(inputs_ys, task_src)
        disc = (-K.log(1-disc_src + EPS)
                -K.log(disc_tgt + EPS))
        
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
            pred = K.concatenate((disc_src, disc_tgt), axis=0)
            true = K.concatenate((K.zeros_like(disc_src),
                                  K.ones_like(disc_tgt)), axis=0)
            metrics[name] = metric(true, pred)
        return metrics