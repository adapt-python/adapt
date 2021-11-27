"""
WDGRL
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, subtract
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from adapt.utils import (GradientHandler,
                         check_arrays)
from adapt.feature_based import BaseDeepFeature

EPS = K.epsilon()


class _Interpolation(Layer):
    """
    Layer that produces interpolates points between
    two entries, with the distance of the interpolation
    to the first entry.
    """
    
    def call(self, inputs):
        Xs = inputs[0]
        Xt = inputs[1]
        batch_size = tf.shape(Xs)[0]
        dim = tf.shape(Xs)[1:]
        alphas = tf.random.uniform([batch_size]+[1]*len(dim))
        tiled_shape = tf.concat(([1], dim), 0)
        tiled_alphas = tf.tile(alphas, tiled_shape)
        differences = Xt - Xs
        interpolates = Xs + tiled_alphas * differences
        distances = K.sqrt(K.mean(K.square(tiled_alphas * differences),
                          axis=[i for i in range(1, len(dim))]) + EPS)
        return interpolates, distances


class WDGRL(BaseDeepFeature):
    """
    WDGRL (Wasserstein Distance Guided Representation Learning) is an
    unsupervised domain adaptation method on the model of the 
    :ref:`DANN <adapt.feature_based.DANN>`. In WDGRL the discriminator
    is used to approximate the Wasserstein distance between the
    source and target encoded distributions in the spirit of WGAN.
    
    The optimization formulation is the following:
    
    .. math::
    
        \min_{\phi, F} & \; \mathcal{L}_{task}(F(\phi(X_S)), y_S) +
        \lambda \\left(D(\phi(X_S)) - D(\phi(X_T)) \\right) \\\\
        \max_{D} & \; \\left(D(\phi(X_S)) - D(\phi(X_T)) \\right) -
        \\gamma (||\\nabla D(\\alpha \phi(X_S) + (1- \\alpha) \phi(X_T))||_2 - 1)^2
        
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi, F, D` are respectively the **encoder**, the **task**
      and the **discriminator** networks
    - :math:`\lambda` is the trade-off parameter.
    - :math:`\\gamma` is the gradient penalty parameter.
    
    .. figure:: ../_static/images/wdgrl.png
        :align: center
        
        WDGRL architecture (source: [1])
    
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
        be ``(None, 1)``.
        
    lambda_ : float or None (default=1)
        Trade-off parameter. This parameter gives the trade-off
        for the encoder between learning the task and matching
        the source and target distribution. If `lambda_`is small
        the encoder will focus on the task. If `lambda_=0`, WDGRL
        is equivalent to a "source only" method.
        
    gamma : float (default=1.)
        Gradient penalization parameter. To well approximate the
        Wasserstein, the `discriminator`should be 1-Lipschitz.
        This constraint is imposed by the gradient penalty term
        of the optimization. The good value `gamma` to use is
        not easy to find. One can check through the metrics that
        the gradient penalty term is in the same order than the
        "disc loss". If `gamma=0`, no penalty is given on the
        discriminator gradient.
        
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
    >>> from adapt.feature_based import WDGRL
    >>> np.random.seed(0)
    >>> Xs = np.concatenate((np.random.random((100, 1)),
    ...                      np.zeros((100, 1))), 1)
    >>> Xt = np.concatenate((np.random.random((100, 1)),
    ...                      np.ones((100, 1))), 1)
    >>> ys = 0.2 * Xs[:, 0]
    >>> yt = 0.2 * Xt[:, 0]
    >>> model = WDGRL(lambda_=0., random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_["task_t"][-1]
    0.0223...
    >>> model = WDGRL(lambda_=1, random_state=0)
    >>> model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
    >>> model.history_["task_t"][-1]
    0.0044...
        
    See also
    --------
    DANN
    ADDA
    DeepCORAL
    
        References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1707.01217.pdf>`_ Shen, J., Qu, Y., Zhang, W., \
and Yu, Y. Wasserstein distance guided representation learning for domain adaptation. \
In AAAI, 2018.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 lambda_=1.,
                 gamma=1.,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 copy=True,
                 random_state=None):
        
        self.lambda_ = lambda_
        self.gamma = gamma        
        super().__init__(encoder, task, discriminator,
                         loss, metrics, optimizer, copy,
                         random_state)

    
    def create_model(self, inputs_Xs, inputs_Xt):

        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)
        task_src = self.task_(encoded_src)
        task_tgt = self.task_(encoded_tgt)
        
        flip = GradientHandler(-self.lambda_, name="flip")
        no_grad = GradientHandler(0, name="no_grad")
        
        disc_src = flip(encoded_src)
        disc_src = self.discriminator_(disc_src)
        disc_tgt = flip(encoded_tgt)
        disc_tgt = self.discriminator_(disc_tgt)
        
        encoded_src_no_grad = no_grad(encoded_src)
        encoded_tgt_no_grad = no_grad(encoded_tgt)
        
        interpolates, distances = _Interpolation()([encoded_src_no_grad, encoded_tgt_no_grad])
        disc_grad = K.abs(
            subtract([self.discriminator_(interpolates), self.discriminator_(encoded_src_no_grad)])
        )
        disc_grad /= distances
        
        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       disc_src=disc_src,
                       disc_tgt=disc_tgt,
                       disc_grad=disc_grad)
        return outputs

    
    def get_loss(self, inputs_ys, inputs_yt,
                  task_src, task_tgt,
                  disc_src, disc_tgt,
                  disc_grad):
        
        loss_task = self.loss_(inputs_ys, task_src)
        loss_disc = K.mean(disc_src) - K.mean(disc_tgt)
        gradient_penalty = K.mean(K.square(disc_grad-1.))
                            
        loss = K.mean(loss_task) - K.mean(loss_disc) + self.gamma * K.mean(gradient_penalty)
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt,
                     task_src, task_tgt,
                     disc_src, disc_tgt, disc_grad):
        metrics = {}
        
        task_s = self.loss_(inputs_ys, task_src)
        disc = K.mean(disc_src) - K.mean(disc_tgt)
        grad_pen = K.square(disc_grad-1.)
        
        metrics["task_s"] = K.mean(task_s)
        metrics["disc"] = K.mean(disc)
        metrics["grad_pen"] = self.gamma * K.mean(grad_pen)
       
        if inputs_yt is not None:
            task_t = self.loss_(inputs_yt, task_tgt)
            metrics["task_t"] = K.mean(task_t)
        
        names_task, names_disc = self._get_metric_names()
        
        for metric, name in zip(self.metrics_task_, names_task):
            metrics[name + "_s"] = metric(inputs_ys, task_src)
            if inputs_yt is not None:
                metrics[name + "_t"] = metric(inputs_yt, task_tgt)
        return metrics