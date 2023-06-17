"""
WDGRL
"""

import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc

EPS = np.finfo(np.float32).eps


@make_insert_doc(["encoder", "task", "discriminator"])
class WDGRL(BaseAdaptDeep):
    """
    WDGRL: Wasserstein Distance Guided Representation Learning
    
    WDGRL is an unsupervised domain adaptation method on the model of the 
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
    lambda_ : float or tensorflow Variable (default=1)
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
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        task network.
        
    discriminator_ : tensorflow Model
        discriminator network.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import WDGRL
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = WDGRL(lambda_=1., gamma=1., Xt=Xt, metrics=["acc"], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 100ms/step - loss: 0.2112 - acc: 0.7500
    0.21115829050540924
        
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
                 Xt=None,
                 lambda_=0.1,
                 gamma=10.,
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
    
    
    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
       
        # loss
        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:           

            # Forward pass
            Xs_enc = self.encoder_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)
            ys_disc = self.discriminator_(Xs_enc, training=True)
            
            Xt_enc = self.encoder_(Xt, training=True)
            yt_disc = self.discriminator_(Xt_enc, training=True)
                       
            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))
            
            # 1-Lipschitz penalization
            batch_size = tf.shape(Xs_enc)[0]
            dim = len(Xs_enc.shape)-1
            alphas = tf.random.uniform([batch_size]+[1]*dim)
            # tiled_shape = tf.concat(([1], dim), 0)
            # tiled_alphas = tf.tile(alphas, tiled_shape)
            # differences = Xt_enc - Xs_enc
            interpolations = alphas * Xs_enc + (1.-alphas) * Xt_enc
                        
            with tf.GradientTape() as tape_pen:
                tape_pen.watch(interpolations)
                inter_disc = self.discriminator_(interpolations)
            gradients_pen = tape_pen.gradient(inter_disc, interpolations)
            norm_pen = tf.sqrt(tf.reduce_sum(tf.square(gradients_pen),
                                             axis=[i for i in range(1, dim)]) + EPS)
            penalty = self.gamma * tf.reduce_mean(tf.square(1. - norm_pen))
            
            # Compute the loss value
            task_loss = self.task_loss_(ys, ys_pred)
            task_loss = tf.reduce_mean(task_loss)
            
            disc_loss_enc = tf.reduce_mean(ys_disc) - tf.reduce_mean(yt_disc)
            
            enc_loss = task_loss - self.lambda_ * disc_loss_enc
            
            disc_loss = disc_loss_enc + penalty
            
            task_loss += sum(self.task_.losses)
            disc_loss += sum(self.discriminator_.losses)
            enc_loss += sum(self.encoder_.losses)
            
        # Compute gradients
        trainable_vars_task = self.task_.trainable_variables
        trainable_vars_enc = self.encoder_.trainable_variables
        trainable_vars_disc = self.discriminator_.trainable_variables
        
        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
        gradients_disc = disc_tape.gradient(disc_loss, trainable_vars_disc)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))
        self.optimizer_disc.apply_gradients(zip(gradients_disc, trainable_vars_disc))
        
        # Update metrics
        self.compiled_metrics.update_state(ys, ys_pred)
        self.compiled_loss(ys, ys_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        disc_metrics = self._get_disc_metrics(ys_disc, yt_disc)
        logs.update(disc_metrics)
        logs.update({"gp": penalty})
        return logs
    
    
    def _get_disc_metrics(self, ys_disc, yt_disc):
        disc_dict = {}
        disc_dict["disc_loss"] = tf.reduce_mean(ys_disc) - tf.reduce_mean(yt_disc)
        for m in self.disc_metrics:
            disc_dict["disc_%s"%m.name] = tf.reduce_mean(
                m(ys_disc, yt_disc))
        return disc_dict