"""
DANN
"""

import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc

EPS = np.finfo(np.float32).eps


@make_insert_doc(["encoder", "task", "discriminator"])
class DANN(BaseAdaptDeep):
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
                 Xt=None,
                 yt=None,
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
        
        # Single source
        Xs = Xs[0]
        ys = ys[0]
        
        if self.lambda_ is None:
            _is_lambda_None = 1.
            lambda_ = 0.
        else:
            _is_lambda_None = 0.
            lambda_ = float(self.lambda_)
       
        # loss
        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:           
            
            # Compute lambda
            self.steps_.assign_add(1.)
            progress = self.steps_ / self.total_steps_
            _lambda_ = 2. / (1. + tf.exp(-self.gamma * progress)) - 1.
            _lambda_ = (_is_lambda_None * _lambda_ +
                        (1. - _is_lambda_None) * lambda_)
            
            # Forward pass
            Xs_enc = self.encoder_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)
            ys_disc = self.discriminator_(Xs_enc, training=True)
            
            Xt_enc = self.encoder_(Xt, training=True)
            yt_disc = self.discriminator_(Xt_enc, training=True)
            
            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))
            
            # Compute the loss value
            task_loss = self.task_loss_(ys, ys_pred)
            
            disc_loss = (-tf.math.log(ys_disc + EPS)
                         -tf.math.log(1-yt_disc + EPS))
            
            task_loss = tf.reduce_mean(task_loss)
            disc_loss = tf.reduce_mean(disc_loss)
            
            enc_loss = task_loss - _lambda_ * disc_loss
            
            task_loss += sum(self.task_.losses)
            disc_loss += sum(self.discriminator_.losses)
            enc_loss += sum(self.encoder_.losses)
            
        print(task_loss.shape, enc_loss.shape, disc_loss.shape)
            
        # Compute gradients
        trainable_vars_task = self.task_.trainable_variables
        trainable_vars_enc = self.encoder_.trainable_variables
        trainable_vars_disc = self.discriminator_.trainable_variables
        
        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
        gradients_disc = disc_tape.gradient(disc_loss, trainable_vars_disc)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer.apply_gradients(zip(gradients_enc, trainable_vars_enc))
        self.optimizer.apply_gradients(zip(gradients_disc, trainable_vars_disc))
        
        # Update metrics
        self.compiled_metrics.update_state(ys, ys_pred)
        self.compiled_loss(ys, ys_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        disc_metrics = self._get_disc_metrics(ys_disc, yt_disc)
        logs.update({"disc_loss": disc_loss})
        logs.update(disc_metrics)
        logs.update({"lambda": _lambda_})
        return logs
        
    
    def _get_disc_metrics(self, ys_disc, yt_disc):
        disc_dict = {}
        for m in self.disc_metrics:
            disc_dict["disc_%s"%m.name] = tf.reduce_mean(0.5 * (
                m(tf.ones_like(ys_disc), ys_disc)+
                m(tf.zeros_like(yt_disc), yt_disc)
            ))
        return disc_dict