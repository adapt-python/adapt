"""
DANN
"""

import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import check_network, get_default_encoder, get_default_task

EPS = np.finfo(np.float32).eps


@make_insert_doc(["encoder", "task"])
class DeepCORAL(BaseAdaptDeep):
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
    lambda_ : float or tensorflow Variable (default=1.)
        Trade-off parameter.
        
    match_mean : bool (default=False)
        Weither to match the means of source
        and target or not. If ``False`` only
        the second moment is matched as in the
        original algorithm.

    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        task network.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import DeepCORAL
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = DeepCORAL(lambda_=1., Xt=Xt, metrics=["acc"], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 99ms/step - loss: 0.2029 - acc: 0.6800
    0.2029329240322113
        
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
                 Xt=None,
                 lambda_=1.,
                 match_mean=False,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
            
        if self.match_mean:
            _match_mean = 1.
        else:
            _match_mean = 0.

        # loss
        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:           
                        
            # Forward pass
            Xs_enc = self.encoder_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)
            
            Xt_enc = self.encoder_(Xt, training=True)
            
            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                       
            batch_size = tf.cast(tf.shape(Xs_enc)[0], Xs_enc.dtype)

            factor_1 = 1. / (batch_size - 1. + EPS)
            factor_2 = 1. / batch_size

            sum_src = tf.reduce_sum(Xs_enc, axis=0)
            sum_src_row = tf.reshape(sum_src, (1, -1))
            sum_src_col = tf.reshape(sum_src, (-1, 1))

            cov_src = factor_1 * (
                tf.matmul(tf.transpose(Xs_enc), Xs_enc) -
                factor_2 * tf.matmul(sum_src_col, sum_src_row)
            )

            sum_tgt = tf.reduce_sum(Xt_enc, axis=0)
            sum_tgt_row = tf.reshape(sum_tgt, (1, -1))
            sum_tgt_col = tf.reshape(sum_tgt, (-1, 1))

            cov_tgt = factor_1 * (
                tf.matmul(tf.transpose(Xt_enc), Xt_enc) -
                factor_2 * tf.matmul(sum_tgt_col, sum_tgt_row)
            )
            
            mean_src = tf.reduce_mean(Xs_enc, 0)
            mean_tgt = tf.reduce_mean(Xt_enc, 0)
            
            # Compute the loss value
            task_loss = self.task_loss_(ys, ys_pred)
            disc_loss_cov = 0.25 * tf.square(cov_src - cov_tgt)
            disc_loss_mean = tf.square(mean_src - mean_tgt)
            
            task_loss = tf.reduce_mean(task_loss)
            disc_loss_cov = tf.reduce_mean(disc_loss_cov)
            disc_loss_mean = tf.reduce_mean(disc_loss_mean)
            disc_loss = self.lambda_ * (disc_loss_cov + _match_mean * disc_loss_mean)
            
            task_loss += sum(self.task_.losses)
            disc_loss += sum(self.encoder_.losses)
            
        # Compute gradients
        trainable_vars_task = self.task_.trainable_variables
        trainable_vars_enc = self.encoder_.trainable_variables
        
        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(disc_loss, trainable_vars_enc)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))
        
        # Update metrics
        self.compiled_metrics.update_state(ys, ys_pred)
        self.compiled_loss(ys, ys_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        logs.update({"disc_loss": disc_loss})
        return logs


    def _initialize_networks(self):
        if self.encoder is None:
            self.encoder_ = get_default_encoder(name="encoder", state=self.random_state)
        else:
            self.encoder_ = check_network(self.encoder,
                                          copy=self.copy,
                                          name="encoder")
        if self.task is None:
            self.task_ = get_default_task(name="task", state=self.random_state)
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")