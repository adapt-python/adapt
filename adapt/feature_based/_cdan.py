"""
CDAN
"""

import warnings
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Input, subtract, Dense, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from adapt.feature_based import BaseDeepFeature
from adapt.utils import (GradientHandler,
                         check_arrays,
                         check_one_array,
                         check_network)


EPS = K.epsilon()


def _get_default_classifier():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model


class CDAN(BaseDeepFeature):
    """
    CDAN (Conditional Adversarial Domain Adaptation) is an
    unsupervised domain adaptation method on the model of the 
    :ref:`DANN <adapt.feature_based.DANN>`. In CDAN the discriminator
    is conditioned on the prediction of the task network for
    source and target data. This should , in theory, focus the
    source-target matching of instances belonging to the same class.
    
    To condition the **discriminator** network on each class, a
    multilinear map of shape: ``nb_class * encoder.output_shape[1]``
    is given as input. If the shape is too large (>4096), a random
    sub-multilinear map of lower dimension is considered.
    
    The optimization formulation of CDAN is the following:
    
    .. math::
    
        \min_{\phi, F} & \; \mathcal{L}_{task}(F(\phi(X_S)), y_S) -
        \lambda \\left( \log(1 - D(\phi(X_S) \\bigotimes F(X_S)) + \\\\
        \log(D(\phi(X_T) \\bigotimes F(X_T)) \\right) \\\\
        \max_{D} & \; \log(1 - D(\phi(X_S) \\bigotimes F(X_S)) + \\\\
        \log(D(\phi(X_T) \\bigotimes F(X_T))
        
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi, F, D` are respectively the **encoder**, the **task**
      and the **discriminator** networks
    - :math:`\lambda` is the trade-off parameter.
    - :math:`\phi(X_S) \\bigotimes F(X_S)` is the multilinear map between
      the encoded sources and the task predictions.
    
    In CDAN+E, an entropy regularization is added to prioritize the
    transfer of easy-to-transfer exemples. The optimization formulation
    of CDAN+E is the following:
    
    .. math::
    
        \min_{\phi, F} & \; \mathcal{L}_{task}(F(\phi(X_S)), y_S) -
        \lambda \\left( \log(1 - W_S D(\phi(X_S) \\bigotimes F(X_S)) + \\\\
        W_T \log(D(\phi(X_T) \\bigotimes F(X_T)) \\right) \\\\
        \max_{D} & \; \log(1 - W_S D(\phi(X_S) \\bigotimes F(X_S)) + \\\\
        W_T \log(D(\phi(X_T) \\bigotimes F(X_T))
        
    Where:
    
    - :math:`W_S = 1+\exp{-\\text{entropy}(F(X_S))}`
    - :math:`\\text{entropy}(F(X_S)) = - \sum_{i < C} F(X_S)_i \log(F(X_S)_i)`
      with :math:`C` the number of classes.
      
    .. figure:: ../_static/images/cdan.png
        :align: center
        
        CDAN architecture (source: [1])
    
    Notes
    -----
    CDAN is specific for multi-class classification tasks. Be sure to add a
    softmax activation at the end of the task network.
    
    Parameters
    ----------
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
        
    task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
        ``task`` should end with a softmax activation.
        
    discriminator : tensorflow Model (default=None)
        Discriminator netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as discriminator
        network. Note that the output shape of the discriminator should
        be ``(None, 1)`` and the input shape: 
        ``(None, encoder.output_shape[1] * nb_class)``.
        
    lambda_ : float or None (default=1)
        Trade-off parameter. This parameter gives the trade-off
        for the encoder between learning the task and matching
        the source and target distribution. If `lambda_`is small
        the encoder will focus on the task. If `lambda_=0`, CDAN
        is equivalent to a "source only" method.
        
    entropy : boolean (default=True)
        Whether to use or not the entropy regularization.
        Adding this regularization will prioritize the
        ``discriminator`` on easy-to-transfer examples.
        This, in theory, should make the transfer "safer".
        
    max_features : int (default=4096)
        If ``encoder.output_shape[1] * nb_class)`` is higer than
        ``max_features`` the multilinear map is produced with
        considering random sub vectors of the encoder and task outputs.
        
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
        
    See also
    --------
    DANN
    ADDA
    WDGRL
    
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1705.10667.pdf>`_ Long, M., Cao, \
Z., Wang, J., and Jordan, M. I. "Conditional adversarial domain adaptation". \
In NIPS, 2018
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 lambda_=1.,
                 entropy=True,
                 max_features=4096,
                 loss="mse",
                 metrics=None,
                 optimizer=None,
                 copy=True,
                 random_state=None):
        
        self.lambda_ = lambda_
        self.entropy = entropy
        self.max_features = max_features
        
        if task is None:
            task = _get_default_classifier()
        super().__init__(encoder, task, discriminator,
                         loss, metrics, optimizer, copy,
                         random_state)

    
    def create_model(self, inputs_Xs, inputs_Xt):
        encoded_src = self.encoder_(inputs_Xs)
        encoded_tgt = self.encoder_(inputs_Xt)
        task_src = self.task_(encoded_src)
        task_tgt = self.task_(encoded_tgt)
        
        no_grad = GradientHandler(0., name="no_grad")
        flip = GradientHandler(-self.lambda_, name="flip")
        
        task_src_nograd = no_grad(task_src)
        task_tgt_nograd = no_grad(task_tgt)
        
        if task_src.shape[1] * encoded_src.shape[1] > self.max_features:
            self._random_task = tf.random.normal([task_src.shape[1],
                                            self.max_features])
            self._random_enc = tf.random.normal([encoded_src.shape[1],
                                           self.max_features])
            
            mapping_task_src = tf.matmul(task_src_nograd, self._random_task)
            mapping_enc_src = tf.matmul(encoded_src, self._random_enc)
            mapping_src = tf.multiply(mapping_enc_src, mapping_task_src)
            mapping_src /= (tf.math.sqrt(tf.cast(self.max_features, tf.float32)) + EPS)
            
            mapping_task_tgt = tf.matmul(task_tgt_nograd, self._random_task)
            mapping_enc_tgt = tf.matmul(encoded_tgt, self._random_enc)
            mapping_tgt = tf.multiply(mapping_enc_tgt, mapping_task_tgt)
            mapping_tgt /= (tf.math.sqrt(tf.cast(self.max_features, tf.float32)) + EPS)
        
        else:
            mapping_src = tf.matmul(
                tf.expand_dims(encoded_src, 2),
                tf.expand_dims(task_src_nograd, 1))
            mapping_tgt = tf.matmul(
                tf.expand_dims(encoded_tgt, 2),
                tf.expand_dims(task_tgt_nograd, 1))

            mapping_src = Flatten("channels_first")(mapping_src)
            mapping_tgt = Flatten("channels_first")(mapping_tgt)
        
        disc_src = flip(mapping_src)
        disc_src = self.discriminator_(disc_src)
        disc_tgt = flip(mapping_tgt)
        disc_tgt = self.discriminator_(disc_tgt)
        
        outputs = dict(task_src=task_src,
                       task_tgt=task_tgt,
                       disc_src=disc_src,
                       disc_tgt=disc_tgt,
                       task_src_nograd=task_src_nograd,
                       task_tgt_nograd=task_tgt_nograd)
        return outputs

    
    def get_loss(self, inputs_ys,
                 task_src, task_tgt,
                 disc_src, disc_tgt,
                 task_src_nograd,
                 task_tgt_nograd):
        
        loss_task = self.loss_(inputs_ys, task_src)
        
        if self.entropy:            
            entropy_src = -tf.reduce_sum(task_src_nograd *
                                         tf.math.log(task_src_nograd+EPS),
                                         axis=1, keepdims=True)
            entropy_tgt = -tf.reduce_sum(task_tgt_nograd *
                                         tf.math.log(task_tgt_nograd+EPS),
                                         axis=1, keepdims=True)
            weight_src = 1.+tf.exp(-entropy_src)
            weight_tgt = 1.+tf.exp(-entropy_tgt)
            weight_src /= (tf.reduce_mean(weight_src) + EPS)
            weight_tgt /= (tf.reduce_mean(weight_tgt) + EPS)
            weight_src *= .5
            weight_tgt *= .5
            
            assert str(weight_src.shape) == str(disc_src.shape)
            assert str(weight_tgt.shape) == str(disc_tgt.shape)
            
            loss_disc = (-tf.math.log(1-weight_src*disc_src + EPS)
                         -tf.math.log(weight_tgt*disc_tgt + EPS))
        else:
            loss_disc = (-tf.math.log(1-disc_src + EPS)
                         -tf.math.log(disc_tgt + EPS))
        
        loss = tf.reduce_mean(loss_task) + tf.reduce_mean(loss_disc)
        return loss
    
    
    def get_metrics(self, inputs_ys, inputs_yt,
                     task_src, task_tgt,
                     disc_src, disc_tgt,
                    task_src_nograd,
                    task_tgt_nograd):
        metrics = {}
        
        task_s = self.loss_(inputs_ys, task_src)
        
        if self.entropy:            
            entropy_src = -tf.reduce_sum(task_src_nograd *
                                         tf.math.log(task_src_nograd+EPS),
                                         axis=1, keepdims=True)
            entropy_tgt = -tf.reduce_sum(task_tgt_nograd *
                                         tf.math.log(task_tgt_nograd+EPS),
                                         axis=1, keepdims=True)
            weight_src = 1.+tf.exp(-entropy_src)
            weight_tgt = 1.+tf.exp(-entropy_tgt)
            weight_src /= (tf.reduce_mean(weight_src) + EPS)
            weight_tgt /= (tf.reduce_mean(weight_tgt) + EPS)
            weight_src *= .5
            weight_tgt *= .5            
            disc = (-tf.math.log(1-weight_src*disc_src + EPS)
                         -tf.math.log(weight_tgt*disc_tgt + EPS))
        else:
            disc = (-tf.math.log(1-disc_src + EPS)
                         -tf.math.log(disc_tgt + EPS))
        
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
    
    
    def _build(self, shape_Xs, shape_ys,
                    shape_Xt, shape_yt):
        
        # Call predict to avoid strange behaviour with
        # Sequential model whith unspecified input_shape
        zeros_enc_ = self.encoder_.predict(np.zeros((1,) + shape_Xt));
        zeros_task_ = self.task_.predict(zeros_enc_);
        if zeros_task_.shape[1] * zeros_enc_.shape[1] > self.max_features:
            self.discriminator_.predict(np.zeros((1, self.max_features)))
        else:
            zeros_mapping_ = np.matmul(np.expand_dims(zeros_enc_, 2),
                                       np.expand_dims(zeros_task_, 1))
            zeros_mapping_ = np.reshape(zeros_mapping_, (1, -1))
            self.discriminator_.predict(zeros_mapping_);
                
        inputs_Xs = Input(shape_Xs)
        inputs_ys = Input(shape_ys)
        inputs_Xt = Input(shape_Xt)
                
        if shape_yt is None:
            inputs_yt = None
            inputs = [inputs_Xs, inputs_ys, inputs_Xt]
        else:
            inputs_yt = Input(shape_yt)
            inputs = [inputs_Xs, inputs_ys,
                      inputs_Xt, inputs_yt]
        
        outputs = self.create_model(inputs_Xs=inputs_Xs,
                                    inputs_Xt=inputs_Xt)
        
        self.model_ = Model(inputs, outputs)
        
        loss = self.get_loss(inputs_ys=inputs_ys,
                              **outputs)
        metrics = self.get_metrics(inputs_ys=inputs_ys,
                                    inputs_yt=inputs_yt,
                                    **outputs)
        
        self.model_.add_loss(loss)
        for k in metrics:            
            self.model_.add_metric(tf.reduce_mean(metrics[k]),
                                   name=k, aggregation="mean")
        
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model_.compile(optimizer=self.optimizer)
        self.history_ = {}
        return self
    
    
    def predict_disc(self, X):
        X_enc = self.encoder_.predict(X)
        X_task = self.task_.predict(X_enc)
        if X_enc.shape[1] * X_task.shape[1] > self.max_features:
            X_enc = X_enc.dot(self._random_enc.numpy())
            X_task = X_task.dot(self._random_task.numpy())
            X_disc = X_enc * X_task
            X_disc /= np.sqrt(self.max_features)
        else:
            X_disc = np.matmul(np.expand_dims(X_enc, 2),
                               np.expand_dims(X_task, 1))
            X_disc = X_disc.transpose([0, 2, 1])
            X_disc = X_disc.reshape(-1, X_enc.shape[1] * X_task.shape[1])
        y_disc = self.discriminator_.predict(X_disc)
        return y_disc