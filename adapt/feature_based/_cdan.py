"""
CDAN
"""

import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc
from tensorflow.keras.initializers import GlorotUniform
from adapt.utils import (check_network,
                         get_default_encoder,
                         get_default_discriminator)

EPS = np.finfo(np.float32).eps


def _get_default_classifier(name=None, state=None):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Flatten())
    if state is None:
        model.add(tf.keras.layers.Dense(10, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation="relu"))
        model.add(tf.keras.layers.Dense(2, activation="softmax"))
    else:
        model.add(tf.keras.layers.Dense(10, activation="relu",
                                       kernel_initializer=GlorotUniform(seed=state)))
        model.add(tf.keras.layers.Dense(10, activation="relu",
                                       kernel_initializer=GlorotUniform(seed=state)))
        model.add(tf.keras.layers.Dense(2, activation="softmax",
                                       kernel_initializer=GlorotUniform(seed=state)))
    return model


@make_insert_doc(["encoder"])
class CDAN(BaseAdaptDeep):
    """
    CDAN: Conditional Adversarial Domain Adaptation
    
    CDAN is an unsupervised domain adaptation method on the model of the 
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
        \lambda \\left( \log(1 - D(\phi(X_S) \\otimes F(X_S)) +
        \log(D(\phi(X_T) \\otimes F(X_T)) \\right) \\\\
        \max_{D} & \; \log(1 - D(\phi(X_S) \\otimes F(X_S)) +
        \log(D(\phi(X_T) \\otimes F(X_T))
        
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi, F, D` are respectively the **encoder**, the **task**
      and the **discriminator** networks
    - :math:`\lambda` is the trade-off parameter.
    - :math:`\phi(X_S) \\otimes F(X_S)` is the multilinear map between
      the encoded sources and the task predictions.
    
    In CDAN+E, an entropy regularization is added to prioritize the
    transfer of easy-to-transfer exemples. The optimization formulation
    of CDAN+E is the following:
    
    .. math::
    
        \min_{\phi, F} & \; \mathcal{L}_{task}(F(\phi(X_S)), y_S) -
        \lambda \\left( \log(1 - W_S D(\phi(X_S) \\otimes F(X_S)) +
        W_T \log(D(\phi(X_T) \\otimes F(X_T)) \\right) \\\\
        \max_{D} & \; \log(1 - W_S D(\phi(X_S) \\otimes F(X_S)) +
        W_T \log(D(\phi(X_T) \\otimes F(X_T))
        
    Where:
    
    - :math:`W_S = 1+\exp^{-\\text{ent}(F(X_S))}`
    - :math:`\\text{ent}(F(X_S)) = - \sum_{i < C} F(X_S)_i \log(F(X_S)_i)`
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
        
    See also
    --------
    DANN
    ADDA
    WDGRL
    
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import CDAN
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> ys = np.stack([ys, np.abs(1-ys)], 1)
    >>> yt = np.stack([yt, np.abs(1-yt)], 1)
    >>> model = CDAN(lambda_=0.1, Xt=Xt, metrics=["acc"], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 106ms/step - loss: 0.1081 - acc: 0.8400
    0.10809497535228729
    
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
                 Xt=None,
                 yt=None,
                 lambda_=1.,
                 entropy=True,
                 max_features=4096,
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
            
            Xt_enc = self.encoder_(Xt, training=True)
            yt_pred = self.task_(Xt_enc, training=True)
            
            if self.is_overloaded_:
                mapping_task_src = tf.matmul(ys_pred, self._random_task)
                mapping_enc_src = tf.matmul(Xs_enc, self._random_enc)
                mapping_src = tf.multiply(mapping_enc_src, mapping_task_src)
                mapping_src /= (tf.math.sqrt(tf.cast(self.max_features, tf.float32)) + EPS)

                mapping_task_tgt = tf.matmul(yt_pred, self._random_task)
                mapping_enc_tgt = tf.matmul(Xt_enc, self._random_enc)
                mapping_tgt = tf.multiply(mapping_enc_tgt, mapping_task_tgt)
                mapping_tgt /= (tf.math.sqrt(tf.cast(self.max_features, tf.float32)) + EPS)
                
            else:
                mapping_src = tf.matmul(
                    tf.expand_dims(Xs_enc, 2),
                    tf.expand_dims(ys_pred, 1))
                mapping_tgt = tf.matmul(
                    tf.expand_dims(Xt_enc, 2),
                    tf.expand_dims(yt_pred, 1))
                
                dim = int(np.prod(mapping_src.get_shape()[1:]))
                mapping_src = tf.reshape(mapping_src, (-1, dim))
                mapping_tgt = tf.reshape(mapping_tgt, (-1, dim))
                
            ys_disc = self.discriminator_(mapping_src)
            yt_disc = self.discriminator_(mapping_tgt)
            
            if self.entropy:
                entropy_src = -tf.reduce_sum(ys_pred *
                                             tf.math.log(ys_pred+EPS),
                                             axis=1, keepdims=True)
                entropy_tgt = -tf.reduce_sum(yt_pred *
                                             tf.math.log(yt_pred+EPS),
                                             axis=1, keepdims=True)
                weight_src = 1.+tf.exp(-entropy_src)
                weight_tgt = 1.+tf.exp(-entropy_tgt)
                weight_src /= (tf.reduce_mean(weight_src) + EPS)
                weight_tgt /= (tf.reduce_mean(weight_tgt) + EPS)
                weight_src *= .5
                weight_tgt *= .5

                assert str(weight_src.shape) == str(ys_disc.shape)
                assert str(weight_tgt.shape) == str(yt_disc.shape)

                disc_loss = (-weight_src*tf.math.log(ys_disc + EPS)
                             -weight_tgt*tf.math.log(1-yt_disc + EPS))
            else:
                disc_loss = (-tf.math.log(ys_disc + EPS)
                             -tf.math.log(1-yt_disc + EPS))
                        
            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))
            
            # Compute the loss value
            task_loss = self.task_loss_(ys, ys_pred)
            
            task_loss = tf.reduce_mean(task_loss)
            disc_loss = tf.reduce_mean(disc_loss)
            
            enc_loss = task_loss - self.lambda_ * disc_loss
            
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
        logs.update({"disc_loss": disc_loss})
        logs.update(disc_metrics)
        return logs
    
    
    def _get_disc_metrics(self, ys_disc, yt_disc):
        disc_dict = {}
        for m in self.disc_metrics:
            disc_dict["disc_%s"%m.name] = tf.reduce_mean(0.5 * (
                m(tf.ones_like(ys_disc), ys_disc)+
                m(tf.zeros_like(yt_disc), yt_disc)
            ))
        return disc_dict
    
    
    def _initialize_weights(self, shape_X):
        self(np.zeros((1,) + shape_X))
        Xs_enc = self.encoder_(np.zeros((1,) + shape_X), training=True)
        ys_pred = self.task_(Xs_enc, training=True)
        if Xs_enc.get_shape()[1] * ys_pred.get_shape()[1] > self.max_features:
            self.is_overloaded_ = True
            self._random_task = tf.random.normal([ys_pred.get_shape()[1],
                                        self.max_features])
            self._random_enc = tf.random.normal([Xs_enc.get_shape()[1],
                                           self.max_features])
            self.discriminator_(np.zeros((1, self.max_features)))
        else:
            self.is_overloaded_ = False
            self.discriminator_(np.zeros((1, Xs_enc.get_shape()[1] * ys_pred.get_shape()[1])))
    
    
    def _initialize_networks(self):
        if self.encoder is None:
            self.encoder_ = get_default_encoder(name="encoder", state=self.random_state)
        else:
            self.encoder_ = check_network(self.encoder,
                                          copy=self.copy,
                                          name="encoder")
        if self.task is None:
            self.task_ = _get_default_classifier(name="task", state=self.random_state)
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")
        if self.discriminator is None:
            self.discriminator_ = get_default_discriminator(name="discriminator", state=self.random_state)
        else:
            self.discriminator_ = check_network(self.discriminator,
                                                copy=self.copy,
                                                name="discriminator")
        
    
    
    # def _initialize_networks(self, shape_Xt):
        # Call predict to avoid strange behaviour with
        # Sequential model whith unspecified input_shape
        # zeros_enc_ = self.encoder_.predict(np.zeros((1,) + shape_Xt));
        # zeros_task_ = self.task_.predict(zeros_enc_);
        # if zeros_task_.shape[1] * zeros_enc_.shape[1] > self.max_features:
        #     self.discriminator_.predict(np.zeros((1, self.max_features)))
        # else:
        #     zeros_mapping_ = np.matmul(np.expand_dims(zeros_enc_, 2),
        #                                np.expand_dims(zeros_task_, 1))
        #     zeros_mapping_ = np.reshape(zeros_mapping_, (1, -1))
        #     self.discriminator_.predict(zeros_mapping_);
    
    
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
            # X_disc = X_disc.transpose([0, 2, 1])
            X_disc = X_disc.reshape(-1, X_enc.shape[1] * X_task.shape[1])
        y_disc = self.discriminator_.predict(X_disc)
        return y_disc