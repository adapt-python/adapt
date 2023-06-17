"""
DANN
"""

import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import check_network

EPS = np.finfo(np.float32).eps


# class SetEncoder(tf.keras.callbacks.Callback):
    
#     def __init__(self):
#         self.pretrain = True
    
#     def on_epoch_end(self, epoch, logs=None):
#         if (not logs.get("pretrain")) and self.pretrain:
#             self.pretrain = False
#             self.model.encoder_.set_weights(
#                 self.model.encoder_src_.get_weights())
            


@make_insert_doc(["encoder", "task", "discriminator"])
class ADDA(BaseAdaptDeep):
    """
    ADDA: Adversarial Discriminative Domain Adaptation

    ADDA is a feature-based domain adaptation method.
    
    The purpose of ADDA is to build a new feature representation
    in which source and target data could not be distinguished by
    any **discriminator** network. This feature representation is
    built with two **encoder** networks:
    
    - a **source encoder** trained to provide good features in order
      to learn the task on the source domain. The task is learned
      through a **task** network trained with the **source encoder**.
    - a **target encoder** trained to fool a **discriminator** network
      which tries to classify source and target data in the encoded space.
      The **target encoder** and the **discriminator** are trained
      in an adversarial fashion in the same way as GAN.
      
    The parameters of the four networks are optimized in a two stage
    algorithm where **source encoder** and **task** networks are first
    fitted according to the following optimization problem:
    
    .. math::
    
        \min_{\phi_S, F} \mathcal{L}_{task}(F(\phi_S(X_S)), y_S)
    
    In the second stage, **target encoder** and **discriminator**
    networks are fitted according to:
    
    .. math::
    
        \min_{\phi_T} & \; - \log(D(\phi_T(X_T)))) \\\\
        \min_{D} & \; - \log(D(\phi_S(X_S))) - \log(1 - D(\phi_T(X_T)))
    
    Where:
    
    - :math:`(X_S, y_S), (X_T)` are respectively the labeled source data
      and the unlabeled target data.
    - :math:`\phi_S, \phi_T, F, D` are respectively the **source encoder**,
      the **target encoder**, the **task** and the **discriminator** networks.
    
    The method has been originally introduced for **unsupervised**
    classification DA but it could be widen to other task in **supervised**
    DA straightforwardly.
    
    .. figure:: ../_static/images/adda.png
        :align: center
        
        Overview of the ADDA approach (source: [1])
    
    Parameters
    ----------
    pretrain : bool (default=True)
        Weither to perform pretraining of the ``encoder_src_``
        and ``task_`` networks on source data or not.
        separated compile and fit arguments for the
        pretraining can be given by using the prefix
        ``pretrain__`` as ``pretrain__epochs=10`` or
        ``pretrain__learning_rate=0.1`` for instance.
        If no pretrain arguments are given, the training
        arguments are used by default
        
    tol : float (default=0.001)
        Tolerance on the loss for early stopping of 
        pretraining.
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        task network.
        
    discriminator_ : tensorflow Model
        discriminator network.
        
    encoder_src_ : tensorflow Model
        Source encoder network
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import ADDA
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = ADDA(Xt=Xt, metrics=["acc"], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 153ms/step - loss: 0.0960 - acc: 0.9300
    0.09596743434667587
    
    
    See also
    --------
    DANN
    DeepCORAL
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1702.05464.pdf>`_ E. Tzeng, J. Hoffman, \
K. Saenko, and T. Darrell. "Adversarial discriminative domain adaptation". \
In CVPR, 2017.
    """
    def __init__(self,
                 encoder=None,
                 task=None,
                 discriminator=None,
                 Xt=None,
                 pretrain=True,
                 tol=0.001,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
    
    def _initialize_pretain_networks(self):
        self.encoder_.set_weights(
        self.encoder_src_.get_weights())
    
    
    def pretrain_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)

        # loss
        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:                       
            # Forward pass
            Xs_enc = self.encoder_src_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)

            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))

            # Compute the loss value
            loss = tf.reduce_mean(self.task_loss_(ys, ys_pred))
            task_loss = loss + sum(self.task_.losses)
            enc_loss = loss + sum(self.encoder_src_.losses)
            
        # Compute gradients
        trainable_vars_task = self.task_.trainable_variables
        trainable_vars_enc = self.encoder_src_.trainable_variables

        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))

        # Update metrics
        self.compiled_metrics.update_state(ys, ys_pred)
        self.compiled_loss(ys, ys_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        return logs
    
    
    def train_step(self, data):
        # Pretrain
        if self.pretrain_:
            return self.pretrain_step(data)
        
        else:
            # Unpack the data.
            Xs, Xt, ys, yt = self._unpack_data(data)
            
            # loss
            with tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:                       
                # Forward pass
                if self.pretrain:
                    Xs_enc = self.encoder_src_(Xs, training=False)
                else:
                    # encoder src is not needed if pretrain=False
                    Xs_enc = Xs
                    
                ys_disc = self.discriminator_(Xs_enc, training=True)

                Xt_enc = self.encoder_(Xt, training=True)
                yt_disc = self.discriminator_(Xt_enc, training=True)

                # Compute the loss value
                disc_loss = (-tf.math.log(ys_disc + EPS)
                             -tf.math.log(1-yt_disc + EPS))

                enc_loss = -tf.math.log(yt_disc + EPS)

                disc_loss = tf.reduce_mean(disc_loss)
                enc_loss = tf.reduce_mean(enc_loss)

                disc_loss += sum(self.discriminator_.losses)
                enc_loss += sum(self.encoder_.losses)

            # Compute gradients
            trainable_vars_enc = self.encoder_.trainable_variables
            trainable_vars_disc = self.discriminator_.trainable_variables

            gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
            gradients_disc = disc_tape.gradient(disc_loss, trainable_vars_disc)

            # Update weights
            self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))
            self.optimizer_disc.apply_gradients(zip(gradients_disc, trainable_vars_disc))

            # Update metrics
            # self.compiled_metrics.update_state(ys, ys_pred)
            # self.compiled_loss(ys, ys_pred)
            # Return a dict mapping metric names to current value
            # logs = {m.name: m.result() for m in self.metrics}
            logs = self._get_disc_metrics(ys_disc, yt_disc)
            return logs
    
    
    def _get_disc_metrics(self, ys_disc, yt_disc):
        disc_dict = {}
        disc_dict["disc_loss"] = tf.reduce_mean(
            (-tf.math.log(ys_disc + EPS)
             -tf.math.log(1-yt_disc + EPS))
        )
        for m in self.disc_metrics:
            disc_dict["disc_%s"%m.name] = tf.reduce_mean(0.5 * (
                m(tf.ones_like(ys_disc), ys_disc)+
                m(tf.zeros_like(yt_disc), yt_disc)
            ))
        return disc_dict
    
    
    def _initialize_weights(self, shape_X):
        # Init weights encoder
        self(np.zeros((1,) + shape_X))
        
        # Set same weights to encoder_src
        if self.pretrain:
            # encoder src is not needed if pretrain=False
            self.encoder_(np.zeros((1,) + shape_X))
            self.encoder_src_ = check_network(self.encoder_,
                                              copy=True,
                                              name="encoder_src")
        
        
    def transform(self, X, domain="tgt"):
        """
        Return the encoded features of X.
        
        Parameters
        ----------
        X : array
            input data
            
        domain: str (default="tgt")
            If domain is ``"tgt"`` or ``"target"``,
            the target encoder is used.
            If domain is ``"src"`` or ``"source"``,
            the source encoder is used.
            
        Returns
        -------
        X_enc : array
            predictions of encoder network
        """
        if domain in ["tgt", "target"]:
            return self.encoder_.predict(X)
        elif domain in ["src", "source"]:
            return self.encoder_src_.predict(X)
        else:
            raise ValueError("`domain `argument "
                             "should be `tgt` or `src`, "
                             "got, %s"%domain)
    
    
    def predict_disc(self, X, domain="tgt"):
        """
        Return predictions of the discriminator on the encoded features.
        
        Parameters
        ----------
        X : array
            input data
            
        domain: str (default="tgt")
            If domain is ``"tgt"`` or ``"target"``,
            the target encoder is used.
            If domain is ``"src"`` or ``"source"``,
            the source encoder is used.
            
        Returns
        -------
        y_disc : array
            predictions of discriminator network
        """     
        return self.discriminator_.predict(self.transform(X, domain=domain))
    
    
    def predict_task(self, X, domain="tgt"):
        """
        Return predictions of the task on the encoded features.
        
        Parameters
        ----------
        X : array
            input data
            
        domain: str (default="tgt")
            If domain is ``"tgt"`` or ``"target"``,
            the target encoder is used.
            If domain is ``"src"`` or ``"source"``,
            the source encoder is used.
            
        Returns
        -------
        y_task : array
            predictions of task network
        """     
        return self.task_.predict(self.transform(X, domain=domain))
        