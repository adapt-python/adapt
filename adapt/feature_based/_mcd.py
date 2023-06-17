"""
DANN
"""

import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import check_network, get_default_encoder, get_default_task

EPS = np.finfo(np.float32).eps


@make_insert_doc(["encoder", "task"])
class MCD(BaseAdaptDeep):
    """
    MCD: Maximum Classifier Discrepancy
    
    MCD is a feature-based domain adaptation method originally introduced
    for unsupervised classification DA.
    
    The goal of MCD is to find a new representation of the input features which
    minimizes the discrepancy between the source and target domains 
    
    The discrepancy is estimated through adversarial training of three networks:
    An encoder and two classifiers. These two learn the task on the source domains
    and are used to compute the discrepancy. A reversal layer is placed between
    the encoder and the two classifiers to perform adversarial training.
    
    Parameters
    ----------       
    pretrain : bool (default=True)
        Weither to pretrain the networks or not.
        If True, the three networks are fitted on source
        labeled data.
        
    n_steps : int (default=1)
        Number of encoder updates for one discriminator update.
    
    Attributes
    ----------
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        Principal task network.
        
    discriminator_ : tensorflow Model
        Secondary task network.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    Examples
    --------
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import MCD
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = MCD(pretrain=True, n_steps=1, Xt=Xt, metrics=["acc"], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 147ms/step - loss: 0.2527 - acc: 0.6200
    0.25267112255096436
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1712.02560.pdf>`_ K. Saito, K. Watanabe, \
Y. Ushiku, and T. Harada. "Maximum  classifier  discrepancy  for  unsupervised  \
domain adaptation". In CVPR, 2018.
    """
    def __init__(self, 
                 encoder=None,
                 task=None,
                 Xt=None,
                 pretrain=True,
                 n_steps=1,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
    
    
    def pretrain_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)

        # loss
        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:                
            # Forward pass
            Xs_enc = self.encoder_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)
            ys_disc = self.discriminator_(Xs_enc, training=True)

            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))
            ys_disc = tf.reshape(ys_disc, tf.shape(ys))

            # Loss
            task_loss = tf.reduce_mean(self.task_loss_(ys, ys_pred))
            disc_loss = tf.reduce_mean(self.task_loss_(ys, ys_disc))
            enc_loss = task_loss + disc_loss
            
            # Compute the loss value
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
        return logs
    
    
    
    def train_step(self, data):
        # Pretrain
        if self.pretrain_:
            return self.pretrain_step(data)
        
        else:
            # Unpack the data.
            Xs, Xt, ys, yt = self._unpack_data(data)
            
            
            for _ in range(self.n_steps-1):
                with tf.GradientTape() as enc_tape:
                    Xt_enc = self.encoder_(Xt, training=True)
                    yt_pred = self.task_(Xt_enc, training=True)
                    yt_disc = self.discriminator_(Xt_enc, training=True)

                    # Reshape
                    yt_pred = tf.reshape(yt_pred, tf.shape(ys))
                    yt_disc = tf.reshape(yt_disc, tf.shape(ys))

                    discrepancy = tf.reduce_mean(tf.abs(yt_pred - yt_disc))
                    enc_loss = discrepancy
                    enc_loss += sum(self.encoder_.losses)
                    
                # Compute gradients
                trainable_vars_enc = self.encoder_.trainable_variables
                gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
                self.optimizer.apply_gradients(zip(gradients_enc, trainable_vars_enc))
            
            # loss
            with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
                # Forward pass
                Xs_enc = self.encoder_(Xs, training=True)
                ys_pred = self.task_(Xs_enc, training=True)
                ys_disc = self.discriminator_(Xs_enc, training=True)

                Xt_enc = self.encoder_(Xt, training=True)
                yt_pred = self.task_(Xt_enc, training=True)
                yt_disc = self.discriminator_(Xt_enc, training=True)

                # Reshape
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                yt_pred = tf.reshape(yt_pred, tf.shape(ys))
                ys_disc = tf.reshape(ys_disc, tf.shape(ys))
                yt_disc = tf.reshape(yt_disc, tf.shape(ys))

                # Compute the loss value
                task_loss = self.task_loss_(ys, ys_pred)
                disc_loss = self.task_loss_(ys, ys_disc)

                task_loss = tf.reduce_mean(task_loss)
                disc_loss = tf.reduce_mean(disc_loss)

                discrepancy = tf.reduce_mean(tf.abs(yt_pred - yt_disc))
                task_loss -= discrepancy
                disc_loss -= discrepancy
                enc_loss = discrepancy

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
            logs.update({"disc_loss": discrepancy})
            return logs
    
    
    def predict_avg(self, X):
        """
        Return the average predictions between
        ``task_`` and ``discriminator_`` networks.
        
        Parameters
        ----------
        X : array
            Input data
            
        Returns
        -------
        y_avg : array
            Average predictions
        """
        ypt = self.task_.predict(self.transform(X))
        ypd = self.discriminator_.predict(self.transform(X))
        yp_avg = 0.5 * (ypt+ypd)
        return yp_avg
        

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
        if self.task is None:
            self.discriminator_ = get_default_task(name="discriminator", state=self.random_state)
        else:
            # Impose Copy, else undesired behaviour
            self.discriminator_ = check_network(self.task,
                                                copy=True, 
                                                name="discriminator")
            
    
    def _initialize_weights(self, shape_X):
        # Init weights encoder
        self(np.zeros((1,) + shape_X))
        X_enc = self.encoder_(np.zeros((1,) + shape_X))
        self.task_(X_enc)
        self.discriminator_(X_enc)
        
        # Add noise to discriminator in order to
        # differentiate from task
        weights = self.discriminator_.get_weights()
        for i in range(len(weights)):
            weights[i] += (0.01 * weights[i] *
                           np.random.standard_normal(weights[i].shape))
        self.discriminator_.set_weights(weights)