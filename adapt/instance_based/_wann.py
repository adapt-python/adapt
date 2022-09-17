"""
Weighting Adversarial Neural Network (WANN)
"""
import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import check_network, get_default_task

EPS = np.finfo(np.float32).eps


@make_insert_doc(["task", "weighter"], supervised=True)
class WANN(BaseAdaptDeep):
    """
    WANN : Weighting Adversarial Neural Network
    
    WANN is an instance-based domain adaptation method suited for regression tasks.
    It supposes the supervised setting where some labeled target data are available.
    
    The goal of WANN is to compute a source instances reweighting which correct
    "shifts" between source and target domain. This is done by minimizing the
    Y-discrepancy distance between source and target distributions
    
    WANN involves three networks:
    
    - the weighting network which learns the source weights.
    - the task network which learns the task.
    - the discrepancy network which is used to estimate a distance 
      between the reweighted source and target distributions: the Y-discrepancy
    
    Parameters
    ----------
    pretrain : bool (default=True)
        Weither to perform pretraining of the ``weighter``
        network or not. If True, the ``weighter`` is 
        pretrained in order to predict 1 for each source.
    
    C : float (default=1.)
        Clipping constant for the weighting networks
        regularization. Low value of ``C`` produce smoother
        weighting map. If ``C<=0``, No regularization is added.
        
    Attributes
    ----------
    weighter_ : tensorflow Model
        weighting network.
        
    task_ : tensorflow Model
        task network.
        
    discriminator_ : tensorflow Model
        discriminator network.
        
    history_ : dict
        history of the losses and metrics across the epochs.

    Examples
    --------
    >>> from adapt.utils import make_regression_da
    >>> from adapt.instance_based import WANN
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> model = WANN(Xt=Xt[:10], yt=yt[:10], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 106ms/step - loss: 0.1096
    0.10955706238746643

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/2006.08251.pdf>`_ A. de Mathelin, \
G. Richard, F. Deheeger, M. Mougeot and N. Vayatis  "Adversarial Weighting \
for Domain Adaptation in Regression". In ICTAI, 2021.
    """
    
    def __init__(self,
                 task=None,
                 weighter=None,
                 Xt=None,
                 yt=None,
                 pretrain=True,
                 C=1.,
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
                
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
    
    def _initialize_networks(self):
        if self.weighter is None:
            self.weighter_ = get_default_task(name="weighter", state=self.random_state)
            if self.C > 0.:
                self.weighter_ = self._add_regularization(self.weighter_)
        else:
            if self.C > 0.:
                self.weighter_ = self._add_regularization(self.weighter)
            self.weighter_ = check_network(self.weighter,
                                          copy=self.copy,
                                          name="weighter")
        if self.task is None:
            self.task_ = get_default_task(name="task", state=self.random_state)
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")
        if self.task is None:
            self.discriminator_ = get_default_task(name="discriminator", state=self.random_state)
        else:
            self.discriminator_ = check_network(self.task,
                                                copy=self.copy,
                                                name="discriminator")


    def _add_regularization(self, weighter):                
        for i in range(len(weighter.layers)):
            if hasattr(weighter.layers[i], "kernel_constraint"):
                setattr(weighter.layers[i],
                        "kernel_constraint",
                        tf.keras.constraints.MaxNorm(self.C))
            if hasattr(weighter.layers[i], "bias_constraint"):
                setattr(weighter.layers[i],
                        "bias_constraint",
                        tf.keras.constraints.MaxNorm(self.C))
        return weighter
        
    
    def pretrain_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)

        # loss
        with tf.GradientTape() as tape:                       
            # Forward pass
            weights = tf.math.abs(self.weighter_(Xs, training=True))
            
            loss = tf.reduce_mean(
                tf.square(weights - tf.ones_like(weights)))
            
            # Compute the loss value
            loss += sum(self.weighter_.losses)
            
        # Compute gradients
        trainable_vars = self.weighter_.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        logs = {"loss": loss}
        return logs
        
    
    def call(self, X):
        return self.task_(X)
    
    
    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
        
        if self.pretrain_:
            return self.pretrain_step(data)
        
        else:
            # loss
            with tf.GradientTape() as task_tape, tf.GradientTape() as weight_tape, tf.GradientTape() as disc_tape:

                # Forward pass
                weights = tf.abs(self.weighter_(Xs, training=True))
                ys_pred = self.task_(Xs, training=True)
                ys_disc = self.discriminator_(Xs, training=True)

                yt_pred = self.task_(Xt, training=True)
                yt_disc = self.discriminator_(Xt, training=True)

                # Reshape
                ys_pred = tf.reshape(ys_pred, tf.shape(ys))
                ys_disc = tf.reshape(ys_disc, tf.shape(ys))
                yt_pred = tf.reshape(yt_pred, tf.shape(yt))
                yt_disc = tf.reshape(yt_disc, tf.shape(yt))

                # Compute the loss value
                task_loss = self.task_loss_(ys, ys_pred)
                disc_src = self.task_loss_(ys, ys_disc)
                disc_tgt = self.task_loss_(yt, yt_disc)

                weights = tf.reshape(weights, tf.shape(task_loss))

                task_loss = weights * task_loss
                disc_src = weights * disc_src

                task_loss = tf.reduce_mean(task_loss)
                disc_src = tf.reduce_mean(disc_src)
                disc_tgt = tf.reduce_mean(disc_tgt)

                disc_loss = disc_src - disc_tgt

                weight_loss = task_loss - disc_loss

                task_loss += sum(self.task_.losses)
                disc_loss += sum(self.discriminator_.losses)
                weight_loss += sum(self.weighter_.losses)

            # Compute gradients
            trainable_vars_task = self.task_.trainable_variables
            trainable_vars_weight = self.weighter_.trainable_variables
            trainable_vars_disc = self.discriminator_.trainable_variables

            gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
            gradients_weight = weight_tape.gradient(weight_loss, trainable_vars_weight)
            gradients_disc = disc_tape.gradient(disc_loss, trainable_vars_disc)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
            self.optimizer.apply_gradients(zip(gradients_weight, trainable_vars_weight))
            self.optimizer.apply_gradients(zip(gradients_disc, trainable_vars_disc))

            # Update metrics
            self.compiled_metrics.update_state(ys, ys_pred)
            self.compiled_loss(ys, ys_pred)
            # Return a dict mapping metric names to current value
            logs = {m.name: m.result() for m in self.metrics}
            return logs
    
    
    def predict_weights(self, X):
        """
        Return the predictions of weighting network
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        array:
            weights
        """
        return np.abs(self.weighter_.predict(X))
    
    
    def predict_disc(self, X):
        """
        Return predictions of the discriminator.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_disc : array
            predictions of discriminator network
        """
        return self.discriminator_.predict(X)