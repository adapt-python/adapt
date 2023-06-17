"""
Importance Weighting Network (IWN)
"""
import warnings
import inspect
from copy import deepcopy

import numpy as np
from sklearn.utils import check_array
import tensorflow as tf
from tensorflow.keras import Model

from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import (check_arrays, check_network, get_default_task,
                         set_random_seed, check_estimator, check_sample_weight)

EPS = np.finfo(np.float32).eps


def pairwise_euclidean(X, Y):
    X2 = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), [1, tf.shape(Y)[0]])
    Y2 = tf.tile(tf.reduce_sum(tf.square(Y), axis=1, keepdims=True), [1, tf.shape(X)[0]])
    XY = tf.matmul(X, tf.transpose(Y))
    return X2 + tf.transpose(Y2) - 2*XY


def weighted_mmd(Xs, Xt, weights, gamma=1.):
    n = tf.cast(tf.shape(Xs)[0], Xs.dtype)
    m = tf.cast(tf.shape(Xt)[0], Xt.dtype)
    gamma = tf.cast(gamma, Xt.dtype)
    
    weights = tf.cast(weights, Xt.dtype)
    weights = tf.reshape(weights, (-1, 1))
    weights /= (tf.reduce_mean(weights))
    
    Wij = tf.matmul(weights, tf.reshape(weights, (1, -1)))
    
    Mxx = Wij * tf.exp(-gamma * pairwise_euclidean(Xs, Xs))
    mxx = tf.reduce_mean(Mxx)
    
    Myy = tf.exp(-gamma * pairwise_euclidean(Xt, Xt))
    myy = tf.reduce_mean(Myy)
    
    Mxy = weights * tf.exp(-gamma * pairwise_euclidean(Xs, Xt))
    mxy = tf.reduce_mean(Mxy)
    
    return mxx + myy -2*mxy


@make_insert_doc(["estimator", "weighter"])
class IWN(BaseAdaptDeep):
    """
    IWN : Importance Weighting Network
    
    IWN is an instance-based method for unsupervised domain adaptation.
    
    The goal of IWN is to reweight the source instances in order to
    minimize the Maximum Mean Discreancy (MMD) between the reweighted
    source and the target distributions.
    
    IWN uses a weighting network to parameterized the weights of the
    source instances. The MMD is computed with gaussian kernels
    parameterized by the bandwidth :math:`\sigma`. The :math:`\sigma`
    parameter is updated during the IWN optimization in order to
    maximize the discriminative power of the MMD.
    
    Parameters
    ----------
    pretrain : bool (default=True)
        Weither to perform pretraining of the ``weighter``
        network or not. If True, the ``weighter`` is 
        pretrained in order to predict 1 for each source data.
    
    sigma_init : float (default=.1)
        Initialization for the kernel bandwidth
        
    update_sigma : bool (default=True)
        Weither to update the kernel bandwidth or not.
        If `False`, the bandwidth stay equal to `sigma_init`.
        
    Attributes
    ----------
    weighter_ : tensorflow Model
        weighting network.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        
    sigma_ : tf.Variable
        fitted kernel bandwidth.
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import IWN
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = IWN(RidgeClassifier(0.), Xt=Xt, sigma_init=0.1, random_state=0,
    ...             pretrain=True, pretrain__epochs=100, pretrain__verbose=0)
    >>> model.fit(Xs, ys, epochs=100, batch_size=256, verbose=1)
    >>> model.score(Xt, yt)
    0.78
    
    See also
    --------
    KMM
    WANN
    
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/2209.04215.pdf>`_ A. de Mathelin, F. Deheeger, \
M. Mougeot and N. Vayatis "Fast and Accurate Importance Weighting for \
Correcting Sample Bias" In ECML-PKDD, 2022.
    """
    
    def __init__(self,
                 estimator=None,
                 weighter=None,
                 Xt=None,
                 yt=None,
                 pretrain=True,
                 sigma_init=.1,
                 update_sigma=True,
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
        else:
            self.weighter_ = check_network(self.weighter,
                                          copy=self.copy,
                                          name="weighter")
        self.sigma_ = tf.Variable(self.sigma_init,
                                  trainable=self.update_sigma)
        
    
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
        return self.weighter_(X)
    
    
    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
        
        if self.pretrain_:
            return self.pretrain_step(data)
        
        else:
            # loss
            with tf.GradientTape() as weight_tape, tf.GradientTape() as sigma_tape:

                # Forward pass
                weights = tf.abs(self.weighter_(Xs, training=True))
                
                loss = weighted_mmd(Xs, Xt, weights, self.sigma_)
                loss_sigma = -loss

                loss += sum(self.weighter_.losses)

            # Compute gradients
            trainable_vars = self.weighter_.trainable_variables
            
            gradients = weight_tape.gradient(loss, trainable_vars)
            gradients_sigma = sigma_tape.gradient(loss_sigma, [self.sigma_])
            
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.optimizer.apply_gradients(zip(gradients_sigma, [self.sigma_]))

            # Return a dict mapping metric names to current value
            logs = {"loss": loss, "sigma": self.sigma_}
            return logs
            
    
    def fit(self, X, y=None, Xt=None, yt=None, domains=None,
            fit_params_estimator={}, **fit_params):
        weights = self.fit_weights(X, Xt, **fit_params)
        self.fit_estimator(X, y, sample_weight=weights, **fit_params_estimator)
        return self
        
        
    def fit_weights(self, Xs, Xt, **fit_params):
        """
        Fit importance weighting.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        fit_params : key, value arguments
            Arguments given to the fit method of the model
            (epochs, batch_size, callbacks...).
            
        Returns
        -------
        weights_ : sample weights
        """
        super().fit(Xs, np.zeros(len(Xs)), Xt, None, None, **fit_params)
        return self.predict_weights(Xs)
    
    
    def fit_estimator(self, X, y, sample_weight=None,
                      random_state=None, warm_start=True,
                      **fit_params):
        """
        Fit estimator on X, y.
        
        Parameters
        ----------
        X : array
            Input data.
            
        y : array
            Output data.
            
        sample_weight : array
            Importance weighting.
            
        random_state : int (default=None)
            Seed of the random generator
            
        warm_start : bool (default=True)
            If True, continue to fit ``estimator_``,
            else, a new estimator is fitted based on
            a copy of ``estimator``. (Be sure to set
            ``copy=True`` to use ``warm_start=False``)
            
        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator and to the compile method
            for tensorflow estimator.
            
        Returns
        -------
        estimator_ : fitted estimator
        """
        X, y = check_arrays(X, y, accept_sparse=True)
        set_random_seed(random_state)

        if (not warm_start) or (not hasattr(self, "estimator_")):
            estimator = self.estimator
            self.estimator_ = check_estimator(estimator,
                                              copy=self.copy,
                                              force_copy=True)
            if isinstance(self.estimator_, Model):
                compile_params = {}
                if estimator._is_compiled:
                    compile_params["loss"] = deepcopy(estimator.loss)
                    compile_params["optimizer"] = deepcopy(estimator.optimizer)
                else:
                    raise ValueError("The given `estimator` argument"
                                     " is not compiled yet. "
                                     "Please give a compiled estimator or "
                                     "give a `loss` and `optimizer` arguments.")
                self.estimator_.compile(**compile_params)

        fit_args = [
            p.name
            for p in inspect.signature(self.estimator_.fit).parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        if "sample_weight" in fit_args:
            sample_weight = check_sample_weight(sample_weight, X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator_.fit(X, y,
                                    sample_weight=sample_weight,
                                    **fit_params)
        else:
            if sample_weight is None:
                self.estimator_.fit(X, y, **fit_params)
            else:
                sample_weight = check_sample_weight(sample_weight, X)
                sample_weight /= sample_weight.sum()
                bootstrap_index = np.random.choice(
                len(X), size=len(X), replace=True,
                p=sample_weight)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.estimator_.fit(X[bootstrap_index],
                                        y[bootstrap_index],
                                        **fit_params)
        return self.estimator_

    
    
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
        return np.abs(self.weighter_.predict(X)).ravel()
    

    def predict(self, X, domain=None, **predict_params):
        """
        Return estimator predictions
        
        Parameters
        ----------
        X : array
            input data
        
        domain : str (default=None)
            Not used. For compatibility with `adapt` objects
        
        Returns
        -------
        y_pred : array
            prediction of the Adapt Model.
        """
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        return self.estimator_.predict(X, **predict_params)


    def score(self, X, y, sample_weight=None, domain=None):
        """
        Return the estimator score.
        
        Call `score` on sklearn estimator and 
        `evaluate` on tensorflow Model.
        
        Parameters
        ----------
        X : array
            input data
            
        y : array
            output data
            
        sample_weight : array (default=None)
            Sample weights
             
        domain : str (default=None)
            Not used.
            
        Returns
        -------
        score : float
            estimator score.
        """
        X, y = check_arrays(X, y, accept_sparse=True)
        
        if hasattr(self.estimator_, "score"):
            score = self.estimator_.score(X, y, sample_weight)
        elif hasattr(self.estimator_, "evaluate"):
            if np.prod(X.shape) <= 10**8:
                score = self.estimator_.evaluate(
                    X, y,
                    sample_weight=sample_weight,
                    batch_size=len(X)
                )
            else:
                score = self.estimator_.evaluate(
                    X, y,
                    sample_weight=sample_weight
                )
            if isinstance(score, (tuple, list)):
                score = score[0]
        else:
            raise ValueError("Estimator does not implement"
                             " score or evaluate method")
        return score