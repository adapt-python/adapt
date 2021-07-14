"""
Regular Transfer
"""

import copy
import warnings

import numpy as np
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  LogisticRegression,
                                  RidgeClassifier)
from sklearn.exceptions import NotFittedError
from scipy.optimize import minimize
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

from adapt.utils import (check_arrays,
                         check_one_array,
                         check_estimator,
                         check_network)


def get_zeros_network():
    """
    Return a tensorflow Model of two hidden layers
    with 10 neurons each and relu activations. The
    last layer is composed of one neuron with linear
    activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu",
                    kernel_initializer="zeros",
                    bias_initializer="zeros"))
    model.add(Dense(10, activation="relu",
                    kernel_initializer="zeros",
                    bias_initializer="zeros"))
    model.add(Dense(1, activation=None,
                    kernel_initializer="zeros",
                    bias_initializer="zeros"))
    return model


class RegularTransferLR:
    """
    Regular Transfer with Linear Regression
    
    RegularTransferLR is a parameter-based domain adaptation method.
    
    The method is based on the assumption that a good target estimator
    can be obtained by adapting the parameters of a pre-trained source
    estimator using a few labeled target data.
    
    The approach consist in fitting a linear estimator on target data
    according to an objective function regularized by the euclidean
    distance between source and target parameters:
    
    .. math::
    
        \\beta_T = \\underset{\\beta \in \\mathbb{R}^p}{\\text{argmin}}
        \\, ||X_T\\beta - y_T||^2 + \\lambda ||\\beta - \\beta_S||^2
        
    Where:
    
    - :math:`\\beta_T` are the target model parameters.
    - :math:`\\beta_S = \\underset{\\beta \\in \\mathbb{R}^p}{\\text{argmin}}
      \\, ||X_S\\beta - y_S||^2` are the source model parameters.
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the source and
      the target labeled data.
    - :math:`p` is the number of features in :math:`X_T`
      (:math:`+1` if ``intercept`` is True).
    - :math:`\\lambda` is a trade-off parameter.

    Parameters
    ----------
    estimator : sklearn LinearRegression or Ridge instance
        Estimator used to learn the task.
        
    lambda_ : float (default=1.0)
        Trade-Off parameter.
        
    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
            
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.

    Attributes
    ----------
    estimator_ : instance of LinearRegression or Ridge
        Estimator.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.parameter_based import RegularTransferLR
    >>> from sklearn.linear_model import LinearRegression
    >>> np.random.seed(0)
    >>> Xs = np.random.randn(50) * 0.1
    >>> Xs = np.concatenate((Xs, Xs + 1.))
    >>> Xt = np.random.randn(100) * 0.1
    >>> ys = (np.array([-0.2 * x if x<0.5 else 1. for x in Xs])
    ...       + 0.1 * np.random.randn(100))
    >>> yt = 0.75 * Xt + 0.1 * np.random.randn(100)
    >>> lr = LinearRegression()
    >>> lr.fit(Xs.reshape(-1, 1), ys)
    >>> lr.score(Xt.reshape(-1, 1), yt)
    0.2912...
    >>> rt = RegularTransferLR(lr, lambda_=0.01, random_state=0)
    >>> rt.fit(Xt[:10], yt[:10])
    >>> rt.estimator_.score(Xt.reshape(-1, 1), yt)
    0.3276...
        
    See also
    --------
    RegularTransferLC, RegularTransferNN

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    def __init__(self,
                 estimator,
                 lambda_=1.0,
                 copy=True,
                 random_state=None):
        
        if (not isinstance(estimator, LinearRegression) and
            not isinstance(estimator, Ridge)):
            raise ValueError("`estimator` argument should be a"
                             " `LinearRegression` or `Ridge`"
                             " instance.")
        
        if not hasattr(estimator, "coef_"):
            raise NotFittedError("`estimator` argument is not fitted yet, "
                                 "please call `fit` on `estimator`.")

        self.estimator_ = check_estimator(estimator,
                                          copy=copy)
        self.lambda_ = lambda_
        self.copy = copy
        self.random_state = random_state

    
    def fit(self, Xt, yt, **fit_params):
        """
        Fit RegularTransferLR.

        Parameters
        ----------
        Xt : numpy array
            Target input data.

        yt : numpy array
            Target output data.

        Returns
        -------
        self : returns an instance of self
        """        
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        Xt = check_one_array(Xt)
        yt = check_one_array(yt)
                
        if self.estimator_.fit_intercept:
            beta_src = np.concatenate((
                np.array([self.estimator_.intercept_]),
                self.estimator_.coef_
            ))
            Xt = np.concatenate(
                (np.ones((len(Xt), 1)), Xt),
                axis=-1)
        else:
            beta_src = self.estimator_.coef_
        
        def func(beta):
            return (np.linalg.norm(Xt.dot(beta.reshape(-1, 1)) - yt) ** 2 +
                    self.lambda_ * np.linalg.norm(beta - beta_src) ** 2)

        beta_tgt = minimize(func, beta_src)['x']

        if self.estimator_.fit_intercept:
            self.estimator_.intercept_ = beta_tgt[[0]]
            self.estimator_.coef_ = np.array([beta_tgt[1:]])
        else:
            self.estimator_.intercept_ = np.array([0.])
            self.estimator_.coef_ = np.array([beta_tgt])
        return self


    def predict(self, X):
        """
        Return the predictions of the target estimator.

        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Prediction of target estimator.
        """
        X = check_one_array(X)
        return self.estimator_.predict(X)
    
    
    
class RegularTransferLC:
    """
    Regular Transfer for Binary Linear Classification
    
    RegularTransferLC is a parameter-based domain adaptation method.
    
    The method is based on the assumption that a good target estimator
    can be obtained by adapting the parameters of a pre-trained source
    estimator using a few labeled target data.
    
    The approach consist in fitting a linear estimator on target data
    according to an objective function regularized by the euclidean
    distance between source and target parameters:
    
    .. math::
    
        \\beta_T = \\underset{\\beta \\in \\mathbb{R}^p}{\\text{argmin}}
        \\, \\ell(\\beta, X_T, y_T) + \\lambda ||\\beta - \\beta_S||^2
        
    Where:
    
    - :math:`\\ell` is the log-likelihood function.
    - :math:`\\beta_T` are the target model parameters.
    - :math:`\\beta_S = \\underset{\\beta \\in \\mathbb{R}^p}{\\text{argmin}}
      \\, \\ell(\\beta, X_S, y_S)` are the source model parameters.
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the source and
      the target labeled data.
    - :math:`p` is the number of features in :math:`X_T`
      (:math:`+1` if ``intercept`` is True).
    - :math:`\\lambda` is a trade-off parameter.

    Parameters
    ----------
    estimator : sklearn LogisticRegression or RidgeClassifier instance
        Estimator used to learn the task.
        
    lambda_ : float (default=1.0)
        Trade-Off parameter.
        
    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
            
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.

    Attributes
    ----------
    estimator_ : instance of LogisticRegression or RidgeClassifier
        Estimator.
            
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.parameter_based import RegularTransferLC
    >>> from sklearn.linear_model import LogisticRegression
    >>> np.random.seed(0)
    >>> Xs = np.random.randn(50) * 0.1
    >>> Xs = np.concatenate((Xs, Xs + 1.))
    >>> Xt = np.random.randn(100) * 0.1
    >>> ys = (Xs < 0.1).astype(int)
    >>> yt = (Xt < 0.05).astype(int)
    >>> lc = LogisticRegression()
    >>> lc.fit(Xs.reshape(-1, 1), ys)
    >>> lc.score(Xt.reshape(-1, 1), yt)
    0.67
    >>> rt = RegularTransferLC(lc, lambda_=0.01, random_state=0)
    >>> rt.fit(Xt[:10], yt[:10])
    >>> rt.estimator_.score(Xt.reshape(-1, 1), yt)
    0.67

    See also
    --------
    RegularTransferLR, RegularTransferNN

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    def __init__(self,
                 estimator,
                 lambda_=1.0,
                 copy=True,
                 random_state=None):
        
        if (not isinstance(estimator, LogisticRegression) and
            not isinstance(estimator, RidgeClassifier)):
            raise ValueError("`estimator` argument should be a"
                             " `LogisticRegression` or `RidgeClassifier`"
                             " instance.")
        
        if not hasattr(estimator, "coef_"):
            raise NotFittedError("`estimator` argument is not fitted yet, "
                                 "please call `fit` on `estimator`.")

        self.estimator_ = check_estimator(estimator,
                                          copy=copy)
        self.lambda_ = lambda_
        self.copy = copy
        self.random_state = random_state


    def fit(self, Xt, yt, **fit_params):
        """
        Fit RegularTransferLC.

        Parameters
        ----------
        Xt : numpy array
            Target input data.

        yt : numpy array
            Target output data.

        Returns
        -------
        self : returns an instance of self
        """        
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        Xt = check_one_array(Xt)
        yt = check_one_array(yt)
                
        if self.estimator_.fit_intercept:
            beta_src = np.concatenate((
                np.array([self.estimator_.intercept_[0]]),
                self.estimator_.coef_[0]
            ))
            Xt = np.concatenate(
                (np.ones((len(Xt), 1)), Xt),
                axis=-1)
        else:
            beta_src = self.estimator_.coef_[0]
        
        def func(beta):
            return (np.sum(np.log(1 + np.exp(
                    -(2*yt-1) * Xt.dot(beta.reshape(-1, 1))))) +
                    self.lambda_ * np.linalg.norm(beta - beta_src) ** 2)

        beta_tgt = minimize(func, beta_src)['x']

        if self.estimator_.fit_intercept:
            self.estimator_.intercept_ = beta_tgt[[0]]
            self.estimator_.coef_ = np.array([beta_tgt[1:]])
        else:
            self.estimator_.intercept_ = np.array([0.])
            self.estimator_.coef_ = np.array([beta_tgt])
        return self


    def predict(self, X):
        """
        Return the predictions of the target estimator.

        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Prediction of target estimator.
        """
        X = check_one_array(X)
        return self.estimator_.predict(X)


class RegularTransferNN:
    """
    Regular Transfer with Neural Network
    
    RegularTransferNN is a parameter-based domain adaptation method.
    
    The method is based on the assumption that a good target estimator
    can be obtained by adapting the parameters of a pre-trained source
    estimator using a few labeled target data.
    
    The approach consist in fitting a neural network on target data
    according to an objective function regularized by the euclidean
    distance between source and target parameters:
    
    .. math::
    
        \\beta_T = \\underset{\\beta=(\\beta_1, ... , \\beta_D)}{\\text{argmin}}
        \\, ||f(X_T, \\beta) - y_T||^2 + \sum_{i=1}^{D}
        \\lambda_i ||\\beta_i - {\\beta_S}_i||^2
        
    Where:
    
    - :math:`f` is a neural network with :math:`D` layers.
    - :math:`\\beta_T` are the parameters of the target neural network.
    - :math:`\\beta_S = \\underset{\\beta}{\\text{argmin}}
      \\, ||f(X_S,\\beta) - y_S||^2` are the source neural network parameters.
    - :math:`(X_S, y_S), (X_T, y_T)` are respectively the source and
      the target labeled data.
    - :math:`\\lambda_i` is the trade-off parameter of layer :math:`i`.
    
    Different trade-off can be given to the layer of the 
    neural network through the ``lambdas`` parameter.
    Some layers can also be frozen during training via
    the ``training`` parameter.
    
    .. figure:: ../_static/images/regulartransfer.png
        :align: center
        
        Transferring parameters of a CNN pretrained on Imagenet
        (source: [2])

    Parameters
    ----------
    network : tensorflow Model (default=None)
        Base netwok. If ``None``, a neural network with two
        hidden layers of 10 neurons with ReLU activation each
        is used and all weights initialized to zeros.
        
    lambdas : float or list of float, optional (default=1.0)
        Trade-off parameters.
        If a list is given, values from ``lambdas`` are assigned
        successively to the list of ``network`` layers with 
        weights parameters going from the last layer to the first one.
        If the length of ``lambdas`` is smaller than the length of
        ``network`` layers list, the last trade-off value will be
        asigned to the remaining layers.
        
    trainables : boolean or list of boolean, optional (default=True)
        Whether to train the layer or not.
        If a list is given, values from ``trainables`` are assigned
        successively to the list of ``network`` layers with 
        weights parameters going from the last layer to the first one.
        If the length of ``trainables`` is smaller than the length of
        ``network`` layers list, the last trade-off value will be
        asigned to the remaining layers.
        
    regularizer : str (default="l2")
        Regularizing function used. Possible values:
        [`l2`, `l1`]
        
    copy : boolean (default=True)
        Whether to make a copy of ``network`` or not.
        
    random_state : int (default=None)
        Seed of random generator.
    
    compil_params : key, value arguments, optional
        Additional arguments for autoencoder compiler
        (loss, optimizer...).
        If none, loss is set to ``"mean_squared_error"``
        and optimizer to ``Adam(0.001)``.

    Attributes
    ----------
    network_ : tensorflow Model
        Network.
        
    history_ : dict
        history of the losses and metrics across the epochs
        of the network training.
        
    Examples
    --------
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from adapt.parameter_based import RegularTransferNN
    >>> np.random.seed(0)
    >>> tf.random.set_seed(0)
    >>> Xs = np.random.randn(50) * 0.1
    >>> Xs = np.concatenate((Xs, Xs + 1.))
    >>> Xt = np.random.randn(100) * 0.1
    >>> ys = (np.array([-0.2 * x if x<0.5 else 1. for x in Xs])
    ...       + 0.1 * np.random.randn(100))
    >>> yt = 0.75 * Xt + 0.1 * np.random.randn(100)
    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Dense(1))
    >>> model.compile(optimizer="adam", loss="mse")
    >>> model.predict(Xt.reshape(-1,1))
    >>> model.fit(Xs.reshape(-1, 1), ys, epochs=300, verbose=0)
    >>> np.abs(model.predict(Xt).ravel() - yt).mean()
    0.48265...
    >>> rt = RegularTransferNN(model, lambdas=0.01, random_state=0)
    >>> rt.fit(Xt[:10], yt[:10], epochs=300, verbose=0)
    >>> rt.predict(Xt.reshape(-1, 1))
    >>> np.abs(rt.predict(Xt).ravel() - yt).mean()
    0.23114...
        
    See also
    --------
    RegularTransferLR, RegularTransferLC

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.

    .. [2] `[2] <https://hal.inria.fr/hal-00911179v1/document>`_ \
Oquab M., Bottou L., Laptev I., Sivic J. "Learning and \
transferring mid-level image representations using convolutional \
neural networks". In CVPR, 2014.
    """
    def __init__(self,
                 network=None,
                 lambdas=1.0,
                 trainables=True,
                 regularizer="l2",
                 copy=True,
                 random_state=None,
                 **compil_params):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        if network is None:
            network = get_zeros_network()
        
        self.network_ = check_network(network, copy=copy,
                                           compile_=False)
        self.lambdas = lambdas
        self.trainables = trainables
        self.copy = copy
        self.random_state = random_state
        self.compil_params = compil_params
        self.is_built_ = False
        
        if regularizer in ["l1", "l2"]:
            self.regularizer = regularizer
        else:
            raise ValueError("`regularizer` argument should be "
                             "'l1' or 'l2', got, %s"%str(regularizer))
        
        if not hasattr(self.lambdas, "__iter__"):
            self.lambdas = [self.lambdas]
        if not hasattr(self.trainables, "__iter__"):
            self.trainables = [self.trainables]


    def _build(self, shape_X):
        self.history_ = {}
        self.network_.predict(np.zeros((1,) + shape_X))
        self._add_regularization()
        
        compil_params = copy.deepcopy(self.compil_params)
        if not "loss" in compil_params:
            compil_params["loss"] = "mean_squared_error"
        if not "optimizer" in compil_params:
            compil_params["optimizer"] = Adam(0.001)
        
        self.network_.compile(**compil_params)
        return self


    def fit(self, Xt, yt, **fit_params):
        """
        Fit RegularTransferNN.

        Parameters
        ----------
        Xt : numpy array
            Target input data.

        yt : numpy array
            Target output data.

        Returns
        -------
        self : returns an instance of self
        """        
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        Xt = check_one_array(Xt)
        yt = check_one_array(yt)
        
        if not self.is_built_:
            self._build(Xt.shape[1:])
            self.is_built_ = True
        
        hist = self.network_.fit(Xt, yt, **fit_params)
        
        for k, v in hist.history.items():
            self.history_[k] = self.history_.get(k, []) + v
        
        return self
    
    
    def _get_regularizer(self, old_weight, weight, lambda_=1.):
        if self.regularizer == "l2":
            def regularizer():
                return lambda_ * K.mean(K.square(old_weight - weight))
        if self.regularizer == "l1":
            def regularizer():
                return lambda_ * K.mean(K.abs(old_weight - weight))
        return regularizer


    def _add_regularization(self):
        i = 0
        for layer in reversed(self.network_.layers):
            if (hasattr(layer, "weights") and 
            layer.weights is not None and
            len(layer.weights) != 0):
                if i >= len(self.trainables):
                    trainable = self.trainables[-1]
                else:
                    trainable = self.trainables[i]
                if i >= len(self.lambdas):
                    lambda_ = self.lambdas[-1]
                else:
                    lambda_ = self.lambdas[i]
                if not trainable:
                    layer.trainable = False
                for weight in reversed(layer.weights):
                    old_weight = tf.identity(weight)
                    old_weight.trainable = False
                    self.network_.add_loss(self._get_regularizer(
                        old_weight, weight, lambda_))
                i += 1
        
        
    def predict(self, X):
        """
        Return predictions of target network.
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            predictions of target network
        """
        X = check_one_array(X)
        return self.network_.predict(X)