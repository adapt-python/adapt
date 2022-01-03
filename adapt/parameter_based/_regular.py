"""
Regular Transfer
"""

import numpy as np
from sklearn.exceptions import NotFittedError
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

from adapt.base import BaseAdaptEstimator, BaseAdaptDeep, make_insert_doc
from adapt.utils import (check_arrays,
                         set_random_seed,
                         check_estimator,
                         check_network)


def get_zeros_network(name=None):
    """
    Return a tensorflow Model of two hidden layers
    with 10 neurons each and relu activations. The
    last layer is composed of one neuron with linear
    activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential(name=name)
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


@make_insert_doc()
class RegularTransferLR(BaseAdaptEstimator):
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
    lambda_ : float (default=1.0)
        Trade-Off parameter.

    Attributes
    ----------
    estimator_ : Same class as estimator
        Fitted Estimator.
        
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
                 estimator=None,
                 Xt=None,
                 yt=None,
                 lambda_=1.,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
                
        if not hasattr(estimator, "coef_"):
            raise NotFittedError("`estimator` argument has no ``coef_`` attribute, "
                                 "please call `fit` on `estimator` or use "
                                 "another estimator.")

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)

    
    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit RegularTransferLR.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.
            
        fit_params : key, value arguments
            Not used. Here for sklearn compatibility.

        Returns
        -------
        self : returns an instance of self
        """        
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)
        
        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)
        
        if self.estimator_.fit_intercept:
            beta_src = np.concatenate((
                self.estimator_.intercept_ * np.ones(yt.shape).mean(0, keepdims=True),
                self.estimator_.coef_.transpose()
            ))
            Xt = np.concatenate(
                (np.ones((len(Xt), 1)), Xt),
                axis=-1)
        else:
            beta_src = self.estimator_.coef_.transpose()
        
        func = self._get_func(Xt, yt, beta_src)
        
        beta_tgt = minimize(func, beta_src)['x']
        beta_tgt = beta_tgt.reshape(beta_src.shape)

        if self.estimator_.fit_intercept:
            self.estimator_.intercept_ = beta_tgt[0]
            self.estimator_.coef_ = beta_tgt[1:].transpose()
        else:
            self.estimator_.coef_ = beta_tgt.transpose()
        return self
    
    
    def _get_func(self, Xt, yt, beta_src):
        def func(beta):
            beta = beta.reshape(beta_src.shape)
            return (np.linalg.norm(Xt.dot(beta) - yt) ** 2 +
                    self.lambda_ * np.linalg.norm(beta - beta_src) ** 2)
        return func

    

@make_insert_doc()
class RegularTransferLC(RegularTransferLR):
    """
    Regular Transfer for Linear Classification
    
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
    lambda_ : float (default=1.0)
        Trade-Off parameter.

    Attributes
    ----------
    estimator_ : Same class as estimator
        Fitted Estimator.
            
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
    ### TODO reshape yt for multiclass.
    
    def _get_func(self, Xt, yt, beta_src):
        def func(beta):
            beta = beta.reshape(beta_src.shape)
            return (np.sum(np.log(1 + np.exp(
                -(2*yt-1) * Xt.dot(beta)))) +
                self.lambda_ * np.linalg.norm(beta - beta_src) ** 2)
        return func


@make_insert_doc(["task"])
class RegularTransferNN(BaseAdaptDeep):
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
    lambdas : float or list of float, optional (default=1.0)
        Trade-off parameters.
        If a list is given, values from ``lambdas`` are assigned
        successively to the list of ``network`` layers with 
        weights parameters going from the last layer to the first one.
        If the length of ``lambdas`` is smaller than the length of
        ``network`` layers list, the last trade-off value will be
        asigned to the remaining layers.

    Attributes
    ----------
    task_ : tensorflow Model
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
                 task=None,
                 Xt=None,
                 yt=None,
                 lambdas=1.0,
                 regularizer="l2",
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
        
        if not regularizer in ["l1", "l2"]:
            raise ValueError("`regularizer` argument should be "
                             "'l1' or 'l2', got, %s"%str(regularizer))
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit RegularTransferNN.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.
            
        fit_params : key, value arguments
            Arguments given to the fit method of the model
            (epochs, batch_size, callbacks...).

        Returns
        -------
        self : returns an instance of self
        """        
        Xt, yt = self._get_target_data(Xt, yt)
        Xs = Xt
        ys = yt
        return super().fit(Xs, ys, Xt=Xt, yt=yt, **fit_params)
    
    
    def _initialize_networks(self):
        if self.task is None:
            self.task_ = get_zeros_network(name="task")
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")
        self._add_regularization()
    
    
    def _get_regularizer(self, old_weight, weight, lambda_=1.):
        if self.regularizer == "l2":
            def regularizer():
                return lambda_ * tf.reduce_mean(tf.square(old_weight - weight))
        if self.regularizer == "l1":
            def regularizer():
                return lambda_ * tf.reduce_mean(tf.abs(old_weight - weight))
        return regularizer


    def _add_regularization(self):
        i = 0
        if not hasattr(self.lambdas, "__iter__"):
            lambdas = [self.lambdas]
        else:
            lambdas = self.lambdas
        
        for layer in reversed(self.task_.layers):
            if (hasattr(layer, "weights") and 
            layer.weights is not None and
            len(layer.weights) != 0):
                if i >= len(lambdas):
                    lambda_ = lambdas[-1]
                else:
                    lambda_ = lambdas[i]
                for weight in reversed(layer.weights):
                    old_weight = tf.identity(weight)
                    old_weight.trainable = False
                    self.add_loss(self._get_regularizer(
                        old_weight, weight, lambda_))
                i += 1
        
        
    def call(self, inputs):
        return self.task_(inputs)
    
    
    def transform(self, X):
        """
        Return X
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        X_enc : array
            predictions of encoder network
        """
        return X
    
    
    def predict_disc(self, X):
        """
        Not used.
        """     
        pass