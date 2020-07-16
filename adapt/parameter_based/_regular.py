"""
Regular Transfer
"""

import os
import tempfile

import numpy as np
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  LogisticRegression,
                                  RidgeClassifier)
from scipy.optimize import minimize
import tensorflow.keras.backend as K

from adapt.utils import (check_indexes,
                         check_estimator,
                         check_network,
                         get_default_task)


def _get_custom_regularizer_l2(source_weight_matrix_i, alpha_i):
    def custom_regularizer(weight_matrix):
        return alpha_i * K.sum(
            K.square(weight_matrix - source_weight_matrix_i))
    return custom_regularizer


def _add_regularization(model, lambdas, trainables):
    # Init regularizers list
    regularizers = []
    config = model.get_config()

    for layer, lambda_, trainable, i in zip(model.layers,
                                          lambdas,
                                          trainables,
                                          range(len(model.layers))):
        if hasattr(layer, 'kernel_regularizer'):
            regularizers.append(_get_custom_regularizer_l2(
                layer.get_weights()[0], lambda_))
            config['layers'][i]['config']['kernel_regularizer'] = (
            regularizers[-1])

        if (hasattr(layer, 'use_bias') and layer.use_bias and
            hasattr(layer, 'bias_regularizer')):
            regularizers.append(_get_custom_regularizer_l2(
                layer.get_weights()[1], lambda_))  
            config['layers'][i]['config']['bias_regularizer'] = (
                regularizers[-1])

        if hasattr(layer, 'trainable'):
            config['layers'][i]['config']['trainable'] = trainable

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(),
                                    'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    new_model = model.__class__.from_config(config)

    # Reload the model weights
    new_model.load_weights(tmp_weights_path, by_name=True)

    # Compile
    new_model.compile(optimizer=model.optimizer,
                      loss=model.loss
                     )
    return new_model


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
    get_estimator : callable or object, optional (default=None)
        Constructor for the source estimator.
        The estimator returned by ``get_estimator`` should belong
        to one of these two ``scikit-learn`` class:
        ``LinearRegression`` or ``Ridge``.
        If get_estimator is ``None``, a ``LinearRegression``
        object will be used by default as estimator.
        
    lambdap : float, optional (default=1.0)
        Trade-Off parameter.
        
    intercept : boolean, optional (default=True)
        If True, use intercept for target estimator
        
    fit_source : boolean, optional (default=True)
        If ``fit_source`` is set to ``True``, the estimator
        returned by ``get_estimator`` will be fitted on
        source data.
        If ``False``, the source estimator will be
        considered already fitted and no preliminary
        training on source data will be done.
        
    kwargs : key, value arguments, optional
        Additional arguments for ``get_estimator``.

    Attributes
    ----------
    estimator_src_ : instance of LinearRegression or Ridge
        Fitted source estimator.
        
    coef_ : numpy array
        Coefficient of target linear estimator
        
    intercept_ : float
        Intercept of target linear estimator

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    def __init__(self,
                 get_estimator=None,
                 lambdap=1.0,
                 intercept=True,
                 fit_source=True,
                 **kwargs):
        self.get_estimator = get_estimator
        self.lambdap = lambdap
        self.intercept = intercept
        self.fit_source = fit_source
        self.kwargs = kwargs

        if self.get_estimator is None:
            self.get_estimator = LinearRegression


    def fit(self, X, y, src_index, tgt_index,
            fit_params_src=None, **fit_params_tgt):
        """
        Fit RegularTransferLR
        
        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data.

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target unlabeled data in X, y.
        
        fit_params_src : dict, optional (default=None)
            Arguments given to the fit method of the
            source estimator (epochs, batch_size...).
            If None, ``fit_params_src = fit_params_tgt``
        
        fit_params_tgt : key, value arguments
            Arguments given to the fit method of the
            target estimator (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        if fit_params_src is None:
            fit_params_src = fit_params_tgt
        
        check_indexes(src_index, tgt_index)
        
        self.estimator_src_ = check_estimator(self.get_estimator,
                                      **self.kwargs)
        if (not isinstance(self.estimator_src_, LinearRegression) and
            not isinstance(self.estimator_src_, Ridge)):
            raise ValueError("'get_estimator' should return a"
                             " LinearRegression or Ridge instance.")
        if self.fit_source:
            self.estimator_src_.fit(X[src_index], y[src_index],
                                    **fit_params_src)
        
        if self.intercept:
            beta_src = np.concatenate((
                np.array([self.estimator_src_.intercept_]),
                self.estimator_src_.coef_
            ))
            Xt = np.concatenate(
                (np.ones((len(tgt_index), 1)), X[tgt_index]),
                axis=1)
        else:
            beta_src = self.estimator_src_.coef_
            Xt = X[tgt_index]
        yt = y[tgt_index]
        
        def func(beta):
            return (np.linalg.norm(Xt.dot(beta.reshape(-1, 1)) - yt) ** 2 +
                    self.lambdap * np.linalg.norm(beta - beta_src) ** 2)

        beta_tgt = minimize(func, beta_src)['x']

        if self.intercept:
            self.intercept_ = beta_tgt[0]
            self.coef_ = beta_tgt[1:]
        else:
            self.intercept_ = 0.
            self.coef_ = beta_tgt        
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
        return (X.dot(self.coef_.reshape(-1, 1))
                + self.intercept_).ravel()
    
    
    
class RegularTransferLC:
    """
    Regular Transfer with Linear Classification
    
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
    get_estimator : callable or object, optional (default=None)
        Constructor for the source estimator.
        The estimator returned by ``get_estimator`` should belong
        to one of these two ``scikit-learn`` class:
        ``LogisticRegression`` or ``RidgeClassifier``.
        If get_estimator is ``None``, a ``LogisticRegression``
        object will be used by default as estimator.
        
    lambdas : float, optional (default=1.0)
        Trade-Off parameter.
        
    intercept : boolean, optional (default=True)
        If True, use intercept for target estimator
        
    fit_source : boolean, optional (default=True)
        If ``fit_source`` is set to ``True``, the estimator
        returned by ``get_estimator`` will be fitted on
        source data.
        If ``False``, the source estimator will be
        considered already fitted and no preliminary
        training on source data will be done.
        
    kwargs : key, value arguments, optional
        Additional arguments for ``get_estimator``.

    Attributes
    ----------
    estimator_src_ : instance of LinearRegression or Ridge
        Fitted source estimator.
        
    coef_ : numpy array
        Coefficient of target linear estimator
        
    intercept_ : float
        Intercept of target linear estimator

    References
    ----------
    .. [1] `[1] <https://www.microsoft.com/en-us/research/wp-\
content/uploads/2004/07/2004-chelba-emnlp.pdf>`_ C. Chelba and \
A. Acero. "Adaptation of maximum entropy classifier: Little data \
can help a lot". In EMNLP, 2004.
    """
    def __init__(self,
                 get_estimator=None,
                 lambdap=1.0,
                 intercept=True,
                 fit_source=True,
                 **kwargs):
        self.get_estimator = get_estimator
        self.lambdap = lambdap
        self.intercept = intercept
        self.fit_source = fit_source
        self.kwargs = kwargs

        if self.get_estimator is None:
            self.get_estimator = LogisticRegression


    def fit(self, X, y, src_index, tgt_index,
            fit_params_src=None, **fit_params_tgt):
        """
        Fit RegularTransferLR
        
        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data. Binary {-1, 1}

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target unlabeled data in X, y.
        
        fit_params_src : dict, optional (default=None)
            Arguments given to the fit method of the
            source estimator (epochs, batch_size...).
            If None, ``fit_params_src = fit_params_tgt``
        
        fit_params_tgt : key, value arguments
            Arguments given to the fit method of the
            target estimator (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        if fit_params_src is None:
            fit_params_src = fit_params_tgt
        
        check_indexes(src_index, tgt_index)
        
        if not np.all(np.isin(
            y[np.concatenate((src_index, tgt_index))], [-1., 1.])):
            raise ValueError("y values should be in {-1, 1}")
        
        self.estimator_src_ = check_estimator(self.get_estimator,
                                      **self.kwargs)
        if (not isinstance(self.estimator_src_, LogisticRegression) and
            not isinstance(self.estimator_src_, RidgeClassifier)):
            raise ValueError("'get_estimator' should return a"
                             " LogisticRegression or RidgeClassifier"
                             " instance.")
        if self.fit_source:
            self.estimator_src_.fit(X[src_index], y[src_index],
                                    **fit_params_src)
        
        if self.intercept:
            beta_src = np.concatenate((
                np.array([self.estimator_src_.intercept_[0]]),
                self.estimator_src_.coef_[0]
            ))
            Xt = np.concatenate(
                (np.ones((len(tgt_index), 1)), X[tgt_index]),
                axis=1)
        else:
            beta_src = self.estimator_src_.coef_[0]
            Xt = X[tgt_index]
        yt = y[tgt_index]
        
#         assert False, "%s"%str(beta_src)
        
        def func(beta):
            return (np.sum(np.log(1 + np.exp(
                    -yt * Xt.dot(beta.reshape(-1, 1)).ravel()))) +
                    self.lambdap * np.linalg.norm(beta - beta_src) ** 2)

        beta_tgt = minimize(func, beta_src)['x']

        if self.intercept:
            self.intercept_ = beta_tgt[0]
            self.coef_ = beta_tgt[1:]
        else:
            self.intercept_ = 0.
            self.coef_ = beta_tgt        
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
        y = X.dot(self.coef_.reshape(-1, 1)) + self.intercept_
        return np.sign(1 / (1 + np.exp(-y.ravel())) - 0.5)


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

    Parameters
    ----------
    get_network : callable or object, optional (default=None)
        Constructor for the source and target networks.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network and an ``output_shape``
        argument giving the shape of the last layer.
        If ``None``, a shallow network with linear activation
        is returned.
        
    lambdas : float or list of float, optional (default=1.0)
        Trade-off parameters.
        If a list is given, values from ``lambdas`` are assigned
        successively to the list of network trainable layers.
        If length of ``lambdas`` is smaller than length of
        trainable layers list, the last trade-off value will
        be asigned to the remaining layers.
        
    trainable : boolean or list of boolean, optional (default=True)
        Set trainable argument of target neural network layers.
        If a list is given, values from ``trainable`` are assigned
        successively to the list of network trainable layers.
        If length of ``trainable`` is smaller than length of
        trainable layers list, the trainable argument of the
        remaining layers will be set to ``True``.
    
    fit_source : boolean, optional (default=True)
        If ``fit_source`` is set to ``True``, the network
        returned by ``get_network`` will be fitted on
        source data.
        If ``False``, the source network will be
        considered already fitted and no preliminary
        training on source data will be done.
        
    kwargs : key, value arguments, optional
        Additional arguments for ``get_network``.

    Attributes
    ----------
    model_src_ : tensorflow Model
        Fitted source network.
        
    model_tgt_ : tensorflow Model
        Fitted target network.

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
                 get_network=None,
                 lambdas=1.0,
                 trainable=True,
                 fit_source=True,
                 **kwargs):
        self.get_network = get_network
        self.lambdas = lambdas
        self.trainable = trainable
        self.fit_source = fit_source
        self.kwargs = kwargs

        if self.get_network is None:
            self.get_network = get_default_task


    def fit(self, X, y, src_index, tgt_index,
            fit_params_src=None, **fit_params_tgt):
        """
        Fit RegularTransferNN
        
        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data.

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target unlabeled data in X, y.
        
        fit_params_src : dict, optional (default=None)
            Arguments given to the fit method of the
            source estimator (epochs, batch_size...).
            If None, ``fit_params_src = fit_params_tgt``
        
        fit_params_tgt : key, value arguments
            Arguments given to the fit method of the
            target estimator (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        if fit_params_src is None:
            fit_params_src = fit_params_tgt
        
        check_indexes(src_index, tgt_index)
        
        self.model_src_ = check_network(self.get_network,
                                        "get_network",
                                        input_shape=X.shape[1:],
                                        output_shape=y.shape[1:],
                                        **self.kwargs)
        if self.fit_source:
            self.model_src_.fit(X[src_index], y[src_index],
                                **fit_params_src)
        layers = self.model_src_.layers

        lambdas, trainables = self._get_lambdas_and_trainables(layers)
        
        self.model_tgt_ = _add_regularization(self.model_src_,
                                              lambdas,
                                              trainables)
        self.model_tgt_.fit(X[tgt_index], y[tgt_index],
                            **fit_params_tgt)
        return self
    
    
    def _get_lambdas_and_trainables(self, layers):
        if not hasattr(self.lambdas, '__iter__'):
            try:
                _lambdas = float(self.lambdas)
                lambdas = [_lambdas for _ in layers]
            except:
                raise Exception("%s is not a valid type for lambdas,"
                                " please provide a list or a float" % 
                                str(type(self.lambdas)))
        else:
            lambdas = (list(self.lambdas)[:len(layers)] +
                       [self.lambdas[-1]] * 
                       (len(layers) - len(self.lambdas)))

        if not hasattr(self.trainable, '__iter__'):
            if isinstance(self.trainable, bool):
                trainables = [self.trainable for _ in layers]
            else:
                raise Exception("%s is not a valid type for trainable,"
                                " please provide a list or a boolean" %
                                str(type(self.trainable)))
        else:
            trainables = (list(self.trainable)[:len(layers)] +
                          [True] * (len(layers) - len(self.lambdas)))
        return lambdas, trainables
        
        
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
        return self.model_tgt_.predict(X)