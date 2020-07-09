"""
Regular Transfer
"""



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

    def __init__(self):
        pass


    def fit(self):
        """
        Fit 
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass


    def predict(self):
        """
        Predict
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass
    
    
    
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
        If get_estimator is ``None``, a ``LinearRegression``
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

    def __init__(self):
        pass


    def fit(self):
        """
        Fit 
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass


    def predict(self):
        """
        Predict
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass


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
        \\, ||f(X_T, \\beta) - y_T||^2 + \sum_{i=1}^{D} \\lambda_i ||\\beta_i - {\\beta_S}_i||^2
        
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
    get_estimator : callable or object, optional (default=None)
        Constructor for the source estimator.
        The estimator returned by ``get_estimator`` should belong
        to one of these two ``scikit-learn`` class:
        ``LinearRegression`` or ``Ridge``.
        If get_estimator is ``None``, a ``LinearRegression``
        object will be used by default as estimator.
        
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

    .. [2] `[2] <https://hal.inria.fr/hal-00911179v1/document>`_ \
Oquab M., Bottou L., Laptev I., Sivic J. "Learning and \
transferring mid-level image representations using convolutional \
neural networks". In CVPR, 2014.
    """

    def __init__(self):
        pass


    def fit(self):
        """
        Fit 
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass


    def predict(self):
        """
        Predict
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass