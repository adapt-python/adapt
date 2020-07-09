"""
Kullback-Leibler Importance Estimation Procedure
"""



class KLIEP:
    """
    KLIEP: Kullback–Leibler Importance Estimation Procedure
    
    KLIEP is an instance-based method for domain adaptation. 
    
    The purpose of the algorithm is to correct the difference between
    input distributions of source and target domains. This is done by
    finding a the source instances **reweighting** which minimizes the 
    **Kullback-Leibler divergence** between source and target distributions.
    
    The source instance weights are given by the following formula:
    
    .. math::
    
        w(x) = \sum_{x_i \in X_T} \\alpha_i K_{\sigma}(x, x_i)
        
    Where:
    
    - :math:`x, x_i` are input instances.
    - :math:`X_T` is the target input data.
    - :math:`\\alpha_i` are the basis functions coefficients.
    - :math:`K_{\sigma}(x, x_i) = \\text{exp}(-\\frac{||x - x_i||^2}{2\sigma^2})`
      are kernel functions of bandwidth :math:`\sigma`.
      
    KLIEP algorithm consists in finding the optimal :math:`\\alpha_i` according to
    the following optimization problem:
    
    .. math::
    
        \max_{\\alpha_i } \sum_{x_i \in X_T} \\text{log}(
        \sum_{x_j \in X_T} \\alpha_i K_{\sigma}(x_j, x_i))
        
    Subject to:
    
    .. math::
    
        \sum_{x_k \in X_S} \sum_{x_j \in X_T} \\alpha_i K_{\sigma}(x_j, x_k)) = n_S
        
    Where:
    
    - :math:`X_T` is the source input data of size :math:`n_S`.
    
    The above OP is solved through gradient descent algorithm.
    
    Furthemore a LCV procedure can be added to select the appropriate
    bandwidth :math:`\sigma`. The parameter is then selected using
    cross-validation on the :math:`J` score defined as follow:
    :math:`J = \\frac{1}{|\\mathcal{X}|} \\sum_{x \\in \\mathcal{X}} \\text{log}(w(x))`
    
    Finally, an estimator is fitted using the reweighted labeled source instances.
    
    KLIEP method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.
    
    Parameters
    ----------
    get_estimator : callable or object, optional (default=None)
        Constructor for the estimator.
        If a callable function is given it should return an estimator
        object (with ``fit`` and ``predict`` methods).
        If a class is given, a new instance of this class will
        be built and used as estimator.
        If get_estimator is ``None``, a ``LinearRegression`` object will be
        used by default as estimator.
        
    sigmas : float or list of float, optional (default=0.1)
        Kernel bandwidths.
        If ``sigmas`` is a list of multiple values, the
        kernel bandwidth is selected with the LCV procedure.
        
    cv : int, optional (default=5)
        Cross-validation split parameter.
        Used only if sigmas has more than one value.
        
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
        
    sigma_ = float
        Sigma selected for the kernel
        
    self.j_scores_ = list of float
        List of J scores.
        
    estimator_ : object
        Fitted estimator.

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3248-direct-importance-estimation\
-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf>`_ \
M. Sugiyama, S. Nakajima, H. Kashima, P. von Bünau and  M. Kawanabe. \
"Direct importance estimation with model selection and its application \
to covariateshift adaptation". In NIPS 2007
    """

    def __init__(self):
        pass


    def fit(self):
        pass


    def predict(self):
        pass
