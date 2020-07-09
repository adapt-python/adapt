"""
Transfer Adaboost
"""


class _AdaBoostR2Prime:
    """
    AdaBoostR2Prime
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



class TrAdaBoost:
    """
    Transfer AdaBoost for Classification
    
    TrAdaBoost algorithm is a **supervised** instances-based domain
    adaptation method suited for **classification** tasks.
    
    The method is based on a "**reverse boosting**" principle where the
    weights of source instances poorly predicted decrease at each
    boosting iteration whereas the ones of target instances increase.
    
    The algorithm performs the following steps:
    
    - **1.** Normalize weights: :math:`\\sum w_S + \\sum w_T = 1`.
    - **2.** Fit an estimator :math:`f` on source and target labeled data
      :math:`(X_S, y_S), (X_T, y_T)` with the respective importances
      weights: :math:`w_S, w_T`.
    - **3.** Compute error vectors of training instances:
    
      - :math:`\\epsilon_S = L_{01}(f(X_S), y_S)`.
      - :math:`\\epsilon_T = L_{01}(f(X_T), y_T)`.
      
    - **4.** Compute total weighted error of target instances:
      :math:`E_T = \\frac{1}{n_T} w_T^T \\epsilon_T`.
    - **5.** Update source and target weights:
    
       - :math:`w_S = w_S \\beta^{\\epsilon_S}`.
       - :math:`w_T = w_T \\beta_T^{-\\epsilon_T}`.
       
      Where:
      
      - :math:`\\beta = 1 \\setminus (1 + \\sqrt{2 \\text{ln} n_S \\setminus N})`.
      - :math:`\\beta_T = E_T \\setminus (1 - E_T)`.
      
    - **6.** Return to step **1** and loop until the number :math:`N`
      of boosting iteration is reached.
      
    The prediction are then given by the vote of the :math:`N`
    computed estimators weighted by their respective parameter
    :math:`\\beta_T`.
    
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
        
    n_estimator : int, optional (default=10)
        Number of boosting iteration.
        
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    estimators_ : list of object
        List of fitted estimators
        
    estimator_errors_ : list of float
        List of weighted estimator errors computed on
        labeled target data.

    References
    ----------
    .. [1] `[1] <http://www.cs.ust.hk/~qyang/Docs/2007/tradaboost.pdf>`_ Dai W., \
Yang Q., Xue G., and Yu Y. "Boosting for transfer learning". In ICML, 2007.
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



class TrAdaBoostR2:
    """
    Transfer AdaBoost for Regression
    
    TrAdaBoostR2 algorithm is a **supervised** instances-based domain
    adaptation method suited for **regression** tasks.
    
    The method is based on a "**reverse boosting**" principle where the
    weights of source instances poorly predicted decrease at each
    boosting iteration whereas the ones of target instances increase.
    
    The algorithm performs the following steps:
    
    - **1.** Normalize weights: :math:`\\sum w_S + \\sum w_T = 1`.
    - **2.** Fit an estimator :math:`f` on source and target labeled data
      :math:`(X_S, y_S), (X_T, y_T)` with the respective importances
      weights: :math:`w_S, w_T`.
    - **3.** Compute error vectors of training instances:
    
      - :math:`\\epsilon_S = L(f(X_S), y_S)`.
      - :math:`\\epsilon_T = L(f(X_T), y_T)`.
      
    - **4** Normalize error vectors:
    
      - :math:`\\epsilon_S = \\epsilon_S \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.
      - :math:`\\epsilon_T = \\epsilon_T \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.
    
    - **5.** Compute total weighted error of target instances:
      :math:`E_T = \\frac{1}{n_T} w_T^T \\epsilon_T`.
    
    
    - **6.** Update source and target weights:
    
       - :math:`w_S = w_S \\beta^{\\epsilon_S}`.
       - :math:`w_T = w_T \\beta_T^{-\\epsilon_T}`.
       
      Where:
      
      - :math:`\\beta = 1 \\setminus (1 + \\sqrt{2 \\text{ln} n_S \\setminus N})`.
      - :math:`\\beta_T = E_T \\setminus (1 - E_T)`.
      
    - **7.** Return to step **1** and loop until the number :math:`N`
      of boosting iteration is reached.
      
    The prediction are then given by the weighted median of the
    :math:`N \\setminus 2` last estimators.
    
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
        
    n_estimator : int, optional (default=10)
        Number of boosting iteration.
        
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    estimators_ : list of object
        List of fitted estimators
        
    estimator_errors_ : list of float
        List of weighted estimator errors computed on
        labeled target data.

    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
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
    
    
    
class TwoStageTrAdaBoostR2:
    """
    Two Stage Transfer AdaBoost for Regression
    
    TwoStageTrAdaBoostR2 algorithm is a **supervised** instances-based
    domain adaptation method suited for **regression** tasks.
    
    The method is based on a "**reverse boosting**" principle where the
    weights of source instances poorly predicted decrease at each
    boosting iteration whereas the ones of target instances increase.
    
    This "two stages" version of TrAdaBoostR2 algorithm update separately
    the weights of source and target instances.
    
    In a first stage, the weights of source instances are
    frozen whereas the ones of target instances are updated according to
    the classical AdaBoostR2 algorithm. In a second stage, the weights of
    target instances are now frozen whereas the ones of source instances
    are updated according to the TrAdaBoost algorithm.
    
    At each first stage, a cross-validation score is computed with the
    labeled target data available. The CV scores obtained are used at 
    the end to select the best estimator whithin all boosting iterations.
       
    The algorithm performs the following steps:
    
    - **1.** Normalize weights: :math:`\\sum w_S + \\sum w_T = 1`.
    - **2.** Fit an AdaBoostR2 estimator :math:`f` on source and target
      labeled data :math:`(X_S, y_S), (X_T, y_T)` with the respective
      importances initial weights: :math:`w_S, w_T`. During training
      of the AdaBoost estimator, the source weights :math:`w_S` are
      frozen.
    - **3.** Compute a cross-validation score on :math:`(X_T, y_T)`
    - **4.** Compute error vectors of training instances:
    
      - :math:`\\epsilon_S = L(f(X_S), y_S)`.
      - :math:`\\epsilon_T = L(f(X_T), y_T)`.
      
    - **5** Normalize error vectors:
    
      - :math:`\\epsilon_S = \\epsilon_S \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`.
      - :math:`\\epsilon_T = \\epsilon_T \\setminus
        max_{\\epsilon \\in \\epsilon_S \cup \\epsilon_T} \\epsilon`. 
    
    - **6.** Update source and target weights:
    
       - :math:`w_S = w_S \\beta_S^{\\epsilon_S} \\setminus Z`.
       - :math:`w_T = w_T \\setminus Z`.
       
      Where:
      
      - :math:`Z` is a normalizing constant.
      - :math:`\\beta_S` is chosen such that the sum of target weights
        :math:`w_T` is equal to :math:`\\frac{n_T}{n_T + n_S}
        + \\frac{t}{N - 1}(1 - \\frac{n_T}{n_T + n_S})` with :math:`t`
        the current boosting iteration number. :math:`\\beta_S` is found
        using binary search.
      
    - **7.** Return to step **1** and loop until the number :math:`N`
      of boosting iteration is reached.
      
    The prediction are then given by the best estimator according
    to cross-validation scores.
    
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
        
    n_estimator : int, optional (default=10)
        Number of boosting iteration in second stage.
        
    n_estimator_fs : int, optional (default=10)
        Number of boosting iteration in first stage
        (given to AdaboostR2 estimators)
        
    cv: int, optional (default=5)
        Split cross-validation parameter.
        
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    estimators_ : list of object
        List of fitted AdaboostR2 estimators for each
        first stage.
        
    estimator_scores_ : list of float
        List of cross-validation scores of estimators.

    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
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