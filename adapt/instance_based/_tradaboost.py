"""
Transfer Adaboost
"""

import numpy as np

from adapt.utils import check_indexes, check_estimator


def _get_median_predict(X, predictions, weights):
    # Sort the predictions
    sorted_idx = np.argsort(predictions, axis=1)

    # Find index of median prediction for each sample
    weight_cdf = np.cumsum(weights[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    median_idx = median_or_above.argmax(axis=1)
    median_estimators = sorted_idx[np.arange(len(X)), median_idx]

    # Return median predictions
    return predictions[np.arange(len(X)), median_estimators]


def _binary_search(func):
    left=0
    right=1
    tol=1.e-3
    best=1
    best_score=1
    for i in range(1000):
        if np.abs(func(left)) < tol:
            best = left
            break
        elif np.abs(func(right)) < tol:
            best = right
            break
        else:
            midle = (left + right) / 2
            if func(midle) < best_score:
                best = midle
                best_score = func(midle)
            if func(midle) * func(left) <= 0:
                right = midle
            else:
                left = midle
    if i >= 999:
        print("Binary search has not converged."
              " Set value to the current best.")
    return best


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
      
    The prediction are then given by the vote of the :math:`N \\setminus 2`
    last computed estimators weighted by their respective parameter
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
        
    n_estimators : int, optional (default=10)
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
        
    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.
        
    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.

    References
    ----------
    .. [1] `[1] <http://www.cs.ust.hk/~qyang/Docs/2007/tradaboost.pdf>`_ Dai W., \
Yang Q., Xue G., and Yu Y. "Boosting for transfer learning". In ICML, 2007.
    """

    def __init__(self, get_estimator=None, n_estimators=10, **kwargs):
        self.get_estimator = get_estimator
        self.n_estimators = n_estimators
        self.kwargs = kwargs


    def fit(self, X, y, src_index, tgt_index,
            sample_weight=None, **fit_params):
        """
        Fit TrAdaBoost
        
        Parameters
        ----------
        X : numpy array
            Input data.

        y : numpy array
            Output data. Binary: {0, 1}.

        src_index : iterable
            indexes of source labeled data in X, y.

        tgt_index : iterable
            indexes of target unlabeled data in X, y.

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.
        
        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index)
        
        n_s = len(src_index) 
        n_t = len(tgt_index)
        
        if sample_weight is None:
            sample_weight_src = np.ones(n_s) / (n_s + n_t)
            sample_weight_tgt = np.ones(n_t) / (n_s + n_t)
        else:
            sum_weights = (sample_weight[src_index].sum() +
                           sample_weight[tgt_index].sum())
            sample_weight_src = sample_weight[src_index] / sum_weights
            sample_weight_tgt = sample_weight[tgt_index] / sum_weights


        self.sample_weights_src_ = []
        self.sample_weights_tgt_ = []
        self.estimators_ = []
        self.estimator_errors_ = []

        for iboost in range(self.n_estimators):
            self.sample_weights_src_.append(
                np.copy(sample_weight_src))
            self.sample_weights_tgt_.append(
                np.copy(sample_weight_tgt))

            sample_weight_src, sample_weight_tgt = self._boost(
                iboost, X, y, src_index, tgt_index,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )

            if sample_weight_src is None:
                break

            sum_weights = (sample_weight_src.sum() +
                           sample_weight_tgt.sum())
            sample_weight_src = sample_weight_src / sum_weights
            sample_weight_tgt = sample_weight_tgt / sum_weights

        self.estimator_errors_ = np.array(self.estimator_errors_)
        return self
        
        
    def _boost(self, iboost, X, y, src_index, tgt_index, 
               sample_weight_src, sample_weight_tgt, **fit_params):

        index = np.concatenate((src_index, tgt_index))
        sample_weight = np.concatenate((sample_weight_src,
                                        sample_weight_tgt))
        
        estimator = check_estimator(self.get_estimator, **self.kwargs)
        
        try:
            estimator.fit(X[index], y[index], 
                          sample_weight=sample_weight,
                          **fit_params)
        except:
            bootstrap_index = np.random.choice(
            index, size=len(index), replace=True,
            p=sample_weight)
            estimator.fit(X[bootstrap_index], y[bootstrap_index],
                          **fit_params)
        
        error_vect_src = np.abs(
            estimator.predict(X[src_index]).ravel() - y[src_index])
        error_vect_tgt = np.abs(
            estimator.predict(X[tgt_index]).ravel() - y[tgt_index])
        error_vect = np.concatenate((error_vect_src, error_vect_tgt))
        
        if isinstance(self, TrAdaBoostR2) or isinstance(self, _AdaBoostR2):
            error_max = error_vect.max()
            if error_max != 0:
                error_vect /= error_max
                error_vect_src /= error_max
                error_vect_tgt /= error_max

        if isinstance(self, _AdaBoostR2):
            estimator_error = (sample_weight * error_vect).sum()
        else:
            estimator_error = ((sample_weight_tgt * error_vect_tgt).sum() /
                               (2 * sample_weight_tgt.sum()))

        assert estimator_error < 0.5, ("est: %s, %s, %s"%(str(error_vect_tgt), str(y[tgt_index]), str(estimator.predict(X[tgt_index]).ravel())))
        
        if estimator_error >= 0.5:
            return None, None
        
        beta_t = estimator_error / (1. - estimator_error)
        
        beta_s = 1. / (1. + np.sqrt(
            2. * np.log(len(src_index)) / self.n_estimators
        ))
        
        if not iboost == self.n_estimators - 1:
            if isinstance(self, _AdaBoostR2):
                sample_weight_tgt = (sample_weight_tgt *
                np.power(beta_t, (1 - error_vect_tgt)))

                sample_weight_tgt *= ((1. - sample_weight_src.sum()) /
                                      sample_weight_tgt.sum())
            else:
                # Source updating weights
                sample_weight_src *= np.power(
                    beta_s, error_vect_src)

                # Target updating weights
                sample_weight_tgt *= np.power(
                    beta_t, - error_vect_tgt)

        self.estimators_.append(estimator)
        self.estimator_errors_.append(estimator_error)
        
        return sample_weight_src, sample_weight_tgt


    def predict(self, X):
        """
        Return weighted vote of estimators.
        
        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Vote results.
        """
        N = len(self.estimators_)
        weights = np.array([err / (1-err) if err!=0 else np.exp(-1)
                   for err in self.estimator_errors_])
        weights = weights[int(N/2):]
        weights = np.resize(weights, (len(X), len(weights))).T
        predictions = np.array([
            est.predict(X).ravel()
            for est in self.estimators_[int(N/2):]])
        return (np.prod(np.power(weights, -predictions), axis=0) >=
                np.prod(np.power(weights,
                (-1/2) * np.ones(predictions.shape)),
                axis=0)).astype(float)



class TrAdaBoostR2(TrAdaBoost):
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
        
    n_estimators : int, optional (default=10)
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

    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.
        
    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.

    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
    """

    def __init__(self, get_estimator=None, n_estimators=10, **kwargs):
        self.get_estimator = get_estimator
        self.n_estimators = n_estimators
        self.kwargs = kwargs


    def fit(self, X, y, src_index, tgt_index,
            sample_weight=None, **fit_params):
        """
        Fit TrAdaBoostR2
        
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

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.
        
        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        return super().fit(X, y, src_index, tgt_index,
                           sample_weight, **fit_params)


    def predict(self, X):
        """
        Return weighted median of estimators.
        
        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Median results.
        """
        N = len(self.estimators_)
        weights = np.array([np.log((1-err)/err)
                  if err!=0 else 1
                  for err in self.estimator_errors_])
        weights = weights[int(N/2):]
        predictions = np.array([
            est.predict(X).ravel()
            for est in self.estimators_[int(N/2):]]).T
        return _get_median_predict(X, predictions, weights)


class _AdaBoostR2(TrAdaBoost):
    """
    AdaBoostR2 object with fixed sample weights.
    """
    def __init__(self,
                 get_estimator=None,
                 n_estimators=10,
                 **kwargs):
        self.get_estimator = get_estimator
        self.n_estimators = n_estimators
        self.kwargs = kwargs


    def fit(self, X, y, src_index, tgt_index,
            sample_weight, **fit_params):
        """
        Fit AdaBoostR2
        """
        return super().fit(X, y, src_index, tgt_index,
                           sample_weight, **fit_params)


    def predict(self, X):
        """
        Predict AdaBoostR2
        """
        N = len(self.estimators_)
        weights = np.array([np.log((1-err)/err)
                  if err!=0 else 1
                  for err in self.estimator_errors_])
        weights = weights[int(N/2):]
        predictions = np.array([
            est.predict(X).ravel()
            for est in self.estimators_[int(N/2):]]).T
        return _get_median_predict(X, predictions, weights)

    
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
        
    n_estimators : int, optional (default=10)
        Number of boosting iteration in second stage.
        
    n_estimators_fs : int, optional (default=10)
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

    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.

    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.

    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
    """
    def __init__(self,
                 get_estimator=None,
                 n_estimators=10,
                 n_estimators_fs=10,
                 cv=5,
                 **kwargs):
        self.get_estimator = get_estimator
        self.n_estimators = n_estimators
        self.n_estimators_fs = n_estimators_fs
        self.cv = cv
        self.kwargs = kwargs


    def fit(self, X, y, src_index, tgt_index, sample_weight=None,
            **fit_params):
        """
        Fit TwoStageTrAdaBoostR2.
        
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

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.
        
        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index)
        
        n_s = len(src_index) 
        n_t = len(tgt_index)
        
        if sample_weight is None:
            sample_weight_src = np.ones(n_s) / (n_s + n_t)
            sample_weight_tgt = np.ones(n_t) / (n_s + n_t)
        else:
            sum_weights = (sample_weight[src_index].sum() +
                           sample_weight[tgt_index].sum())
            sample_weight_src = sample_weight[src_index] / sum_weights
            sample_weight_tgt = sample_weight[tgt_index] / sum_weights


        self.sample_weights_src_ = []
        self.sample_weights_tgt_ = []
        self.estimators_ = []
        self.estimator_errors_ = []

        for iboost in range(self.n_estimators):
            self.sample_weights_src_.append(
                np.copy(sample_weight_src))
            self.sample_weights_tgt_.append(
                np.copy(sample_weight_tgt))

            cv_score = self._cross_val_score(
                X, y, src_index, tgt_index,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )
            
            self.estimator_errors_.append(cv_score.mean())
            
            sample_weight_src, sample_weight_tgt = self._boost(
                iboost, X, y, src_index, tgt_index,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )

            if sample_weight_src is None:
                break

            sum_weights = (sample_weight_src.sum() +
                           sample_weight_tgt.sum())
            sample_weight_src = sample_weight_src / sum_weights
            sample_weight_tgt = sample_weight_tgt / sum_weights

        self.estimator_errors_ = np.array(self.estimator_errors_)
        return self


    def _boost(self, iboost, X, y, src_index, tgt_index, 
               sample_weight_src, sample_weight_tgt, **fit_params):

        estimator = _AdaBoostR2(self.get_estimator,
                                self.n_estimators_fs,
                                **self.kwargs)
        
        sample_weight_fill = np.array([np.nan] * len(X))
        sample_weight_fill[src_index] = sample_weight_src
        sample_weight_fill[tgt_index] = sample_weight_tgt
        
        estimator.fit(X, y, src_index, tgt_index,
                      sample_weight=sample_weight_fill,
                      **fit_params)
        
        error_vect_src = np.abs(
            estimator.predict(X[src_index]).ravel() - y[src_index])
        error_vect_tgt = np.abs(
            estimator.predict(X[tgt_index]).ravel() - y[tgt_index])
        error_vect = np.concatenate((error_vect_src, error_vect_tgt))
        
        error_max = error_vect.max()
        if error_max != 0:
            error_vect /= error_max
            error_vect_src /= error_max
            error_vect_tgt /= error_max

        beta = self._get_beta(iboost,
                              sample_weight_src,
                              sample_weight_tgt,
                              error_vect_src,
                              error_vect_tgt)
        
        if not iboost == self.n_estimators - 1:
            sample_weight_src *= np.power(
                beta, error_vect_src
            )

        self.estimators_.append(estimator)
        return sample_weight_src, sample_weight_tgt


    def _get_beta(self, iboost, sample_weight_src, sample_weight_tgt,
                  error_vect_src, error_vect_tgt):

        n_s = len(sample_weight_src)
        n_t = len(sample_weight_tgt)

        K_t = (n_t/(n_s + n_t) + (iboost/(self.n_estimators - 1)) *
               (1 - n_t/(n_s + n_t)))
        C_t = sample_weight_tgt.sum() * ((1 - K_t) / K_t)
        
        def func(x):
            return np.dot(sample_weight_src,
                   np.power(x, error_vect_src)) - C_t
        return _binary_search(func)


    def _cross_val_score(self, X, y, src_index, tgt_index,
                         sample_weight_src, sample_weight_tgt,
                         **fit_params):

        split = int(len(tgt_index) / self.cv)
        scores = []
        for i in range(self.cv):
            if i == self.cv-1:
                test_index = tgt_index[i * split:]
            else:
                test_index = tgt_index[i * split: (i + 1) * split]
            train_index = np.array(list(set(tgt_index) - set(test_index)))
            
            estimator = _AdaBoostR2(self.get_estimator,
                                    self.n_estimators_fs,
                                    **self.kwargs)

            sample_weight_fill = np.array([np.nan] * len(X))
            sample_weight_fill[src_index] = sample_weight_src
            sample_weight_fill[tgt_index] = sample_weight_tgt
            
            sample_weight_fill[train_index] *= (
                sample_weight_tgt.sum() / 
                sample_weight_fill[train_index].sum()
            )

            estimator.fit(X, y, src_index, train_index,
                          sample_weight=sample_weight_fill,
                          **fit_params)

            scores.append(np.sum(np.square(
                estimator.predict(X[test_index]).ravel() - y[test_index]
            )))
        return np.array(scores)


    def predict(self, X):
        """
        Return predictions of the best estimator according
        to cross-validation scores.
        
        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Best estimator predictions.
        """
        best_estimator = self.estimators_[
            self.estimator_errors_.argmin()]
        return best_estimator.predict(X).ravel()