"""
Transfer Adaboost
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import vstack, issparse

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays, check_estimator, set_random_seed

EPS = np.finfo(float).eps

def _get_median_predict(predictions, weights):
    sorted_idx = np.argsort(predictions, axis=-1)
    # Find index of median prediction for each sample
    weight_cdf = np.cumsum(weights[sorted_idx], axis=-1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[..., -1][..., np.newaxis]
    median_idx = median_or_above.argmax(axis=-1)
    new_predictions = None
    for i in range(median_idx.shape[1]):
        median_estimators = sorted_idx[np.arange(len(predictions)), i, median_idx[:, i]]
        if new_predictions is None:
            new_predictions = predictions[np.arange(len(predictions)), i, median_estimators].reshape(-1,1)
        else:
            new_predictions = np.concatenate((
                new_predictions,
                predictions[np.arange(len(predictions)), i, median_estimators].reshape(-1,1)
            ), axis=1)
    return new_predictions


def _binary_search(func, verbose=1):
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
    if i >= 999 and verbose:
        print("Binary search has not converged."
              " Set value to the current best.")
    return best


@make_insert_doc(supervised=True)
class TrAdaBoost(BaseAdaptEstimator):
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
    n_estimators : int (default=10)
        Number of boosting iteration.
        
    lr : float (default=1.)
        Learning rate. For higher ``lr``, the sample
        weights are updating faster.

    Attributes
    ----------
    estimators_ : list of object
        List of fitted estimators
        
    estimator_errors_ : 1D array of float
        Array of weighted estimator errors computed on
        labeled target data.
        
    estimator_weights_ : 1D array of float
        Array of estimator importance weights.
        
    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.
        
    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import TrAdaBoost
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = TrAdaBoost(RidgeClassifier(), n_estimators=10, Xt=Xt[:10], yt=yt[:10], random_state=0)
    >>> model.fit(Xs, ys)
    Iteration 0 - Error: 0.2550
    Iteration 1 - Error: 0.2820
    Iteration 2 - Error: 0.3011
    Iteration 3 - Error: 0.3087
    Iteration 4 - Error: 0.3046
    Iteration 5 - Error: 0.2933
    Iteration 6 - Error: 0.2819
    Iteration 7 - Error: 0.2747
    Iteration 8 - Error: 0.2712
    Iteration 9 - Error: 0.2698
    >>> model.score(Xt, yt)
    0.66
        
    See also
    --------
    TrAdaBoostR2, TwoStageTrAdaBoostR2

    References
    ----------
    .. [1] `[1] <http://www.cs.ust.hk/~qyang/Docs/2007/tradaboost.pdf>`_ Dai W., \
Yang Q., Xue G., and Yu Y. "Boosting for transfer learning". In ICML, 2007.
    """

    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 n_estimators=10,
                 lr=1.,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)


    def fit(self, X, y, Xt=None, yt=None,
            sample_weight_src=None,
            sample_weight_tgt=None,
            **fit_params):
        """
        Fit TrAdaBoost
        
        Parameters
        ----------
        X : numpy array
            Source input data.

        y : numpy array
            Source output data.
            
        Xt : array (default=None)
            Target input data. If None, the `Xt` argument
            given in `init` is used.

        yt : array (default=None)
            Target input data. If None, the `Xt` argument
            given in `init` is used.
            
        sample_weight_src : numpy array, (default=None)
            Initial sample weight of source data
            
        sample_weight_tgt : numpy array, (default=None)
            Initial sample weight of target data

        fit_params : key, value arguments
            Arguments given to the fit method of the
            estimator.

        Returns
        -------
        self : returns an instance of self
        """
        set_random_seed(self.random_state)
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        Xs, ys = check_arrays(X, y, accept_sparse=True)
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt, accept_sparse=True)
        
        if not isinstance(self, TrAdaBoostR2) and isinstance(self.estimator, BaseEstimator):
            self.label_encoder_ = LabelEncoder()
            ys = self.label_encoder_.fit_transform(ys)
            yt = self.label_encoder_.transform(yt)
        
        n_s = Xs.shape[0]
        n_t = Xt.shape[0]
        
        if sample_weight_src is None:
            sample_weight_src = np.ones(n_s) / (n_s + n_t)
        if sample_weight_tgt is None:
            sample_weight_tgt = np.ones(n_t) / (n_s + n_t)
        
        sum_weights = (sample_weight_src.sum() +
                       sample_weight_tgt.sum())
        sample_weight_src = sample_weight_src / sum_weights
        sample_weight_tgt = sample_weight_tgt / sum_weights

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
                iboost, Xs, ys, Xt, yt,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )
            
            if self.verbose >= 1:
                print("Iteration %i - Error: %.4f"%
                      (iboost, self.estimator_errors_[-1]))

            if sample_weight_src is None:
                break

            sum_weights = (sample_weight_src.sum() +
                           sample_weight_tgt.sum())
            sample_weight_src = sample_weight_src / sum_weights
            sample_weight_tgt = sample_weight_tgt / sum_weights

        self.estimator_weights_ = [
            -np.log(err / (2.-err) + EPS) + 2*EPS
            for err in self.estimator_errors_]
        return self
        
        
    def _boost(self, iboost, Xs, ys, Xt, yt,
               sample_weight_src, sample_weight_tgt,
               **fit_params):
        
        if issparse(Xs):
            X = vstack((Xs, Xt))
        else:
            X = np.concatenate((Xs, Xt))
        y = np.concatenate((ys, yt))
        sample_weight = np.concatenate((sample_weight_src,
                                        sample_weight_tgt))
        
        # Need to rescale sample weight
        estimator = self.fit_estimator(X, y,
                                       sample_weight=sample_weight/sample_weight.mean(),
                                       random_state=None,
                                       warm_start=False,
                                       **fit_params)
        
        if not isinstance(self, TrAdaBoostR2) and isinstance(estimator, BaseEstimator):
            if hasattr(estimator, "predict_proba"):
                ys_pred = estimator.predict_proba(Xs)
                yt_pred = estimator.predict_proba(Xt)
            elif hasattr(estimator, "_predict_proba_lr"):
                ys_pred = estimator._predict_proba_lr(Xs)
                yt_pred = estimator._predict_proba_lr(Xt)
            else:
                ys_pred = estimator.predict(Xs)
                yt_pred = estimator.predict(Xt)
        else:
            ys_pred = estimator.predict(Xs)
            yt_pred = estimator.predict(Xt)
        
        if ys.ndim == 1:
            ys = ys.reshape(-1, 1)
            yt = yt.reshape(-1, 1)
            
        if ys_pred.ndim == 1:
            ys_pred = ys_pred.reshape(-1, 1)
            yt_pred = yt_pred.reshape(-1, 1)
        
        if not isinstance(self, TrAdaBoostR2):
            if isinstance(estimator, BaseEstimator):
                ohe = OneHotEncoder(sparse=False)
                ohe.fit(y.reshape(-1, 1))
                ys = ohe.transform(ys)
                yt = ohe.transform(yt)
                
                if ys_pred.shape[1] == 1:
                    ys_pred = ohe.transform(ys_pred)
                    yt_pred = ohe.transform(yt_pred)
                    
                error_vect_src = np.abs(ys_pred - ys).sum(tuple(range(1, ys.ndim))) / 2.
                error_vect_tgt = np.abs(yt_pred - yt).sum(tuple(range(1, yt.ndim))) / 2.
                    
            else:
                assert np.all(ys_pred.shape == ys.shape)
                error_vect_src = np.abs(ys_pred - ys).sum(tuple(range(1, ys.ndim)))
                error_vect_tgt = np.abs(yt_pred - yt).sum(tuple(range(1, yt.ndim)))
                
                if ys.ndim != 1:
                    error_vect_src /= 2.
                    error_vect_tgt /= 2.
                    
        else:
            error_vect_src = np.abs(ys_pred - ys).mean(tuple(range(1, ys.ndim)))
            error_vect_tgt = np.abs(yt_pred - yt).mean(tuple(range(1, yt.ndim)))
            
            error_max = max(error_vect_src.max(), error_vect_tgt.max())+ EPS
            if error_max > 0:
                error_vect_src /= error_max
                error_vect_tgt /= error_max
        
        # else:
        #     if isinstance(estimator, BaseEstimator):
        #         error_vect_src = (ys_pred != ys).astype(float).ravel()
        #         error_vect_tgt = (yt_pred != yt).astype(float).ravel()
        #         error_vect = np.concatenate((error_vect_src, error_vect_tgt))
        #     else:
        #         if ys.shape[1] == 1:
        #             error_vect_src = (np.abs(ys_pred - ys) > 0.5).astype(float).ravel()
        #             error_vect_tgt = (np.abs(yt_pred - yt) > 0.5).astype(float).ravel()
        #         else:
        #             error_vect_src = (ys_pred.argmax(1) != ys.argmax(1)).astype(float).ravel()
        #             error_vect_tgt = (yt_pred.argmax(1) != yt.argmax(1)).astype(float).ravel()
                
        error_vect = np.concatenate((error_vect_src, error_vect_tgt))
        
        assert sample_weight.ndim == error_vect.ndim

        if isinstance(self, _AdaBoostR2):
            estimator_error = (sample_weight * error_vect).sum()
        else:
            estimator_error = ((sample_weight_tgt * error_vect_tgt).sum() /
                               sample_weight_tgt.sum())
        
        # For multiclassification and regression error can be greater than 0.5
        # if estimator_error > 0.49:
        #     estimator_error = 0.49
        
        self.estimators_.append(estimator)
        self.estimator_errors_.append(estimator_error)
        
        # if estimator_error <= 0.:
        #     return None, None
        
        beta_t = estimator_error / (2. - estimator_error)
        
        beta_s = 1. / (1. + np.sqrt(
            2. * np.log(Xs.shape[0]) / self.n_estimators
        ))
        
        if not iboost == self.n_estimators - 1:
            if isinstance(self, _AdaBoostR2):
                sample_weight_tgt = (sample_weight_tgt *
                np.power(beta_t, self.lr * (1 - error_vect_tgt)))

                sample_weight_tgt *= ((1. - sample_weight_src.sum()) /
                                      sample_weight_tgt.sum())
            else:
                # Source updating weights
                sample_weight_src *= np.power(
                    beta_s, self.lr * error_vect_src)

                # Target updating weights
                sample_weight_tgt *= np.power(
                    beta_t, - self.lr * error_vect_tgt)
        
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
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        N = len(self.estimators_)
        weights = np.array(self.estimator_weights_)
        weights = weights[int(N/2):]
        predictions = []
        for est in self.estimators_[int(N/2):]:
            if isinstance(est, BaseEstimator):
                if hasattr(est, "predict_proba"):
                    y_pred = est.predict_proba(X)
                elif hasattr(est, "_predict_proba_lr"):
                    y_pred = est._predict_proba_lr(X)
                else:
                    labels = est.predict(X)
                    y_pred = np.zeros((len(labels), int(max(labels))+1))
                    y_pred[np.arange(len(labels)), labels] = 1.
            else:
                y_pred = est.predict(X)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                if y_pred.shape[1] == 1:
                    y_pred = np.concatenate((1-y_pred, y_pred),
                                            axis=1)                    
            predictions.append(y_pred)
        predictions = np.stack(predictions, -1)
        weighted_vote = predictions.dot(weights).argmax(1)
        if hasattr(self, "label_encoder_"):
            return self.label_encoder_.inverse_transform(weighted_vote)
        else:
            return weighted_vote


    def predict_weights(self, domain="src"):
        """
        Return sample weights.
        
        Return the final importance weighting.
        
        You can secify between "source" and "target" weights
        with the domain parameter.
        
        Parameters
        ----------
        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"``.

        Returns
        -------
        weights : source sample weights
        """
        if hasattr(self, "sample_weights_src_"):
            if domain in ["src", "source"]:
                return self.sample_weights_src_[-1]
            elif domain in ["tgt", "target"]:
                return self.sample_weights_tgt_[-1]
            else:
                raise ValueError("`domain `argument "
                                 "should be `tgt` or `src`, "
                                 "got, %s"%domain)
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit' first.")
            
    
    def score(self, X, y):
        """
        Return the TrAdaboost score on X, y.
        
        Parameters
        ----------
        X : array
            input data
            
        y : array
            output data
            
        Returns
        -------
        score : float
            estimator score.
        """
        X, y = check_arrays(X, y, accept_sparse=True)
        yp = self.predict(X)
        if isinstance(self, TrAdaBoostR2):
            score = r2_score(y, yp)
        else:
            score = accuracy_score(y, yp)
        return score


@make_insert_doc(supervised=True)
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
    n_estimators : int (default=10)
        Number of boosting iteration.
        
    lr : float (default=1.)
        Learning rate. For higher ``lr``, the sample
        weights are updating faster.

    Attributes
    ----------
    estimators_ : list of object
        List of fitted estimators
        
    estimator_errors_ : 1D array of float
        Array of weighted estimator errors computed on
        labeled target data.
        
    estimator_weights_ : 1D array of float
        Array of estimator importance weights.

    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.
        
    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.
        
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from adapt.utils import make_regression_da
    >>> from adapt.instance_based import TrAdaBoostR2
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> model = TrAdaBoostR2(Ridge(), n_estimators=10, Xt=Xt[:10], yt=yt[:10], random_state=0)
    >>> model.fit(Xs, ys)
    Iteration 0 - Error: 0.4862
    Iteration 1 - Error: 0.5711
    Iteration 2 - Error: 0.6709
    Iteration 3 - Error: 0.7095
    Iteration 4 - Error: 0.7154
    Iteration 5 - Error: 0.6987
    Iteration 6 - Error: 0.6589
    Iteration 7 - Error: 0.5907
    Iteration 8 - Error: 0.4930
    Iteration 9 - Error: 0.3666
    >>> model.score(Xt, yt)
    0.6998064452649377

    See also
    --------
    TrAdaBoost, TwoStageTrAdaBoostR2

    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
    """

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
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        N = len(self.estimators_)
        weights = np.array(self.estimator_weights_)
        weights = weights[int(N/2):]
        predictions = []
        for est in self.estimators_[int(N/2):]:
            y_pred = est.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            predictions.append(y_pred)
        predictions = np.stack(predictions, -1)
        return _get_median_predict(predictions, weights)


class _AdaBoostR2(TrAdaBoostR2):
    """
    AdaBoostR2 object with fixed sample weights.
    """
    pass


@make_insert_doc(supervised=True)    
class TwoStageTrAdaBoostR2(TrAdaBoostR2):
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
    n_estimators : int (default=10)
        Number of boosting iteration.
        
    n_estimators_fs : int (default=10)
        Number of boosting iteration in first stage
        (given to AdaboostR2 estimators)
        
    cv: int, optional (default=5)
        Split cross-validation parameter.
        
    lr : float (default=1.)
        Learning rate. For higher ``lr``, the sample
        weights are updating faster.

    Attributes
    ----------
    estimators_ : list of object
        List of fitted AdaboostR2 estimators for each
        first stage.
        
    estimator_errors_ : 1D array of float
        Array of cross-validation MAE computed on
        labeled target data.

    sample_weights_src_ : list of numpy arrays
        List of source sample weight for each iteration.

    sample_weights_tgt_ : list of numpy arrays
        List of target sample weight for each iteration.
        
    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from adapt.utils import make_regression_da
    >>> from adapt.instance_based import TwoStageTrAdaBoostR2
    >>> Xs, ys, Xt, yt = make_regression_da()
    >>> model = TwoStageTrAdaBoostR2(Ridge(), n_estimators=10, Xt=Xt[:10], yt=yt[:10], random_state=0)
    >>> model.fit(Xs, ys)
    Iteration 0 - Cross-validation score: 0.2956 (0.0905)
    Iteration 1 - Cross-validation score: 0.2956 (0.0905)
    Iteration 2 - Cross-validation score: 0.2614 (0.1472)
    Iteration 3 - Cross-validation score: 0.2701 (0.1362)
    Iteration 4 - Cross-validation score: 0.2745 (0.1280)
    Iteration 5 - Cross-validation score: 0.2768 (0.1228)
    Iteration 6 - Cross-validation score: 0.2782 (0.1195)
    Iteration 7 - Cross-validation score: 0.2783 (0.1170)
    Iteration 8 - Cross-validation score: 0.2767 (0.1156)
    Iteration 9 - Cross-validation score: 0.2702 (0.1155)
    >>> model.score(Xt, yt)
    0.5997110100905185

    See also
    --------
    TrAdaBoost, TrAdaBoostR2

    References
    ----------
    .. [1] `[1] <https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf>`_ \
D. Pardoe and P. Stone. "Boosting for regression transfer". In ICML, 2010.
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 n_estimators=10,
                 n_estimators_fs=10,
                 cv=5,
                 lr=1.,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        super().__init__(estimator=estimator,
                         Xt=Xt,
                         yt=yt,
                         n_estimators=n_estimators,
                         n_estimators_fs = n_estimators_fs,
                         cv=cv,
                         lr=lr,
                         copy=copy,
                         verbose=verbose,
                         random_state=random_state,
                         **params)


    def fit(self, X, y, Xt=None, yt=None,
            sample_weight_src=None,
            sample_weight_tgt=None,
            **fit_params):
        set_random_seed(self.random_state)
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        Xs, ys = check_arrays(X, y, accept_sparse=True)
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt, accept_sparse=True)
        
        n_s = Xs.shape[0]
        n_t = Xt.shape[0]

        if sample_weight_src is None:
            sample_weight_src = np.ones(n_s) / (n_s + n_t)
        if sample_weight_tgt is None:
            sample_weight_tgt = np.ones(n_t) / (n_s + n_t)
        
        sum_weights = (sample_weight_src.sum() +
                       sample_weight_tgt.sum())
        sample_weight_src = sample_weight_src / sum_weights
        sample_weight_tgt = sample_weight_tgt / sum_weights

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
                Xs, ys, Xt, yt,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )
            
            if self.verbose >= 1:
                print("Iteration %i - Cross-validation score: %.4f (%.4f)"%
                      (iboost, np.mean(cv_score), np.std(cv_score)))
            
            sample_weight_src, sample_weight_tgt = self._boost(
                iboost, Xs, ys, Xt, yt,
                sample_weight_src, sample_weight_tgt,
                **fit_params
            )
            
            self.estimator_errors_.append(cv_score.mean())

            if sample_weight_src is None:
                break

            sum_weights = (sample_weight_src.sum() +
                           sample_weight_tgt.sum())
            sample_weight_src = sample_weight_src / sum_weights
            sample_weight_tgt = sample_weight_tgt / sum_weights

        return self


    def _boost(self, iboost, Xs, ys, Xt, yt,
               sample_weight_src, sample_weight_tgt, **fit_params):

        estimator = _AdaBoostR2(estimator=self.estimator,
                                n_estimators=self.n_estimators_fs,
                                lr=self.lr,
                                verbose=self.verbose-1,
                                random_state=None)
        
        if self.verbose > 1:
            print("First Stages...")
        
        estimator.fit(Xs, ys,
                      Xt=Xt, yt=yt,
                      sample_weight_src=sample_weight_src,
                      sample_weight_tgt=sample_weight_tgt,
                      **fit_params)
        
        ys_pred = estimator.predict(Xs)
        yt_pred = estimator.predict(Xt)
        
        if ys_pred.ndim == 1 or ys.ndim == 1:
            ys = ys.reshape(-1, 1)
            yt = yt.reshape(-1, 1)
            ys_pred = ys_pred.reshape(-1, 1)
            yt_pred = yt_pred.reshape(-1, 1)
        
        error_vect_src = np.abs(ys_pred - ys).mean(tuple(range(1, ys.ndim)))
        error_vect_tgt = np.abs(yt_pred - yt).mean(tuple(range(1, yt.ndim)))
        error_vect = np.concatenate((error_vect_src, error_vect_tgt))
        
        error_max = error_vect.max() + EPS
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
                beta, self.lr * error_vect_src
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
        return _binary_search(func, self.verbose)


    def _cross_val_score(self, Xs, ys, Xt, yt,
                         sample_weight_src, sample_weight_tgt,
                         **fit_params):
        if Xt.shape[0] >= self.cv:
            cv = self.cv
        else:
            cv = Xt.shape[0]
        
        tgt_index = np.arange(Xt.shape[0])
        split = int(Xt.shape[0] / cv)
        scores = []
        for i in range(cv):
            if i == cv-1:
                test_index = tgt_index[i * split:]
            else:
                test_index = tgt_index[i * split: (i + 1) * split]
            train_index = list(set(tgt_index) - set(test_index))
            
                    
            if issparse(Xs):
                X = vstack((Xs, Xt[train_index]))
            else:
                X = np.concatenate((Xs, Xt[train_index]))
            y = np.concatenate((ys, yt[train_index]))
            sample_weight = np.concatenate((sample_weight_src,
                                            sample_weight_tgt[train_index]))
            if (len(train_index) > 0 and
                sample_weight_tgt[train_index].sum() != 0):
                sample_weight[-len(train_index):] *= (
                    sample_weight_tgt.sum() / 
                    sample_weight_tgt[train_index].sum()
                )

            estimator = self.fit_estimator(X, y,
                               sample_weight=sample_weight/sample_weight.mean(),
                               random_state=None,
                               warm_start=False,
                               **fit_params)

            y_pred = estimator.predict(Xt[test_index])
            
            y_pred = y_pred.reshape(yt[test_index].shape)
            
            scores.append(np.abs(y_pred - yt[test_index]).mean())
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
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        best_estimator = self.estimators_[
            np.argmin(self.estimator_errors_)]
        return best_estimator.predict(X)


    def predict_weights(self, domain="src"):
        """
        Return sample weights.
        
        Return the source importance weighting
        of the best estimator.
        
        You can secify between "source" and "target" weights
        with the domain parameter.
        
        Parameters
        ----------
        domain : str (default="tgt")
            Choose between ``"source", "src"`` and
            ``"target", "tgt"``.

        Returns
        -------
        weights : source sample weights
        """
        if hasattr(self, "sample_weights_src_"):
            arg = np.argmin(self.estimator_errors_)
            if domain in ["src", "source"]:
                return self.sample_weights_src_[arg]
            elif domain in ["tgt", "target"]:
                return self.sample_weights_tgt_[arg]
            else:
                raise ValueError("`domain `argument "
                                 "should be `tgt` or `src`, "
                                 "got, %s"%domain)
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit' first.")