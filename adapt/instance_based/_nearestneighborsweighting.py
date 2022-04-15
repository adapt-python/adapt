import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import check_array

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed


@make_insert_doc()
class NearestNeighborsWeighting(BaseAdaptEstimator):
    """
    NNW : Nearest Neighbors Weighting
    
    NNW reweights the source instances in order according to
    their number of neighbors in the target dataset.
    
    Parameters
    ----------
    n_neighbors : int, (default=5)
        Number of neighbors to use by default for `kneighbors` queries.
    
    radius : float, (default=1.0)
        Range of parameter space to use by default for `radius_neighbors`
        queries.
    
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, (default='auto')
        Algorithm used to compute the nearest neighbors:
        
        - 'ball_tree' will use ``BallTree``
        - 'kd_tree' will use ``KDTree``
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to ``fit`` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    
    leaf_size : int, (default=30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    
    metric : str or callable, (default='minkowski')
        The distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. For a list of available metrics, see the documentation of
        `sklearn.metrics.DistanceMetric`.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`,
        in which case only "nonzero" elements may be considered neighbors.
    
    p : int, (default=2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    
    metric_params : dict, (default=None)
        Additional keyword arguments for the metric function.
    
    n_jobs : int, (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a `joblib.parallel_backend` context.
        ``-1`` means using all processors.
    
    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
    
    estimator_ : object
        Estimator.
    
    See also
    --------
    KMM
    KLIEP
    
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import NearestNeighborsWeighting
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = NearestNeighborsWeighting(RidgeClassifier(), n_neighbors=5, Xt=Xt, random_state=0)
    >>> model.fit(Xs, ys)
    Fit weights...
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.66
    
    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/2102.02291.pdf>`_ \
M. Loog. "Nearest neighbor-based importance weighting". In MLSP 2012.
    """
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 n_neighbors=5,
                 radius=1.0,
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 metric_params=None,
                 n_jobs=None,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
        
    def fit_weights(self, Xs, Xt, **kwargs):
        """
        Fit importance weighting.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        kwargs : key, value argument
            Not used, present here for adapt consistency.
            
        Returns
        -------
        weights_ : sample weights
        """
        Xs = check_array(Xs)
        Xt = check_array(Xt)
        set_random_seed(self.random_state)
                
        nn_model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                    radius=self.radius,
                                    algorithm=self.algorithm,
                                    leaf_size=self.leaf_size,
                                    metric=self.metric,
                                    p=self.p,
                                    metric_params=self.metric_params,
                                    n_jobs=self.n_jobs)
        nn_model.fit(Xs)
        
        args = nn_model.kneighbors(Xt, return_distance=False)
        args = args.ravel()
        
        indices, weights = np.unique(args, return_counts=True)
        
        self.weights_ = np.zeros(len(Xs))
        self.weights_[indices] = weights
        return self.weights_
    
    
    def predict_weights(self):
        """
        Return fitted source weights
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            return self.weights_
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")