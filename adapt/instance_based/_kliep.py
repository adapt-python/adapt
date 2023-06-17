"""
Kullback-Leibler Importance Estimation Procedure
"""
import itertools
import warnings

import numpy as np
from sklearn.metrics import pairwise
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.metrics.pairwise import KERNEL_PARAMS

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import set_random_seed

EPS = np.finfo(float).eps


@make_insert_doc()
class KLIEP(BaseAdaptEstimator):
    """
    KLIEP: Kullback–Leibler Importance Estimation Procedure
    
    KLIEP is an instance-based method for domain adaptation. 
    
    The purpose of the algorithm is to correct the difference between
    input distributions of source and target domains. This is done by
    finding a source instances **reweighting** which minimizes the 
    **Kullback-Leibler divergence** between source and target distributions.
    
    The source instance weights are given by the following formula:
    
    .. math::
    
        w(x) = \sum_{x_i \in X_T} \\alpha_i K(x, x_i)
        
    Where:
    
    - :math:`x, x_i` are input instances.
    - :math:`X_T` is the target input data.
    - :math:`\\alpha_i` are the basis functions coefficients.
    - :math:`K(x, x_i) = \\text{exp}(-\\gamma ||x - x_i||^2)`
      for instance if ``kernel="rbf"``.
      
    KLIEP algorithm consists in finding the optimal :math:`\\alpha_i` according to
    the following optimization problem:
    
    .. math::
    
        \max_{\\alpha_i } \sum_{x_j \in X_T} \log(
        \sum_{x_i \in X_T} \\alpha_i K(x_j, x_i))
        
    Subject to:
    
    .. math::
    
        \sum_{x_j \in X_S} \sum_{x_i \in X_T} \\alpha_i K(x_j, x_i)) = n_S
        
    Where:
    
    - :math:`X_S` is the source input data of size :math:`n_S`.
    
    The above OP is solved through gradient ascent algorithm.
    
    Furthemore a LCV procedure can be added to select the appropriate
    parameters of the kernel function :math:`K` (typically, the paramter
    :math:`\\gamma` of the Gaussian kernel). The parameter is then selected using
    cross-validation on the :math:`J` score defined as follows:
    :math:`J = \\frac{1}{|\\mathcal{X}|} \\sum_{x \\in \\mathcal{X}} \\text{log}(w(x))`
    
    Finally, an estimator is fitted using the reweighted labeled source instances.
    
    KLIEP method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.
    
    Parameters
    ----------
    kernel : str (default="rbf")
        Kernel metric.
        Possible values: [‘additive_chi2’, ‘chi2’,
        ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
        ‘laplacian’, ‘sigmoid’, ‘cosine’]
    
    sigmas : float or list of float (default=None)
        Deprecated, please use the ``gamma`` parameter
        instead. (See below).
        
    cv : int (default=5)
        Cross-validation split parameter.
        Used only if sigmas has more than one value.
        
    max_centers : int (default=100)
        Maximal number of target instances use to
        compute kernels.
    
    algo : str (default="FW")
        Optimization algorithm.
        Possible values: ['original', 'PG', 'FW']
        
        - 'original' follows the algorithm of [1]. Useful to reproduce the paper's experiences.
        - 'PG' is a improved version of 'original'. A convex projection into the constraints set is used.
        - 'FW' [2] uses the Frank-Wolfe algorithm to solve the above OP.
        
        In general, 'FW' is more efficient than 'original' or 'PG'. 
        In some cases, 'PG' converges faster than 'FW' with a good choice of learning rate.
        
        
    lr : float or list of float (default=np.logspace(-3,1,5))
        Learning rate of the gradient ascent.
        Used only if algo different to 'FW'
        
    tol : float (default=1e-6)
        Optimization threshold.
        
    max_iter : int (default=2000)
        Maximal iteration of the optimization algorithm.
        
    Yields
    ------
    gamma : float or list of float
        Kernel parameter ``gamma``.
        
        - For kernel = chi2::
        
            k(x, y) = exp(-gamma Sum [(x - y)^2 / (x + y)])

        - For kernel = poly or polynomial::
        
            K(X, Y) = (gamma <X, Y> + coef0)^degree
            
        - For kernel = rbf::
        
            K(x, y) = exp(-gamma ||x-y||^2)
        
        - For kernel = laplacian::
        
            K(x, y) = exp(-gamma ||x-y||_1)
        
        - For kernel = sigmoid::
        
            K(X, Y) = tanh(gamma <X, Y> + coef0)
            
        If a list is given, the LCV process is performed to
        select the best parameter ``gamma``.
        
    coef0 : floaf or list of float
        Kernel parameter ``coef0``.
        Used for ploynomial and sigmoid kernels.
        See ``gamma`` parameter above for the 
        kernel formulas.
        If a list is given, the LCV process is performed to
        select the best parameter ``coef0``.
        
    degree : int or list of int
        Degree parameter for the polynomial
        kernel. (see formula in the ``gamma``
        parameter description).
        If a list is given, the LCV process is performed to
        select the best parameter ``degree``.

    Attributes
    ----------
    weights_ : numpy array
        Training instance weights.
        
    best_params_ : float
        Best kernel params combination
        deduced from the LCV procedure.
        
    alphas_ : numpy array
        Basis functions coefficients.
        
    centers_ : numpy array
        Center points for kernels.
        
    j_scores_ : dict
        dict of J scores with the
        kernel params combination as
        keys and the J scores as values.
        
    estimator_ : object
        Fitted estimator.
        
    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from adapt.utils import make_classification_da
    >>> from adapt.instance_based import KLIEP
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = KLIEP(RidgeClassifier(), Xt=Xt, kernel="rbf", gamma=[0.1, 1.], random_state=0)
    >>> model.fit(Xs, ys)
    Fit weights...
    Cross Validation process...
    Parameter {'gamma': 0.1} -- J-score = 0.013 (0.003)
    Parameter {'gamma': 1.0} -- J-score = 0.120 (0.026)
    Fit Estimator...
    >>> model.score(Xt, yt)
    0.85

    See also
    --------
    KMM

    References
    ----------
    .. [1] `[1] <https://papers.nips.cc/paper/3248-direct-importance-estimation\
-with-model-selection-and-its-application-to-covariate-shift-adaptation.pdf>`_ \
M. Sugiyama, S. Nakajima, H. Kashima, P. von Bünau and  M. Kawanabe. \
"Direct importance estimation with model selection and its application \
to covariateshift adaptation". In NIPS 2007
    .. [2] `[2] <https://webdocs.cs.ualberta.ca/~dale/papers/ijcai15.pdf>`_ \
J. Wen, R. Greiner and D. Schuurmans. \
"Correcting Covariate Shift with the Frank-Wolfe Algorithm". In IJCAI 2015
"""
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 kernel="rbf",
                 sigmas=None,
                 max_centers=100,
                 cv=5,
                 algo="FW",
                 lr=[0.001, 0.01, 0.1, 1.0, 10.0],
                 tol=1e-6,
                 max_iter=2000,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        if sigmas is not None:
            warnings.warn("The `sigmas` argument is deprecated, "
              "please use the `gamma` argument instead.",
              DeprecationWarning)
        
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
        
        self.j_scores_ = {}
        
        # LCV GridSearch
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}
        
        # Handle deprecated sigmas (will be removed)
        if (self.sigmas is not None) and (not "gamma" in kernel_params):
            kernel_params["gamma"] = self.sigmas
        
        params_dict = {k: (v if hasattr(v, "__iter__") else [v]) for k, v in kernel_params.items()}
        options = params_dict
        keys = options.keys()
        values = (options[key] for key in keys)
        params_comb = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

        if len(params_comb) > 1:
            # Cross-validation process   
            if len(Xt) < self.cv:
                raise ValueError("Length of Xt is smaller than cv value")

            if self.verbose:
                print("Cross Validation process...")

            shuffled_index = np.arange(len(Xt))
            np.random.shuffle(shuffled_index)
            
            max_ = -np.inf
            for params in params_comb:
                cv_scores = self._cross_val_jscore(Xs, Xt[shuffled_index], params, self.cv)
                self.j_scores_[str(params)] = np.mean(cv_scores)

                if self.verbose:
                    print("Parameters %s -- J-score = %.3f (%.3f)"%
                          (str(params), np.mean(cv_scores), np.std(cv_scores)))

                if self.j_scores_[str(params)] > max_:
                    self.best_params_ = params
                    max_ = self.j_scores_[str(params)]
        else:
            self.best_params_ = params_comb[0]

        self.alphas_, self.centers_ = self._fit(Xs, Xt, self.best_params_)

        self.weights_ = np.dot(
            pairwise.pairwise_kernels(Xs, self.centers_,
                                     metric=self.kernel,
                                     **self.best_params_),
            self.alphas_
            ).ravel()
        return self.weights_
    
    
    def predict_weights(self, X=None):
        """
        Return fitted source weights
        
        If ``None``, the fitted source weights are returned.
        Else, sample weights are computing using the fitted
        ``alphas_`` and the chosen ``centers_``.
        
        Parameters
        ----------
        X : array (default=None)
            Input data.
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            if X is None or not hasattr(self, "alphas_"):
                return self.weights_
            else:
                X = check_array(X)
                weights = np.dot(
                pairwise.pairwise_kernels(X, self.centers_,
                                         metric=self.kernel,
                                         **self.best_params_),
                self.alphas_
                ).ravel()
                return weights
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")

          
    def _fit(self, Xs, Xt, kernel_params):
        if self.algo == "original":
            return self._fit_PG(Xs, Xt, PG=False, kernel_params=kernel_params)
        elif self.algo == "PG":
            return self._fit_PG(Xs, Xt, PG=True, kernel_params=kernel_params)
        elif self.algo == "FW":
            return self._fit_FW(Xs, Xt, kernel_params=kernel_params)
        else :
            raise ValueError("%s is not a valid value of algo"%self.algo)
            
    
    def _fit_PG(self, Xs, Xt, PG, kernel_params):
        alphas = []
        OBJs = []
        
        if type(self.lr) == float or type(self.lr) == int:
            LRs = [self.lr]
        elif type(self.lr) == list or type(self.lr) == np.ndarray:
            LRs = self.lr
        else:
            raise TypeError("invalid argument type for lr")
        
        # For original, no center selection
        if PG:
            centers, A, b = self._centers_selection(Xs, Xt, kernel_params)
        else:
            index_centers = np.random.choice(
                        len(Xt),
                        min(len(Xt), self.max_centers),
                        replace=False)
            centers = Xt[index_centers]
            
            A = pairwise.pairwise_kernels(Xt, centers, metric=self.kernel,
                                      **kernel_params)
            B = pairwise.pairwise_kernels(centers, Xs, metric=self.kernel,
                                          **kernel_params)
            b = np.mean(B, axis=1)
            b = b.reshape(-1, 1)

        for lr in LRs:
            if self.verbose > 1:
                print("learning rate : %s"%lr)
            
            # For original, init alpha = ones and project
            if PG:
                alpha = 1/(len(centers)*b)
            else:
                alpha = np.ones((len(centers), 1))
                alpha = self._projection_original(alpha, b)
                
            alpha = alpha.reshape(-1,1)
            previous_objective = -np.inf
            objective = np.sum(np.log(np.dot(A, alpha) + EPS))
            if self.verbose > 1:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
            k = 0
            while k < self.max_iter and objective-previous_objective > self.tol:
                previous_objective = objective
                alpha_p = np.copy(alpha)
                r = 1./np.clip(np.dot(A, alpha), EPS, np.inf)
                g = np.dot(
                    np.transpose(A), r
                )
                alpha += lr * g
                if PG :
                    alpha = self._projection_PG(alpha, b).reshape(-1,1)
                else :
                    alpha = self._projection_original(alpha, b)
                objective = np.sum(np.log(np.dot(A, alpha) + EPS))
                k += 1

                if self.verbose > 1:
                    if k%100 == 0:
                        print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))
            alphas.append(alpha_p)
            OBJs.append(previous_objective)
        OBJs = np.array(OBJs).ravel()
        return alphas[np.argmax(OBJs)], centers
    
    def _fit_FW(self, Xs, Xt, kernel_params):
        centers, A, b = self._centers_selection(Xs, Xt, kernel_params)

        alpha = 1/(len(centers)*b)
        alpha = alpha.reshape(-1,1)
        objective = np.sum(np.log(np.dot(A, alpha) + EPS))
        if self.verbose > 1:
                print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
        k = 0
        while k < self.max_iter:
            previous_objective = objective
            alpha_p = np.copy(alpha)
            r = 1./np.clip(np.dot(A, alpha), EPS, np.inf)
            g = np.dot(
                np.transpose(A), r
            )
            B = np.diag(1/b.ravel())
            LP = np.dot(g.transpose(), B)
            lr = 2/(k+2)
            alpha = (1 - lr)*alpha + lr*B[np.argmax(LP)].reshape(-1,1)
            objective = np.sum(np.log(np.dot(A, alpha) + EPS))
            k += 1
            
            if self.verbose > 1:
                if k%100 == 0:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))
        return alpha, centers
    
    def _centers_selection(self, Xs, Xt, kernel_params):
        A = np.empty((Xt.shape[0], 0))
        b = np.empty((0,))
        centers = np.empty((0, Xt.shape[1]))
        
        max_centers = min(len(Xt), self.max_centers)
        np.random.seed(self.random_state)
        index = np.random.permutation(Xt.shape[0])
        
        k = 0
        
        while k*max_centers < len(index) and len(centers) < max_centers and k<3:
            index_ = index[k*max_centers:(k+1)*max_centers]
            centers_ = Xt[index_]
            A_ = pairwise.pairwise_kernels(Xt, centers_, metric=self.kernel,
                                      **kernel_params)
            B_ = pairwise.pairwise_kernels(centers_, Xs, metric=self.kernel,
                                          **kernel_params)
            b_ = np.mean(B_, axis=1)
            mask = (b_ < EPS).ravel()
            if np.sum(~mask) > 0 :
                centers_ = centers_[~mask]
                centers = np.concatenate((centers, centers_), axis = 0)
                A = np.concatenate((A, A_[:,~mask]), axis=1)
                b = np.append(b, b_[~mask])
            k += 1
            
        if len(centers) >= max_centers:
            centers = centers[:max_centers]
            A = A[:, :max_centers]
            b = b[:max_centers]
        elif len(centers) > 0:
            warnings.warn("Not enough centers, only %i centers found. Maybe consider a different value of kernel parameter."%len(centers))
        else:
            raise ValueError("No centers found! Please change the value of kernel parameter.")
        
        return centers, A, b.reshape(-1,1)

    def _projection_original(self, alpha, b):
        alpha += b * ((((1-np.dot(np.transpose(b), alpha)) /
                            (np.dot(np.transpose(b), b) + EPS))))
        alpha = np.maximum(0, alpha)
        alpha /= (np.dot(np.transpose(b), alpha) + EPS)
        return alpha
    
    def _projection_PG(self, y, b):
        sort= np.argsort(y.ravel()/b.ravel())
        y_hat = np.array(y).ravel()[sort]
        b_hat = np.array(b).ravel()[sort]
        nu = [(np.dot(y_hat[k:],b_hat[k:])-1)/np.dot(b_hat[k:], b_hat[k:]) for k in range(len(y_hat))]
        k = 0
        for i in range(len(nu)):
            if i == 0 :
                if nu[i]<=y_hat[i]/b_hat[i]:
                    break
            elif (nu[i]>y_hat[i-1]/b_hat[i-1] and nu[i]<=y_hat[i]/b_hat[i]):
                    k = i
                    break
        return np.maximum(0, y-nu[k]*b)
    
    
    def _cross_val_jscore(self, Xs, Xt, kernel_params, cv):        
        split = int(len(Xt) / cv)
        cv_scores = []
        for i in range(cv):
            test_index = np.arange(i * split, (i + 1) * split)
            train_index = np.array(
                list(set(np.arange(len(Xt))) - set(test_index))
            )
            
            try:
                alphas, centers = self._fit(Xs,
                                            Xt[train_index],
                                            kernel_params)

                j_score = np.mean(np.log(
                    np.dot(
                        pairwise.pairwise_kernels(Xt[test_index],
                                                 centers,
                                                 metric=self.kernel,
                                                 **kernel_params),
                        alphas
                    ) + EPS
                ))
            except Exception as e:
                j_score = -np.inf
                
            cv_scores.append(j_score)
        return cv_scores
