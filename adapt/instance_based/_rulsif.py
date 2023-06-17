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
class RULSIF(BaseAdaptEstimator):
    """
    RULSIF: Relative Unconstrained Least-Squares Importance Fitting
    
    RULSIF is an instance-based method for domain adaptation. 
    
    The purpose of the algorithm is to correct the difference between
    input distributions of source and target domains. This is done by
    finding a source instances **reweighting** which minimizes the 
    **relative Person divergence** between source and target distributions.
    
    The source instance weights are given by the following formula:
    
    .. math::
    
        w(x) = \sum_{x_i \in X_T} \\theta_i K(x, x_i)
        
    Where:
    
    - :math:`x, x_i` are input instances.
    - :math:`X_T` is the target input data of size :math:`n_T`.
    - :math:`\\theta_i` are the basis functions coefficients.
    - :math:`K(x, x_i) = \\text{exp}(-\\gamma ||x - x_i||^2)`
      for instance if ``kernel="rbf"``.
      
    RULSIF algorithm consists in finding the optimal :math:`\\theta` according to
    the quadratic problem 
    
    .. math::
    
        \max_{\\theta } \\frac{1}{2}  \\theta^T H \\theta - h^T \\theta  + 
        \\frac{\\lambda}{2} \\theta^T \\theta
        
    where :
    
    .. math::
    
        H_{kl} = \\frac{\\alpha}{n_s} \sum_{x_i \\in X_S}  K(x_i, x_k) K(x_i, x_l) + \\frac{1-\\alpha}{n_T} \\sum_{x_i \\in X_T}  K(x_i, x_k) K(x_i, x_l)
        
    .. math::
    
        h_{k} = \\frac{1}{n_T} \sum_{x_i \\in X_T} K(x_i, x_k)
    
    The above OP is solved by the closed form expression
    
    .. math::
    
        \hat{\\theta}=(H+\\lambda I_{n_s})^{(-1)} h 
    
    Furthemore the method admits a leave one out cross validation score that has a clossed expression 
    and can be used to select the appropriate parameters of the kernel function :math:`K` (typically, the parameter
    :math:`\\gamma` of the Gaussian kernel). The parameter is then selected using
    cross-validation on the :math:`J` score defined as follows:
    
    .. math::
    
        J = -\\frac{\\alpha}{2|X_S|} \\sum_{x \\in X_S} w(x)^2 - \\frac{1-\\alpha}{2|X_T|} \\sum_{x \in X_T} w(x)^2
    
    Finally, an estimator is fitted using the reweighted labeled source instances.
    
    RULSIF method has been originally introduced for **unsupervised**
    DA but it could be widen to **supervised** by simply adding labeled
    target data to the training set.
    
    Parameters
    ----------
    kernel : str (default="rbf")
        Kernel metric.
        Possible values: [‘additive_chi2’, ‘chi2’,
        ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
        ‘laplacian’, ‘sigmoid’, ‘cosine’]
        
    alpha : float (default=0.1)
        Trade-off parameter (between 0 and 1)
        
    lambdas : float or list of float (default=1.)
        Optimization parameter. If a list is given,
        the best lambda will be selected on
        the unsupervised Leave-One-Out J-score.

    max_centers : int (default=100)
        Maximal number of target instances use to
        compute kernels.
        
 
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
        
    thetas_ : numpy array
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
    >>> from adapt.instance_based import RULSIF
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = RULSIF(RidgeClassifier(0.), Xt=Xt, kernel="rbf", alpha=0.1,
    ...                lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], random_state=0)
    >>> model.fit(Xs, ys);
    >>> model.score(Xt, yt)
    0.71

    See also
    --------
    ULSIF
    KLIEP
    
    References
    ----------
    .. [1] `[1] <https://proceedings.neurips.cc/paper/2011/file/\
d1f255a373a3cef72e03aa9d980c7eca-Paper.pdf>`_ \
M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya and  M. Sugiyama. \
"Relative Density-Ratio Estimation for Robust Distribution Comparison". In NIPS 2011
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 kernel="rbf",
                 alpha=0.1,
                 lambdas=1.,
                 max_centers=100,
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
       
        self.j_scores_ = {}
        
        # LCV GridSearch
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}
        
        
        kernel_params_dict = {k:(v if hasattr(v, "__iter__") else [v]) for k, v in kernel_params.items()}
        lambdas_params_dict={"lamb":(self.lambdas if hasattr(self.lambdas, "__iter__") else [self.lambdas])}
        options = kernel_params_dict
        keys = options.keys()
        values = (options[key] for key in keys)
        params_comb_kernel = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
        if len(params_comb_kernel)*len(lambdas_params_dict["lamb"]) > 1:            
            if self.verbose:
                print("Cross Validation process...")
            # Cross-validation process 
            max_ = -np.inf
            N_s=len(Xs)
            N_t=len(Xt)
            N_min = min(N_s, N_t)
            index_centers = np.random.choice(
                            len(Xt),
                            min(len(Xt), self.max_centers),
                            replace=False)
            centers = Xt[index_centers]
            n_centers=min(len(Xt), self.max_centers)
            
            if N_s<N_t:
                index_data = np.random.choice(
                                N_t,
                                N_s,
                                replace=False)
            elif N_t<N_s:
                index_data = np.random.choice(
                                N_s,
                                N_t,
                                replace=False)
                
            
            for params in params_comb_kernel:
              
                if N_s<N_t:
                    phi_t = pairwise.pairwise_kernels(centers,Xt[index_data], metric=self.kernel,
                                                  **params)
                    phi_s = pairwise.pairwise_kernels(centers,Xs, metric=self.kernel,
                                                  **params)  
                elif N_t<N_s:
                    phi_t = pairwise.pairwise_kernels(centers,Xt, metric=self.kernel,
                                                  **params)
                    phi_s = pairwise.pairwise_kernels(centers,Xs[index_data], metric=self.kernel,
                                                  **params) 
                else:
                    phi_t = pairwise.pairwise_kernels(centers,Xt, metric=self.kernel,
                                                  **params)
                    phi_s = pairwise.pairwise_kernels(centers,Xs, metric=self.kernel,
                                                  **params) 
                    
                
                H=self.alpha*np.dot(phi_t, phi_t.T) / N_t + (1-self.alpha)*np.dot(phi_s, phi_s.T) / N_s          
                h = np.mean(phi_t, axis=1)
                h = h.reshape(-1, 1)


                for lamb in lambdas_params_dict["lamb"]:
                    B = H + np.identity(n_centers) * (lamb * (N_t - 1) / N_t)
                    BinvX = np.linalg.solve(B, phi_t)
                    XBinvX = phi_t * BinvX
                    D0 = np.ones(N_min) * N_t- np.dot(np.ones(n_centers), XBinvX)
                    diag_D0 = np.diag((np.dot(h.T, BinvX) / D0).ravel())
                    B0 = np.linalg.solve(B, h * np.ones(N_min)) + np.dot(BinvX, diag_D0)
                    diag_D1 = np.diag(np.dot(np.ones(n_centers), phi_s * BinvX).ravel())
                    B1 = np.linalg.solve(B,  phi_s) + np.dot(BinvX, diag_D1)
                    B2 = (N_t- 1) * (N_s* B0 - B1) / (N_t* (N_s - 1))
                    B2[B2<0]=0
                    r_s = (phi_s * B2).sum(axis=0).T
                    r_t= (phi_t * B2).sum(axis=0).T
                    score = ((1-self.alpha)*(np.dot(r_s.T, r_s).ravel() / 2. + self.alpha*np.dot(r_t.T, r_t).ravel() / 2.  - r_t.sum(axis=0)) /N_min).item()  # LOOCV
                    aux_params={"k":params,"lamb":lamb}
                    self.j_scores_[str(aux_params)]=-1*score
                       
                    if self.verbose:
                        print("Parameters %s -- J-score = %.3f"% (str(aux_params),score))
                    if self.j_scores_[str(aux_params)] > max_:
                        self.best_params_ = aux_params
                        max_ = self.j_scores_[str(aux_params)]
        else:
            self.best_params_ = {"k":params_comb_kernel[0],"lamb": lambdas_params_dict["lamb"]}


        self.thetas_, self.centers_ = self._fit(Xs, Xt, self.best_params_["k"],self.best_params_["lamb"])

        self.weights_ = np.dot(
            pairwise.pairwise_kernels(Xs, self.centers_,
                                     metric=self.kernel,
                                     **self.best_params_["k"]),
            self.thetas_
            ).ravel()
        return self.weights_
    
    
    def predict_weights(self, X=None):
        """
        Return fitted source weights
        
        If ``None``, the fitted source weights are returned.
        Else, sample weights are computing using the fitted
        ``thetas_`` and the chosen ``centers_``.
        
        Parameters
        ----------
        X : array (default=None)
            Input data.
        
        Returns
        -------
        weights_ : sample weights
        """
        if hasattr(self, "weights_"):
            if X is None or not hasattr(self, "thetas_"):
                return self.weights_
            else:
                X = check_array(X)
                weights = np.dot(
                pairwise.pairwise_kernels(X,self.centers_,
                                         metric=self.kernel,
                                         **self.best_params_["k"]),
                self.thetas_
                ).ravel()
                return weights
        else:
            raise NotFittedError("Weights are not fitted yet, please "
                                 "call 'fit_weights' or 'fit' first.")


    def _fit(self, Xs, Xt, kernel_params,lamb):
        index_centers = np.random.choice(
                        len(Xt),
                        min(len(Xt), self.max_centers),
                        replace=False)
        centers = Xt[index_centers]
        n_centers=min(len(Xt), self.max_centers)
        
        phi_t = pairwise.pairwise_kernels( centers,Xt, metric=self.kernel,
                                      **kernel_params)
        phi_s = pairwise.pairwise_kernels(centers,Xs, metric=self.kernel,
                                      **kernel_params)
   
        N_t=len(Xt)
        N_s=len(Xs)
        
        H=self.alpha*np.dot(phi_t, phi_t.T) / N_t + (1-self.alpha)*np.dot(phi_s, phi_s.T) / N_s
        h = np.mean(phi_t, axis=1)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+lamb*np.eye(n_centers), h)
        theta[theta<0]=0
        return theta, centers
