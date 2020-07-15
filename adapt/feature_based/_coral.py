"""
Correlation Alignement Module.
"""

import copy

import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import multiply
from tensorflow.keras import losses
import tensorflow.keras.backend as K

from adapt.utils import check_indexes, check_estimator, check_network


class CORAL:
    """
    CORAL: CORrelation ALignment
    
    CORAL is a feature based domain adaptation method which minimizes
    domain shift by aligning the second-order statistics of source and
    target distributions.
    
    The method transforms source features in order to minimize the
    Frobenius norm between the correlation matrix of the input target
    data and the one of the transformed input source data.
    
    The source features transformation is described by the following
    optimization problem:
    
    .. math::
        
        \min_{A}{||A^T C_S A - C_T||_F^2}
        
    Where:
    
    - :math:`C_S` is the correlation matrix of input source data
    - :math:`C_T` is the correlation matrix of input target data
    
    The solution of this OP can be written with an explicit formula
    and the features transformation can be computed through this
    four steps algorithm:
    
    - :math:`C_S = Cov(X_S) + \\lambda I_p`
    - :math:`C_S = Cov(X_T) + \\lambda I_p`
    - :math:`X_S = X_S C_S^{-\\frac{1}{2}}`
    - :math:`X_S = X_S C_T^{\\frac{1}{2}}`
    
    Where :math:`\\lambda` is a regularization parameter.
    
    Notice that CORAL only uses labeled source and unlabeled target data.
    It belongs then to "unsupervised" domain adaptation methods.
    However, labeled target data can be added to the training process
    straightforwardly.
    
    Parameters
    ----------
    get_estimator : callable or object, optional (default=None)
        Constructor for the estimator.
        If a callable function is given it should return an estimator
        object (with ``fit`` and ``predict`` methods).
        If a class is given, a new instance of this class will
        be built and used as estimator.
        If get_estimator is ``None``, a ``LogisticRegression`` object will be
        used by default as estimator.

    lambdap : float, optional (default=1.0)
        Regularization parameter.
    
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.

    Attributes
    ----------
    estimator_ : object
        Fitted estimator.
        
    Cs_ : numpy array
        Correlation matrix of source features.
        
    Ct_ : numpy array
        Correlation matrix of target features.
        
    See also
    --------
    DeepCORAL

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1511.05547.pdf>`_ Sun B., Feng J., Saenko K. \
"Return of frustratingly easy domain adaptation". In AAAI, 2016.
    """
    def __init__(self, get_estimator=None, lambdap=1.0, **kwargs):
        self.get_estimator = get_estimator
        self.lambdap = lambdap
        self.kwargs = kwargs

        if self.get_estimator is None:
            self.get_estimator = LogisticRegression


    def fit(self, X, y, src_index, tgt_index, tgt_index_labeled=None,
            sample_weight=None, **fit_params):
        """
        Perfrom correlation alignement on input source data to match 
        input target data (given by ``tgt_index``).
        Then fit estimator on the aligned source data and the labeled
        target ones (given by ``tgt_index_labeled``).

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
            
        tgt_index_labeled : iterable, optional (default=None)
            indexes of target labeled data in X, y.

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.

        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)

        Xs = X[src_index]
        ys = y[src_index]
        Xt = X[tgt_index]
        yt = y[tgt_index]

        self.estimator_ = check_estimator(self.get_estimator, **self.kwargs)

        self.Cs_ = np.cov(Xs, rowvar=False) + self.lambdap * np.eye(Xs.shape[1])
        self.Ct_ = np.cov(Xt, rowvar=False) + self.lambdap * np.eye(Xt.shape[1])

        Xs = np.matmul(Xs, linalg.inv(linalg.sqrtm(self.Cs_)))
        Xs = np.matmul(Xs, linalg.sqrtm(self.Ct_))
        
        if tgt_index_labeled is None:
            X = Xs
            y = ys
        else:
            X = np.concatenate((Xs, X[tgt_index_labeled]))
            y = np.concatenate((ys, y[tgt_index_labeled]))
        
        if sample_weight is None:
            self.estimator_.fit(X, y, **fit_params)
        else:
            if tgt_index_labeled is None:
                sample_weight = sample_weight[src_index]
            else:
                sample_weight = np.concatenate((
                    sample_weight[src_index],
                    sample_weight[tgt_index_labeled]
                ))
            self.estimator_.fit(X, y, sample_weight=sample_weight,
                                **fit_params)

        return self


    def predict(self, X):
        """
        Return the predictions of the estimator.

        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Prediction of ``estimator_``.
        """
        return self.estimator_.predict(X)


    def get_features(self, X):
        """
        Return aligned features for X.
        
        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        Xp : array
            Projection of X in encoded feature space.
        """
        Xp = np.matmul(X, linalg.inv(linalg.sqrtm(self.Cs_)))
        Xp = np.matmul(Xp, linalg.sqrtm(self.Ct_))
        return Xp


class DeepCORAL:
    """
    DeepCORAL: Deep CORrelation ALignment
    
    DeepCORAL is an extension of CORAL method. It learns a nonlinear
    transformation which aligns correlations of layer activations in
    deep neural networks.
    
    The method consist in training both an **encoder** and a **task**
    network. The **encoder** network maps input features into new
    encoded ones on which the **task** network is trained.
    
    The parameters of the two networks are optimized in order to
    minimize the following loss function:
    
    .. math::
    
        \mathcal{L} = \mathcal{L}_{task} + \\lambda ||C_S - C_T||_F^2
        
    Where:
    
    - :math:`\mathcal{L}_{task}` is the task loss computed with
      source and labeled target data.
    - :math:`C_S` is the correlation matrix of source data in the
      encoded feature space.
    - :math:`C_T` is the correlation matrix of target data in the
      encoded feature space.
    - :math:`||.||_F` is the Frobenius norm.
    - :math:`\\lambda` is a trade-off parameter.
    
    Thus the **encoder** network learn a new feature representation
    on wich the correlation matrixes of source and target data are
    "close" and where a **task** network is able to learn the task
    with source labeled data.
    
    Notice that DeepCORAL only uses labeled source and unlabeled target
    data. It belongs then to "unsupervised" domain adaptation methods.
    However, labeled target data can be added to the training process
    straightforwardly.
    
    Parameters
    ----------
    get_encoder: callable, optional (default=None)
        Constructor for encoder network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network and an ``output_shape``
        argument giving the shape of the last layer.
        If ``None``, a shallow network with 10 neurons is used
        as encoder network.
        
    get_task: callable, optional (default=None)
        Constructor for task network.
        The constructor should return a tensorflow compiled Model. 
        It should also take at least an ``input_shape`` argument
        giving the input shape of the network.
        If ``None``, a linear network is used as task network.
        
    lambdap : float, optional (default=1.0)
        Trade-Off parameter.
               
    enc_params: dict, optional (default={})
        Additional arguments for ``get_encoder``
        
    task_params: dict, optional (default={})
        Additional arguments for ``get_task``
        
    compil_params: key, value arguments, optional
        Additional arguments for network compiler
        (loss, optimizer...).
        If none, loss is set to ``"mean_squared_error"``
        and optimizer to ``"adam"``.

    Attributes
    ----------
    encoder_ : tensorflow Model
        Fitted encoder network.
        
    task_ : tensorflow Model
        Fitted task network.
    
    model_ : tensorflow Model
        Fitted model: the union of
        encoder and task networks.
        
    See also
    --------
    CORAL

    References
    ----------
    .. [1] `[1] <https://arxiv.org/pdf/1607.01719.pdf>`_ Sun B. and Saenko K. \
"Deep CORAL: correlation alignment for deep domain adaptation." In ICCV, 2016.
    """
    def __init__(self, get_encoder=None, get_task=None, lambdap=1.0,
                 enc_params={}, task_params={}, **compil_params):
        self.get_encoder = get_encoder
        self.get_task = get_task
        self.lambdap = lambdap
        self.enc_params = enc_params
        self.task_params = task_params
        self.compil_params = compil_params


    def fit(self, X, y, src_index, tgt_index, tgt_index_labeled=None,
            sample_weight=None, **fit_params):
        """
        Fit encoder and task networks. 
        
        Source data and unlabeled target data are used for the correlation
        alignment in the encoded space.
        
        Source data and labeled target data are used to learn the task.

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
            
        tgt_index_labeled : iterable, optional (default=None)
            indexes of target labeled data in X, y.

        sample_weight : numpy array, optional (default=None)
            Individual weights for each sample.

        fit_params : key, value arguments
            Arguments given to the fit method of the estimator
            (epochs, batch_size...).

        Returns
        -------
        self : returns an instance of self
        """
        check_indexes(src_index, tgt_index, tgt_index_labeled)
        
        self._create_model(X.shape[1:], y.shape[1:])
        
        if tgt_index_labeled is None:
            task_index = src_index
        else:
            task_index = np.concatenate((src_index, tgt_index_labeled))
        
        max_size = max((len(src_index), len(tgt_index), len(task_index)))
        resized_src_ind = np.resize(src_index, max_size)
        resized_tgt_ind = np.resize(tgt_index, max_size)
        resized_task_ind = np.resize(task_index, max_size)
        
        self.model_.fit([X[resized_src_ind], X[resized_tgt_ind],
                         X[resized_task_ind], y[resized_src_ind],
                         np.ones(max_size)],
                        **fit_params)
        return self
    
    
    def predict(self, X):
        """
        Return the prediction of task network
        on the encoded features.
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        y_pred: array
            prediction of task network
        """
        return self.task_.predict(self.encoder_.predict(X))


    def _create_model(self, shape_X, shape_y):
        
        self.encoder_ = self.get_encoder(
                        input_shape=shape_X,
                        **self.enc_params)
        self.task_ = self.get_task(
                        input_shape=self.encoder_.output_shape[1:],
                        output_shape=shape_y,
                        **self.task_params)
        
        input_src = Input(shape_X)
        input_tgt = Input(shape_X)
        input_task = Input(shape_X)
        output_src = Input(shape_y)
        input_ones = Input((1,))
        
        encoded_src = self.encoder_(input_src)
        encoded_tgt = self.encoder_(input_tgt)
        encoded_task = self.encoder_(input_task)
        
        tasked = self.task_(encoded_task)
        
        compil_params = copy.deepcopy(self.compil_params)
        if "loss" in compil_params:
            task_loss = K.mean(
                self.compil_params["loss"](output_src, tasked)
            )
            compil_params.pop('loss')
        else:
            task_loss = K.mean(
                losses.mean_squared_error(output_src, tasked)
            )

        ones_dot_encoded_src = K.dot(
            K.transpose(input_ones), encoded_src
        )
        corr_src = (1 / (K.sum(input_ones) - 1)) * (
            K.dot(K.transpose(encoded_src), encoded_src) -
            (1 / K.sum(input_ones)) * 
            K.dot(K.transpose(ones_dot_encoded_src),
            ones_dot_encoded_src)
        )
        ones_dot_encoded_tgt = K.dot(
            K.transpose(input_ones), encoded_tgt
        )
        corr_tgt = (1 / (K.sum(input_ones) - 1)) * (
            K.dot(K.transpose(encoded_tgt), encoded_tgt) -
            (1 / K.sum(input_ones)) * 
            K.dot(K.transpose(ones_dot_encoded_tgt),
            ones_dot_encoded_tgt)
        )
        
        corr_loss = (1./4.) * K.mean(K.square(corr_src - corr_tgt))
        
        loss = task_loss + self.lambdap * corr_loss
        
        self.model_ = Model([input_src, input_tgt,
                             input_task, output_src, input_ones],
                            [encoded_src, encoded_tgt, tasked],
                            name="DeepCORAL")
        self.model_.add_loss(loss)
        
        if not "optimizer" in compil_params:
            compil_params["optimizer"] = "adam"
        
        self.model_.compile(**compil_params)
        
        return self
