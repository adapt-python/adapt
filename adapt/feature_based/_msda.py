"""
Marginalized Stacked Denoising Autoencoder
"""

import copy
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, GaussianNoise, Flatten, Reshape, Dense
from tensorflow.keras.optimizers import Adam

from adapt.utils import (check_arrays,
                   check_one_array,
                         check_network,
                         check_estimator)

def _get_default_encoder():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation=None))
    return model


def _get_default_decoder(output_shape):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(np.prod(output_shape), activation=None))
    model.add(Reshape(output_shape))
    return model


class mSDA:
    """
    mSDA: marginalized Stacked Denoising Autoencoder.
    
    mSDA is a feature-based domain adaptation method.
    
    The method use denoising **autoencoders** to learn a new robust
    representation of input data.
    
    mSDA first trains an **autoencoder** (composed of an **encoder**
    and a **decoder** networks) to reconstruct a noisy dataset made
    of target and source input data.
    
    The method is based on the assumption that the fitted **encoder**
    will then provide more robust features to domain shifts.
    
    In a second time, an **estimator** is trained on the encoded feature
    space using labeled source data and labeled target data if provided.
    Thus the algorithm can be used in both **unsupervised** and
    **supervised** DA settings.

    Parameters
    ----------    
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a neural network with two
        hidden layers of 100 neurons with ReLU activation each
        is used. The encoded space is made of one layer of
        10 neurons with linear activation.
        
    decoder : tensorflow Model (default=None)
        Decoder netwok. If ``None``, a neural network with two
        hidden layers of 100 neurons with ReLU activation each
        is used. The output layer is made of ``Xs.shape[1]`` neurons 
        and a linear activation.
    
    estimator : sklearn estimator or tensorflow Model (default=None)
        Estimator used to learn the task. 
        If estimator is ``None``, a ``LinearRegression``
        instance is used as estimator.
        
    noise_lvl : float (default=0.1)
        Standard deviation of gaussian noise added to the input data
        in the denoising autoencoder.
        
    copy : boolean (default=True)
        Whether to make a copy of ``encoder``, ``decoder``
        and ``estimator`` or not.
            
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.
    
    compil_params : key, value arguments, optional
        Additional arguments for autoencoder compiler
        (loss, optimizer...).
        If none, loss is set to ``"mean_squared_error"``
        and optimizer to ``Adam(0.001)``.

    Attributes
    ----------
    encoder_ : tensorflow Model
        Encoder network.
        
    decoder_ : tensorflow Model
        Decoder network.
        
    autoencoder_ : tensorflow Model
        Autoencoder network.
        
    estimator_ : object
        Estimator.
        
    history_ : dict
        history of the losses and metrics across the epochs
        of the autoencoder training.
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.feature_based import mSDA
    >>> from adapt.utils import make_classification_da
    >>> from sklearn.linear_model import LogisticRegression
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = mSDA(estimator=LogisticRegression('none'), random_state=0)
    >>> model.fit(Xs, ys, Xt, epochs=500, verbose=0)
    >>> (model.predict(Xt) == yt).sum() / len(yt)
    0.68
    >>> lr = LogisticRegression('none')
    >>> lr.fit(Xs, ys)
    >>> lr.score(Xt, yt)
    0.58
    
    See also
    --------
    DANN
    DeepCORAL

    References
    ----------
    .. [1] `[1] <https://arxiv.org/ftp/arxiv/papers/1206/1206.4683.pdf>`_ \
M. Chen, Z. E. Xu, K. Q. Weinberger, and F. Sha. \
"Marginalized denoising autoencoders for domain adaptation". In ICML, 2012.
    """

    def __init__(self, 
                 encoder=None, 
                 decoder=None,
                 estimator=None,
                 noise_lvl=0.1,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **compil_params):
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        if encoder is None:
            self.encoder_ = _get_default_encoder()
        else:
            self.encoder_ = check_network(encoder, copy=copy,
                                          display_name="encoder",
                                          compile_=False)
        if decoder is None:
            self.no_decoder_ = True
        else:
            self.no_decoder_ = False
            self.decoder_ = check_network(decoder, copy=copy,
                                          display_name="decoder",
                                          compile_=False)
        
        self.estimator_ = check_estimator(estimator, copy=copy)
        self.noise_lvl = noise_lvl
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state
        self.compil_params = compil_params


    def fit(self, Xs, ys, Xt, fit_params_est=None, **fit_params):
        """
        Fit mSDA.

        Parameters
        ----------
        Xs : numpy array
            Source input data.

        ys : numpy array
            Source output data.

        Xt : numpy array
            Target input data.
            
        fit_params_est : key, value arguments
            Arguments given to the fit method of
            ``estimator``.

        fit_params : key, value arguments
            Arguments given to the fit method of
            ``auto_encoder``.

        Returns
        -------
        self : returns an instance of self
        """        
        Xs, ys, Xt, _ = check_arrays(Xs, ys, Xt, None)
        
        if fit_params_est is None:
            fit_params_est = {}
            
        if self.verbose:
            print("Fit autoencoder...")
        self.fit_embeddings(Xs, Xt, **fit_params)
        
        Xs_emb = self.encoder_.predict(Xs)
        
        if self.verbose:
            print("Fit estimator...")
        self.fit_estimator(Xs_emb, ys, **fit_params_est)
        return self
        
    
    def fit_embeddings(self, Xs, Xt, **fit_params):
        """
        Fit embeddings.
        
        Parameters
        ----------
        Xs : array
            Input source data.
            
        Xt : array
            Input target data.
            
        fit_params : key, value arguments
            Arguments given to the fit method of
            ``auto_encoder``.
            
        Returns
        -------
        Xs_emb, Xt_emb : embedded source and target data
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        if np.any(Xs.shape[1:] != Xt.shape[1:]):
            raise ValueError("Xs and Xt should have same dim, got "
                             "%s and %s"%(str(Xs.shape[1:]), str(Xt.shape[1:])))
        shape = Xs.shape[1:]
        
        if self.no_decoder_:
            self.decoder_ = _get_default_decoder(shape)
            self.no_decoder_ = False
        
        if not hasattr(self, "autoencoder_"):
            self._build(shape)
        
        X = np.concatenate((Xs, Xt))
        hist = self.autoencoder_.fit(X, X, **fit_params)
        
        for k, v in hist.history.items():
            self.history_[k] = self.history_.get(k, []) + v
        
        return self
    
    
    def fit_estimator(self, X, y, **fit_params):
        """
        Fit estimator.
        
        Parameters
        ----------
        X : array
            Input data.
            
        y : array
            Output data.
            
        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator.
            
        Returns
        -------
        estimator_ : fitted estimator
        """
        X = check_one_array(X)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.estimator_.fit(X, y, **fit_params)
        return self.estimator_
        
        
    def _build(self, shape):
        self.history_ = {}
        
        zeros_enc = self.encoder_.predict(np.zeros((1,) + shape))
        self.decoder_.predict(zeros_enc)
        
        inputs = Input(shape)
        noised = GaussianNoise(self.noise_lvl)(inputs)
        encoded = self.encoder_(noised)
        decoded = self.decoder_(encoded)
        self.autoencoder_ = Model(inputs, decoded)
        
        compil_params = copy.deepcopy(self.compil_params)
        if not "loss" in compil_params:
            compil_params["loss"] = "mean_squared_error"        
        if not "optimizer" in compil_params:
            compil_params["optimizer"] = Adam(0.001)
        
        self.autoencoder_.compile(**compil_params)
        return self


    def predict(self, X):
        """
        Return the predictions of the estimator on the encoded
        feature space.

        Parameters
        ----------
        X : array
            Input data.

        Returns
        -------
        y_pred : array
            Prediction of ``estimator_``.
        """
        X = check_one_array(X)
        return self.estimator_.predict(self.predict_features(X))
    
    
    def predict_features(self, X):
        """
        Return the encoded features of X.
        
        Parameters
        ----------
        X: array
            input data
            
        Returns
        -------
        X_enc: array
            predictions of encoder network
        """
        X = check_one_array(X)
        return self.encoder_.predict(X)
