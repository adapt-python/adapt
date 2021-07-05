"""
Utility functions for adapt package.
"""

import warnings
import inspect
from copy import deepcopy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Input
from tensorflow.keras.models import clone_model


def check_one_array(X):
    """
    Check array and reshape 1D array in 2D array
    of shape (-1, 1).

    Parameters
    ----------
    X : numpy array
        Input data.
        
    Returns
    -------
    X
    """
    X = check_array(X, ensure_2d=False, allow_nd=True)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

            
def check_arrays(Xs, ys, Xt, yt=None):
    """
    Check arrays and reshape 1D array in 2D array
    of shape (-1, 1). Check if the length of Xs, Xt
    match the length of ys, yt.

    Parameters
    ----------
    Xs : numpy array
        Source input data.

    ys : numpy array
        Source output data.

    Xt : numpy array
        Target input data.
            
    yt : numpy array, optional (default=None)
        Target output data. `yt` is only used
        for validation metrics.
        
    Returns
    -------
    Xs, ys, Xt, yt
    """
    Xs = check_array(Xs, ensure_2d=False, allow_nd=True)
    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)
    Xt = check_array(Xt, ensure_2d=False, allow_nd=True)
    if Xt.ndim == 1:
        Xt = Xt.reshape(-1, 1)
    ys = check_array(ys, ensure_2d=False, allow_nd=True)
    if ys.ndim == 1:
        ys = ys.reshape(-1, 1)
    
    if len(Xs) != len(ys):
        raise ValueError("Length of Xs and ys mismatch: %i != %i"%
                         (len(Xs), len(ys)))
    
    if yt is not None:
        yt = check_array(yt, ensure_2d=False, allow_nd=True)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)
        if len(Xt) != len(yt):
            raise ValueError("Length of Xt and yt mismatch: %i != %i"%
                             (len(Xt), len(yt)))
    return Xs, ys, Xt, yt


def check_estimator(estimator=None, copy=True,
                    display_name="estimator",
                    task=None,
                    force_copy=False):
    """
    Check estimator.

    Check that ``estimator`` is a sklearn ``BaseEstimator``
    or a tensorflow ``Model``.

    Parameters
    ----------
    estimator : sklearn BaseEstimator or tensorflow Model
        Estimator. If ``None`` a LinearRegression instance
        or a LogisticRegression instance is returned
        depending on the ``task`` argument.

    copy : boolean (default=False)
        Whether to return a copy of the estimator or not.
        If cloning fail, a warning is raised.

    display_name: str (default="estimator")
        Name to display if an error or warning is raised
        
    task : str (default=None)
        Task at hand. Possible value : 
        (``None``, ``"reg"``, ``"class"``)
        
    force_copy : boolean (default=False)
        If True, an error is raised if the cloning failed.
    """
    if estimator is None:
        if task == "class":
            estimator = LogisticRegression()
        else:
            estimator = LinearRegression()
    
    if isinstance(estimator, (BaseEstimator, KerasClassifier, KerasRegressor)):
        if (isinstance(estimator, ClassifierMixin) and task=="reg"):
            raise ValueError("`%s` argument is a sklearn `ClassifierMixin` instance "
                             "whereas the considered object handles only regression task. "
                             "Please provide a sklearn `RegressionMixin` instance or a "
                             "tensorflow Model instance."%display_name)
        if (isinstance(estimator, RegressorMixin) and task=="class"):
            raise ValueError("`%s` argument is a sklearn `RegressionMixin` instance "
                             "whereas the considered object handles only classification task. "
                             "Please provide a sklearn `ClassifierMixin` instance or a "
                             "tensorflow Model instance."%display_name)
        if copy:
            try:
                new_estimator = deepcopy(estimator)
            except Exception as e:
                if force_copy:
                    raise ValueError("`%s` argument can't be duplicated. "
                                     "Recorded exception: %s. "%
                                     (display_name, e))
                else:
                    warnings.warn("`%s` argument can't be duplicated. "
                                  "Recorded exception: %s. "
                                  "The current estimator will be used. "
                                  "Use `copy=False` to hide this warning."%
                                  (display_name, e))
                    new_estimator = estimator
        else:
            new_estimator = estimator
    elif isinstance(estimator, Model):
        new_estimator = check_network(network=estimator,
                                  copy=copy, 
                                  display_name=display_name,
                                  force_copy=force_copy,
                                  compile_=True)
    else:
        raise ValueError("`%s` argument is neither a sklearn `BaseEstimator` "
                         "instance nor a tensorflow Model instance. "
                         "Given argument, %s"%
                         (display_name, str(estimator)))
    return new_estimator


def check_network(network, copy=True,
                  compile_=False,
                  display_name="network",
                  force_copy=False):
    """
    Check if the given network is a tensorflow Model.
    If ``copy`` is ``True``, a copy of the network is
    returned if possible.

    Parameters
    ----------
    network : tensorflow Model
        Network to check.
        
    copy : boolean (default=True)
        Whether to return a copy of the network or not.
        If cloning fail, a warning is raised.
        
    compile_ : boolean (default=False)
        Whether to compile the network after cloning,
        using copy of the network loss and optimizer.

    display_name : str (default="network")
        Name to display if an error or warning is raised
        
    force_copy : boolean (default=False)
        If True, an error is raised if the cloning failed.
    """
    if not isinstance(network, Model):
        raise ValueError('Expected `%s` argument '
                         'to be a `Model` instance, got: %s'%
                         (display_name, str(network)))
    
    if copy:
        try:
            if hasattr(network, "input_shape"):
                shape = network.input_shape[1:]
                new_network = clone_model(network, input_tensors=Input(shape))
                new_network.set_weights(network.get_weights())
            else:
                new_network = clone_model(network)
        except Exception as e:
            if force_copy:
                raise ValueError("`%s` argument can't be duplicated. "
                                 "Recorded exception: %s. "%
                                 (display_name, e))
            else:
                warnings.warn("`%s` argument can't be duplicated. "
                              "Recorded exception: %s. "
                              "The current network will be used. "
                              "Use `copy=False` to hide this warning."%
                              (display_name, e))
                new_network = network
        if compile_:
            if network.optimizer:
                new_network.compile(optimizer=deepcopy(network.optimizer),
                                    loss=deepcopy(network.loss),
                                    metrics=deepcopy(network.metrics))
            else:
                raise ValueError("The given `%s` argument is not compiled yet. "
                                 "Please use `model.compile(optimizer, loss)`."%
                                 (display_name))
    else:        
        new_network = network
    return new_network


def get_default_encoder():
    """
    Return a tensorflow Model of one layer
    with 10 neurons and a relu activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    return model


def get_default_task():
    """
    Return a tensorflow Model of two hidden layers
    with 10 neurons each and relu activations. The
    last layer is composed of one neuron with linear
    activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation=None))
    return model


def get_default_discriminator():
    """
    Return a tensorflow Model of two hidden layers
    with 10 neurons each and relu activations. The
    last layer is composed of one neuron with sigmoid
    activation.

    Returns
    -------
    tensorflow Model
    """
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


@tf.custom_gradient
def _grad_handler(x, lambda_):
    y = tf.identity(x)
    def custom_grad(dy):
        return (lambda_ * dy, 0. * lambda_)
    return y, custom_grad

class GradientHandler(Layer):
    """
    Multiply gradients with a scalar during backpropagation.

    Act as identity in forward step.
    
    Parameters
    ----------
    lambda_init : float (default=1.)
        Scalar multiplier
    """
    def __init__(self, lambda_init=1., name="g_handler"):
        super().__init__(name=name)
        self.lambda_ = tf.Variable(lambda_init,
                                   trainable=False,
                                   dtype="float32")

    def call(self, x):
        """
        Call gradient handler.
        
        Parameters
        ----------
        x: object
            Inputs
            
        Returns
        -------
        x, custom gradient function
        """
        return _grad_handler(x, self.lambda_)


def make_classification_da(n_samples=100, 
                           n_features=2,
                           random_state=2):
    """
    Generate a classification dataset for DA.
    
    Parameters
    ----------
    n_samples : int, optional (default=100)
        Size of source and target samples.
  
    n_features : int, optional (default=2)
        Number of features.
    
    random_state: int, optional (default=0)
        Random state number.
        
    Returns
    -------
    Xs : numpy array
        Source input data

    ys : numpy array
        Source output data
        
    Xt : numpy array
        Target input data

    yt : numpy array
        Target output data
    """
    np.random.seed(random_state)
    Xs, ys = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_features,
                                 n_redundant=0, n_repeated=0,
                                 n_clusters_per_class=1, n_classes=2,
                                 shuffle=False)
    Xt, yt = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_features,
                                 n_redundant=0, n_repeated=0,
                                 n_clusters_per_class=1, n_classes=2,
                                 shuffle=False)
    yt[:int(n_samples/2)] = 1; yt[int(n_samples/2):] = 0
    Xt[:, 0] += 1; Xt[:, 1] += 0.5;

    for i in range(n_features):
        Xs[:, i] = (Xs[:, i]-np.min(Xs[:, i]))/np.max(Xs[:, i]-np.min(Xs[:, i]))
        Xt[:, i] = (Xt[:, i]-np.min(Xt[:, i]))/np.max(Xt[:, i]-np.min(Xt[:, i]))
    
    return Xs, ys, Xt, yt


def make_regression_da(n_samples=100,
                       n_features=1,
                       random_state=0):
    """
    Generate a regression dataset for DA.
    
    Parameters
    ----------
    n_samples : int (default=100)
        Size of source and target samples.
        
    n_features : int (default=1)
        Sample dimension.
    
    random_state: int (default=0)
        Random state number.
        
    Returns
    -------
    Xs : numpy array
        Source input data

    ys : numpy array
        Source output data
        
    Xt : numpy array
        Target input data

    yt : numpy array
        Target output data
    """
    np.random.seed(random_state)
    
    Xs = np.random.uniform(size=(n_samples, n_features)) * 4 - 2
    Xs = np.sort(Xs)
    Xt = np.random.uniform(size=(n_samples, n_features)) * 2.5 + 2
    ys = (Xs[:, 0] + 0.1 * Xs[:, 0] ** 5 +
          np.random.randn(n_samples) * 0.2 + 1)
    yt = (Xt[:, 0] + 0.1 * (Xt[:, 0] - 2) **4  +
          np.random.randn(n_samples) * 0.4 + 1)
    
    Xt = (Xt - Xs.mean(0)) / Xs.std(0)
    yt = (yt - ys.ravel().mean()) / (2 * ys.ravel().std())
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0))
    ys = (ys - ys.ravel().mean()) / (2 * ys.ravel().std())
    
    X = np.concatenate((Xs, Xt))
    
    Xs = ((Xs - X.mean(0)) / X.std(0)) / 3
    Xt = ((Xt - X.mean(0)) / X.std(0)) / 3

    return Xs, ys, Xt, yt