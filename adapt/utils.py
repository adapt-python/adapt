"""
Utility functions for adapt package.
"""


import warnings
import inspect

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Reshape


def check_indexes(src_index, tgt_index, tgt_index_labeled=None):
    """
    Check indexes.

    Check that all given indexes are iterable. The function
    also raises warnings if similar indexes appear in both
    source and target index lists.

    Parameters
    ----------
    src_index : iterable
        Indexes of source instances.

    tgt_index : iterable
        Indexes of target instances.

    tgt_index_labeled : iterable, optional
        Indexes of labeled target instances.
    """
    list_index = [src_index, tgt_index]
    names = ["src_index", "tgt_index"]

    if tgt_index_labeled is not None:
        list_index.append(tgt_index_labeled)
        names.append("tgt_index_labeled")

    for index, name in zip(list_index, names):
        if not hasattr(index, "__iter__"):
            raise ValueError("%s is not an iterable."%name)

    if len(set(src_index) & set(tgt_index)) > 0:
        warnings.warn("Similar indexes appear in both"
                      " src_index and tgt_index")

    if tgt_index_labeled is not None:
        if len(set(src_index) & set(tgt_index_labeled)) > 0:
            warnings.warn("Similar indexes appear in both"
                          " src_index and tgt_index_labeled")


def check_estimator(get_estimator, **kwargs):
    """
    Build and check estimator.

    Check that ``get_estimator`` is a callable or is a class.
    Then, build an estimator and check that it
    implements ``fit`` and ``predict`` methods.

    Parameters
    ----------
    get_estimator : object
        Constructor for the estimator.

    kwargs : key, value arguments, optional
        Additional arguments for the constructor.
    """
    if (hasattr(get_estimator, "__call__")
        or inspect.isclass(get_estimator)):
        try:
            estimator = get_estimator(**kwargs)
        except Exception as err_info:
            raise ValueError("Failed to build estimator with"
                             " 'get_estimator'. "
                             "Please provide a builder function wich "
                             "returns a valid estimator object or "
                             "check the given additional arguments. \n \n"
                             "Exception message: %s"%err_info)
        
    else:
        raise ValueError("get_estimator is neither a callable nor a class")

    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise ValueError("Built estimator does "
                         "not implement fit and predict methods")

    return estimator


def check_network(get_model, constructor_name="get_model", **kwargs):
    """
    Build and check network.

    Check that ``get_model`` is a callable.
    Then, build an estimator and check that it is an 
    instance of tensorflow Model.
    
    If ``input_shape`` is in kwargs, the function checks that
    the built estimator takes an ``input_shape`` arguments.
    
    If ``output_shape`` is in kwargs, the function checks that
    the built estimator takes an ``output_shape`` arguments.

    Parameters
    ----------
    get_model : object
        Constructor for the estimator.

    constructor_name: str, optional (default="get_model")
        Name of contructor variable.

    kwargs : key, value arguments, optional
        Additional arguments for the constructor.
    """
    if hasattr(get_model, "__call__"):
        if ("input_shape" in kwargs and not
            "input_shape" in inspect.getfullargspec(get_model)[0]):
            raise ValueError("Constructor '%s' must take "
                             "an 'input_shape' argument"%constructor_name)
        if ("output_shape" in kwargs and not
            "output_shape" in inspect.getfullargspec(get_model)[0]):
            raise ValueError("Constructor '%s' must take "
                             "an 'output_shape' argument"%constructor_name)
        try:
            model = get_model(**kwargs)
        except Exception as err_info:
            raise ValueError("Failed to build model with constructor '%s'. "
                             "Please provide a builder function wich "
                             "returns a compiled tensorflow Model or "
                             "check the given additional arguments. \n \n"
                             "Exception message: %s"%(constructor_name,
                             err_info))
    else:
        raise ValueError("'%s' is not a callable"%constructor_name)

    if not isinstance(model, Model):
        raise ValueError("Built model from '%s' is not "
                         "a tensorflow Model instance"%constructor_name)
    return model


def get_default_encoder(input_shape):
    """
    Return a compiled tensorflow Model of a shallow network
    with 10 neurons and a linear activation.

    Parameters
    ----------
    input_shape: tuple
        Network input_shape

    Returns
    -------
    tensorflow Model
    """
    inputs = Input(shape=input_shape)
    flattened = Flatten(inputs)
    outputs = Dense(10)(flattened)
    model = Model(inputs, outputs)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def get_default_task(input_shape, output_shape=(1,)):
    """
    Return a compiled tensorflow Model of a linear network
    with a linear activation.

    Parameters
    ----------
    input_shape: tuple
        Network input_shape

    output_shape: tuple, optional (default=(1,))
        Network output_shape

    Returns
    -------
    tensorflow Model
    """
    inputs = Input(shape=input_shape)
    flattened = Flatten(inputs)
    outputs = Dense(np.prod(output_shape))(flattened)
    outputs = Reshape(output_shape)(outputs)
    model = Model(inputs, outputs)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


@tf.custom_gradient
def _grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradientReversal(Layer):
    """
    Inverse sign of gradient during backpropagation.

    Act as identity in forward step.
    """
    def __init__(self):
        super().__init__()

    def call(self, x):
        """
        Call gradient reversal.
        
        Parameters
        ----------
        x: object
            Inputs
            
        Returns
        -------
        func: gradient reversal function.
        """
        return _grad_reverse(x)