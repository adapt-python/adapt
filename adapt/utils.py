"""
Utility functions for adapt package.
"""


import warnings
import inspect

from tensorflow.keras import Model


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
        estimator = get_estimator(**kwargs)
    else:
        raise ValueError("get_estimator is neither a callable nor a class")

    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise ValueError("Built estimator does "
                         "not implement fit and predict methods")

    return estimator


def check_network(get_model, constructor_name="get_model",
                  shape_arg=False, **kwargs):
    """
    Build and check network.

    Check that ``get_model`` is a callable.
    Then, build an estimator and check that it is an 
    instance of tensorflow Model.
    
    If ``shape_arg`` is True, the function checks that
    the built estimator takes an ``input_shape`` arguments.

    Parameters
    ----------
    get_model : object
        Constructor for the estimator.

    constructor_name: str, optional (default="get_model")
        Name of contructor variable.
        
    shape_args: boolean, optional (default=False)
        If True, check that the estimator takes
        an ``input_shape`` argument.
    
    kwargs : key, value arguments, optional
        Additional arguments for the constructor.
    """
    if hasattr(get_model, "__call__"):
        if (shape_arg and not
            "input_shape" in inspect.getfullargspec(get_model)[0]):
            raise ValueError("Constructor %s must take "
                             "an 'input_shape' argument"%constructor_name)
        model = get_model(**kwargs)
    else:
        raise ValueError("%s is not a callable"%constructor_name)

    if not isinstance(model, Model):
        raise ValueError("Built model is not a tensorflow Model instance")

    return model
