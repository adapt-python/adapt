"""
Utility functions for adapt package.
"""

import warnings
import inspect

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
        raise ValueError("Built estimator does not implement fit and predict methods")

    return estimator
