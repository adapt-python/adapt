"""
Base for adapt
"""

import warnings
import inspect
from copy import deepcopy

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.metrics.pairwise import KERNEL_PARAMS
from sklearn.exceptions import NotFittedError
from tensorflow.keras import Model
try:
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
except:
    from scikeras.wrappers import KerasClassifier, KerasRegressor
try:
    from tensorflow.keras.optimizers.legacy import RMSprop
except:
    from tensorflow.keras.optimizers import RMSprop


from adapt.utils import (check_estimator,
                         check_network,
                         check_arrays,
                         set_random_seed,
                         check_sample_weight,
                         accuracy,
                         get_default_encoder,
                         get_default_task,
                         get_default_discriminator)
from adapt.metrics import normalized_linear_discrepancy


base_doc_est = dict(
estimator="""
    estimator : sklearn estimator or tensorflow Model (default=None)
        Estimator used to learn the task. 
        If estimator is ``None``, a ``LinearRegression``
        instance is used as estimator.
""",
encoder="""
    encoder : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a shallow network with 10
        neurons and ReLU activation is used as encoder network.
""",
task="""       
    task : tensorflow Model (default=None)
        Task netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as task network.
""",
discriminator="""     
    discriminator : tensorflow Model (default=None)
        Discriminator netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as discriminator
        network. Note that the output shape of the discriminator should
        be ``(None, 1)`` and a ``sigmoid`` activation should be used.    
""",
weighter="""
    weighter : tensorflow Model (default=None)
        Encoder netwok. If ``None``, a two layers network with 10
        neurons per layer and ReLU activation is used as
        weighter network.
"""
)

base_doc_Xt = """
    Xt : numpy array (default=None)
        Target input data.
"""

base_doc_Xt_yt = """
    Xt : numpy array (default=None)
        Target input data.
            
    yt : numpy array (default=None)
        Target output data.
"""

base_doc_2 ="""
    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
        
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.
    
    params : key, value arguments
        Arguments given at the different level of the adapt object.
        It can be, for instance, compile or fit parameters of the
        estimator or kernel parameters etc...
        Accepted parameters can be found by calling the method
        ``_get_legal_params(params)``.
"""

base_doc_other_params="""
    Yields
    ------
    optimizer : str or instance of tf.keras.optimizers (default="rmsprop")
        Optimizer for the task. It should be an
        instance of tf.keras.optimizers as:
        ``tf.keras.optimizers.SGD(0.001)`` or
        ``tf.keras.optimizers.Adam(lr=0.001, beta_1=0.5)``.
        A string can also be given as ``"adam"``.
        Default optimizer is ``rmsprop``.

    loss : str or instance of tf.keras.losses (default="mse")
        Loss for the task. It should be an
        instance of tf.keras.losses as:
        ``tf.keras.losses.MeanSquaredError()`` or
        ``tf.keras.losses.CategoricalCrossentropy()``.
        A string can also be given as ``"mse"`` or
        ``categorical_crossentropy``.
        Default loss is ``mse``.

    metrics : list of str or list of tf.keras.metrics.Metric instance
        List of metrics to be evaluated by the model during training
        and testing. Typically you will use ``metrics=['accuracy']``.

    optimizer_enc : str or instance of tf.keras.optimizers
        If the Adapt Model has an ``encoder`` attribute,
        a specific optimizer for the ``encoder`` network can
        be given. Typically, this parameter can be used to
        give a smaller learning rate to the encoder.
        If not specified, ``optimizer_enc=optimizer``.

    optimizer_disc : str or instance of tf.keras.optimizers
        If the Adapt Model has a ``discriminator`` attribute,
        a specific optimizer for the ``discriminator`` network can
        be given. If not specified, ``optimizer_disc=optimizer``.

    kwargs : key, value arguments
        Any arguments of the ``fit`` method from the Tensorflow
        Model can be given, as ``epochs`` and ``batch_size``.
        Specific arguments from ``optimizer`` can also be given
        as ``learning_rate`` or ``beta_1`` for ``Adam``.
        This allows to perform ``GridSearchCV`` from scikit-learn
        on these arguments.
"""


def make_insert_doc(estimators=["estimator"], supervised=False):
    """
    Abstract for adding common parameters
    to the docstring
    
    Parameters
    ----------
    estimators : list (default=['estimator'])
        list of estimators docstring to add.
        
    Returns
    -------
    func
    """
    def insert_base_doc(func):
        # Change signature of Deep Model
        if "BaseAdaptDeep" in func.__bases__[0].__name__:
            sign = inspect.signature(func.__init__)
            parameters = dict(sign.parameters)
            parameters.pop("self", None)
            sign = sign.replace(parameters=list(parameters.values()))
            func.__signature__ = sign
        
        doc = func.__doc__
        if "Parameters" in doc:
            splits = doc.split("Parameters")
            n_count = 0
            i = 0
            while (i<len(splits[1])) and (n_count<2):
                char = splits[1][i]
                if char == "\n":
                    n_count+=1
                i+=1

            j = i
            word = ""
            while (j<len(splits[1])) and (word != "---"):
                j+=1
                word = splits[1][j:j+3]
                if word == "---":
                    n_count = 0
                    while (j<len(splits[1])) and (n_count<2):
                        char = splits[1][j]
                        if char == "\n":
                            n_count+=1
                        j-=1
            doc_est = ""
            for est in estimators:
                doc_est += base_doc_est[est]
                
            if supervised:
                doc_1 = base_doc_Xt_yt
            else:
                doc_1 = base_doc_Xt
            
            doc_2 = base_doc_2
            if "BaseAdaptDeep" in func.__bases__[0].__name__:
                doc_2 += base_doc_other_params
            
            splits[1] = (
                splits[1][:i-1]+
                doc_est+doc_1+
                splits[1][i-1:j+1]+
                doc_2+
                splits[1][j+1:]
            )
            new_doc = splits[0]+"Parameters"+splits[1]
        else:
            new_doc = doc

        func.__doc__ = new_doc

        return func
    return insert_base_doc


class BaseAdapt:
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            Not used, here for scikit-learn compatibility.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        legal_params = self._get_legal_params(self.__dict__)
        params_names = set(self.__dict__) & set(legal_params)
        for key in params_names:
            value = getattr(self, key)
            out[key] = value
        return out


    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self
        
        self._check_params(params)
        
        for key, value in params.items():
            setattr(self, key, value)            
        return self


    def unsupervised_score(self, Xs, Xt):
        """
        Return unsupervised score.
        
        The normalized discrepancy distance is computed
        between the reweighted/transformed source input
        data and the target input data.
        
        Parameters
        ----------
        Xs : array
            Source input data.
            
        Xt : array
            Source input data.
            
        Returns
        -------
        score : float
            Unsupervised score.
        """
        Xs = check_array(Xs, accept_sparse=True)
        Xt = check_array(Xt, accept_sparse=True)
        
        if hasattr(self, "transform"):
            args = [
                p.name
                for p in inspect.signature(self.transform).parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            if "domain" in args:
                Xt = self.transform(Xt, domain="tgt")
                Xs = self.transform(Xs, domain="src")
            else:
                Xt = self.transform(Xt)
                Xs = self.transform(Xs)
        elif hasattr(self, "predict_weights"):
            sample_weight = self.predict_weights()
            sample_weight = sample_weight
            
            sample_weight = check_sample_weight(sample_weight, Xs)
            sample_weight /= sample_weight.sum()
            
            set_random_seed(self.random_state)
            bootstrap_index = np.random.choice(
            Xs.shape[0], size=Xs.shape[0], replace=True, p=sample_weight)
            Xs = Xs[bootstrap_index]
        else:
            raise ValueError("The Adapt model should implement"
                             " a transform or predict_weights methods")
        return normalized_linear_discrepancy(Xs, Xt)
    
    
    def _check_params(self, params):
        legal_params = self._get_legal_params(params)
        for key in params:
            if not key in legal_params:
                raise ValueError("%s is not a legal params for %s model. "
                                 "Legal params are: %s"%
                                 (key, self.__class__.__name__, str(legal_params)))
    
    
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(self.__init__, "deprecated_original", self.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % ("dummy", init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    
    def _filter_params(self, func, override={}, prefix=""):
        kwargs = {}
        args = [
            p.name
            for p in inspect.signature(func).parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for key, value in self.__dict__.items():
            new_key = key.replace(prefix+"__", "")
            if new_key in args and prefix in key:
                kwargs[new_key] = value
        kwargs.update(override)
        return kwargs
    
    
    def _get_target_data(self, X, y):
        if X is None:
            X = self.Xt
        if y is None:
            y = self.yt
            
        if X is None:
            raise ValueError("Argument `Xt` should be given in `fit`"
                             " when `self.Xt` is None.")
        return X, y
    
    
    def _check_domains(self, domains):
        domains = check_array(domains, ensure_2d=False)
        if len(domains.shape) > 1:
            raise ValueError("`domains` should be 1D array")
        self._domains_dict = {}
        new_domains = np.zeros(len(domains))
        unique = np.unique(domains)
        for dom, i in zip(unique, range(len(unique))):
            new_domains[domains==dom] = i
            self._domains_dict[i] = dom
        return new_domains



class BaseAdaptEstimator(BaseAdapt, BaseEstimator):
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
        
        set_random_seed(random_state)
        
        self.estimator = estimator
        self.Xt = Xt
        self.yt = yt
        self.verbose = verbose
        self.random_state = random_state
        self.copy = copy
        self._check_params(params)
        
        for key, value in params.items():
            setattr(self, key, value)
    
    
    def fit(self, X, y, Xt=None, yt=None, domains=None, **fit_params):
        """
        Fit Adapt Model.
        
        For feature-based models, the transformation of the
        input features ``Xs`` and ``Xt`` is first fitted. In a second
        stage, the ``estimator_`` is fitted on the transformed features.
        
        For instance-based models, source importance weights are
        first learned based on ``Xs, ys`` and ``Xt``. In a second
        stage, the ``estimator_`` is fitted on ``Xs, ys`` with the learned
        importance weights.

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
            Target input data. Only needed for supervised
            and semi-supervised Adapt model.
            If None, the `yt` argument given in `init` is used.
            
        domains : array (default=None)
            Vector giving the domain for each source
            data. Can be used for multisource purpose.

        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator.

        Returns
        -------
        self : returns an instance of self
        """
        Xt, yt = self._get_target_data(Xt, yt)
        X, y = check_arrays(X, y)
        self.n_features_in_ = X.shape[1]
        if yt is not None:
            Xt, yt = check_arrays(Xt, yt)
        else:
            Xt = check_array(Xt, ensure_2d=True, allow_nd=True)
        set_random_seed(self.random_state)
        
        self.n_features_in_ = X.shape[1]
 
        if hasattr(self, "fit_weights"):
            if self.verbose:
                print("Fit weights...")
            out = self.fit_weights(Xs=X, Xt=Xt,
                                   ys=y, yt=yt,
                                   domains=domains)
            if isinstance(out, tuple):
                self.weights_ = out[0]
                X = out[1]
                y = out[2]
            else:
                self.weights_ = out
            if "sample_weight" in fit_params:
                fit_params["sample_weight"] *= self.weights_
            else:
                fit_params["sample_weight"] = self.weights_
        elif hasattr(self, "fit_transform"):
            if self.verbose:
                print("Fit transform...")
            out = self.fit_transform(Xs=X, Xt=Xt,
                                     ys=y, yt=yt,
                                     domains=domains)
            if isinstance(out, tuple):
                X = out[0]
                y = out[1]
            else:
                X = out
        if self.verbose:
            print("Fit Estimator...")
        self.fit_estimator(X, y, **fit_params)
        return self


    def fit_estimator(self, X, y, sample_weight=None,
                      random_state=None, warm_start=True,
                      **fit_params):
        """
        Fit estimator on X, y.
        
        Parameters
        ----------
        X : array
            Input data.
            
        y : array
            Output data.
            
        sample_weight : array
            Importance weighting.
            
        random_state : int (default=None)
            Seed of the random generator
            
        warm_start : bool (default=True)
            If True, continue to fit ``estimator_``,
            else, a new estimator is fitted based on
            a copy of ``estimator``. (Be sure to set
            ``copy=True`` to use ``warm_start=False``)
            
        fit_params : key, value arguments
            Arguments given to the fit method of
            the estimator and to the compile method
            for tensorflow estimator.
            
        Returns
        -------
        estimator_ : fitted estimator
        """
        X, y = check_arrays(X, y, accept_sparse=True)
        set_random_seed(random_state)

        if (not warm_start) or (not hasattr(self, "estimator_")):
            estimator = self.estimator
            self.estimator_ = check_estimator(estimator,
                                              copy=self.copy,
                                              force_copy=True)
            if isinstance(self.estimator_, Model):
                compile_params = self._filter_params(self.estimator_.compile)
                if not "loss" in compile_params:
                    if estimator._is_compiled:
                        compile_params["loss"] = deepcopy(estimator.loss)
                    else:
                        raise ValueError("The given `estimator` argument"
                                         " is not compiled yet. "
                                         "Please give a compiled estimator or "
                                         "give a `loss` and `optimizer` arguments.")
                if not "optimizer" in compile_params:
                    if estimator._is_compiled:
                        compile_params["optimizer"] = deepcopy(estimator.optimizer)
                else:
                    if not isinstance(compile_params["optimizer"], str):
                        optim_params = self._filter_params(
                            compile_params["optimizer"].__init__)
                        if len(optim_params) > 0:
                            kwargs = compile_params["optimizer"].get_config()
                            kwargs.update(optim_params)
                            optimizer = compile_params["optimizer"].__class__(**kwargs)
                        else:
                            optimizer = compile_params["optimizer"]
                        compile_params["optimizer"] = optimizer
                self.estimator_.compile(**compile_params)

        fit_params = self._filter_params(self.estimator_.fit, fit_params)

        fit_args = [
            p.name
            for p in inspect.signature(self.estimator_.fit).parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        if "sample_weight" in fit_args:
            sample_weight = check_sample_weight(sample_weight, X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimator_.fit(X, y,
                                    sample_weight=sample_weight,
                                    **fit_params)
        else:
            if sample_weight is None:
                self.estimator_.fit(X, y, **fit_params)
            else:
                sample_weight = check_sample_weight(sample_weight, X)
                sample_weight /= sample_weight.sum()
                bootstrap_index = np.random.choice(
                len(X), size=len(X), replace=True,
                p=sample_weight)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.estimator_.fit(X[bootstrap_index],
                                        y[bootstrap_index],
                                        **fit_params)
        return self.estimator_


    def predict_estimator(self, X, **predict_params):
        """
        Return estimator predictions for X.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_pred : array
            prediction of estimator.
        """      
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        predict_params = self._filter_params(self.estimator_.predict,
                                            predict_params)
        return self.estimator_.predict(X, **predict_params)


    def predict(self, X, domain=None, **predict_params):
        """
        Return estimator predictions after
        adaptation.
        
        For feature-based method (object which implements
        a ``transform`` method), the input feature ``X``
        are first transformed. Then the ``predict`` method
        of the fitted estimator ``estimator_`` is applied
        on the transformed ``X``.
        
        Parameters
        ----------
        X : array
            input data
        
        domain : str (default=None)
            For antisymetric feature-based method,
            different transformation of the input X
            are applied for different domains. The domain
            should then be specified between "src" and "tgt".
            If ``None`` the default transformation is the
            target one.
        
        Returns
        -------
        y_pred : array
            prediction of the Adapt Model.
        """
        X = check_array(X, ensure_2d=True, allow_nd=True, accept_sparse=True)
        if hasattr(self, "transform"):
            if domain is None:
                domain = "tgt"
            args = [
                p.name
                for p in inspect.signature(self.transform).parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            if "domain" in args:
                X = self.transform(X, domain=domain)
            else:
                X = self.transform(X)
        return self.predict_estimator(X, **predict_params)


    def score(self, X, y, sample_weight=None, domain=None):
        """
        Return the estimator score.
        
        If the object has a ``transform`` method, the
        estimator is applied on the transformed
        features X. For antisymetric transformation,
        a parameter domain can be set to specified
        between source and target transformation.
        
        Call `score` on sklearn estimator and 
        `evaluate` on tensorflow Model.
        
        Parameters
        ----------
        X : array
            input data
            
        y : array
            output data
            
        sample_weight : array (default=None)
            Sample weights
             
        domain : str (default=None)
            This parameter specifies for antisymetric
            feature-based method which transformation
            will be applied between "source" and "target".
            If ``None`` the transformation by default is
            the target one.
            
        Returns
        -------
        score : float
            estimator score.
        """
        X, y = check_arrays(X, y, accept_sparse=True)
        
        if domain is None:
            domain = "target"
        
        if hasattr(self, "transform"):
            args = [
                p.name
                for p in inspect.signature(self.transform).parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            if "domain" in args:
                X = self.transform(X, domain=domain)
            else:
                X = self.transform(X)
        
        if hasattr(self.estimator_, "score"):
            score = self.estimator_.score(X, y, sample_weight)
        elif hasattr(self.estimator_, "evaluate"):
            if np.prod(X.shape) <= 10**8:
                score = self.estimator_.evaluate(
                    X, y,
                    sample_weight=sample_weight,
                    batch_size=len(X)
                )
            else:
                score = self.estimator_.evaluate(
                    X, y,
                    sample_weight=sample_weight
                )
            if isinstance(score, (tuple, list)):
                score = score[0]
        else:
            raise ValueError("Estimator does not implement"
                             " score or evaluate method")
        return score


    def _get_legal_params(self, params):
        # Warning: additional fit and compile parameters can be given in set_params
        # thus, we should also check estimator, optimizer in __dict__
        legal_params_fct = [self.__init__]
        if "estimator" in params:
            estimator = params["estimator"]
        else:
            estimator = self.estimator
        
        if hasattr(estimator, "fit"):
            legal_params_fct.append(estimator.fit)
        if hasattr(estimator, "predict"):
            legal_params_fct.append(estimator.predict)
        
        if isinstance(estimator, Model):
            legal_params_fct.append(estimator.compile)
            if "optimizer" in params:
                optimizer = params["optimizer"]
            elif hasattr(self, "optimizer"):
                optimizer = self.optimizer
            else:
                optimizer = None
            
            if (optimizer is not None) and (not isinstance(optimizer, str)):
                legal_params_fct.append(optimizer.__init__)
                
        legal_params = ["domain"]
        for func in legal_params_fct:
            args = [
                p.name
                for p in inspect.signature(func).parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            legal_params = legal_params + args
        
        # Add kernel params for kernel based algorithm
        if "kernel" in params:
            kernel = params["kernel"]
        elif hasattr(self, "kernel"):
            kernel = self.kernel
        else:
            kernel = None
        if kernel is not None:
            legal_params += list(KERNEL_PARAMS[kernel])
        legal_params = set(legal_params)
        legal_params.discard("self")
        return legal_params


    def __getstate__(self):
        dict_ = {k: v for k, v in self.__dict__.items()}
        if "estimator_" in dict_:
            if isinstance(dict_["estimator_"], Model):
                dict_["estimator_"] = self._get_config_keras_model(
                dict_["estimator_"]
                )
        if "estimators_" in dict_:
            for i in range(len(dict_["estimators_"])):
                if isinstance(dict_["estimators_"][i], Model):
                    dict_["estimators_"][i] = self._get_config_keras_model(
                    dict_["estimators_"][i]
                    )
        if "estimator" in dict_:
            if isinstance(dict_["estimator"], Model):
                dict_["estimator"] = self._get_config_keras_model(
                dict_["estimator"]
                )
        return dict_


    def __setstate__(self, dict_):
        if "estimator_" in dict_:
            if isinstance(dict_["estimator_"], dict):
                dict_["estimator_"] = self._from_config_keras_model(
                    dict_["estimator_"]
                )
        if "estimators_" in dict_:
            for i in range(len(dict_["estimators_"])):
                if isinstance(dict_["estimators_"][i], dict):
                    dict_["estimators_"][i] = self._from_config_keras_model(
                        dict_["estimators_"][i]
                    )
        if "estimator" in dict_:
            if isinstance(dict_["estimator"], dict):
                dict_["estimator"] = self._from_config_keras_model(
                    dict_["estimator"]
                )
        self.__dict__ = {k: v for k, v in dict_.items()}


    def _get_config_keras_model(self, model):
        if hasattr(model, "input_shape") or model.built:
            weights = model.get_weights()
        else:
            weights = None
        config = model.get_config()
        klass = model.__class__
        
        config = dict(weights=weights,
                     config=config,
                     klass=klass)
        
        if hasattr(model, "loss"):
            config["loss"] = model.loss
            
        if hasattr(model, "optimizer"):
            try:
                config["optimizer_klass"] = model.optimizer.__class__
                config["optimizer_config"] = model.optimizer.get_config()
            except:
                pass
        return config


    def _from_config_keras_model(self, dict_):
        weights = dict_["weights"]
        config = dict_["config"]
        klass = dict_["klass"]

        model = klass.from_config(config)

        if weights is not None:
            model.set_weights(weights)
        
        if "loss" in dict_ and "optimizer_klass" in dict_:
            loss = dict_["loss"]
            optimizer = dict_["optimizer_klass"].from_config(
                    dict_["optimizer_config"])
            try:
                model.compile(loss=loss, optimizer=optimizer)
            except:
                print("Unable to compile model")
        
        return model




class BaseAdaptDeep(Model, BaseAdapt):
    
    
    def __init__(self, 
                 encoder=None,
                 task=None,
                 discriminator=None,
                 Xt=None,
                 yt=None,
                 verbose=1,
                 copy=True,
                 random_state=None,
                 **params):
        
        super().__init__()
        
        self._self_setattr_tracking = False
        self.encoder = encoder
        self.task = task
        self.discriminator = discriminator
        self.Xt = Xt
        self.yt = yt
        self.verbose = verbose
        self.copy = copy
        self.random_state = random_state
        
        self._check_params(params)
        
        for key, value in params.items():
            if key == "metrics":
                key = "_adapt_metrics"
            setattr(self, key, value)
            
        self._self_setattr_tracking = True
    
    
    def fit(self, X, y=None, Xt=None, yt=None, domains=None, **fit_params):
        """
        Fit Model. Note that ``fit`` does not reset
        the model but extend the training.
        
        Notice also that the compile method will be called 
        if the model has not been compiled yet.

        Parameters
        ----------
        X : array or Tensor
            Source input data.

        y : array or Tensor (default=None)
            Source output data.
            
        Xt : array (default=None)
            Target input data. If None, the `Xt` argument
            given in `init` is used.

        yt : array (default=None)
            Target input data. Only needed for supervised
            and semi-supervised Adapt model.
            If None, the `yt` argument given in `init` is used.
            
        domains : array (default=None)
            Vector giving the domain for each source
            data. Can be used for multisource purpose.

        fit_params : key, value arguments
            Arguments given to the fit method of the model
            (epochs, batch_size, callbacks...).

        Returns
        -------
        self : returns an instance of self
        """
        set_random_seed(self.random_state)
            
        # 1. Get Fit params
        fit_params = self._filter_params(super().fit, fit_params)
        
        verbose = fit_params.get("verbose", 1)
        epochs = fit_params.get("epochs", 1)
        batch_size = fit_params.pop("batch_size", 32)
        shuffle = fit_params.pop("shuffle", True)
        buffer_size = fit_params.pop("buffer_size", None)
        validation_data = fit_params.pop("validation_data", None)
        validation_split = fit_params.pop("validation_split", 0.)
        validation_batch_size = fit_params.get("validation_batch_size", batch_size)
        
        # 2. Prepare datasets
        
        ### 2.1 Source
        if not isinstance(X, tf.data.Dataset):
            check_arrays(X, y)
            if len(y.shape) <= 1:
                y = y.reshape(-1, 1)
            
            # Single source
            if domains is None:
                self.n_sources_ = 1
                
                dataset_Xs = tf.data.Dataset.from_tensor_slices(X)
                dataset_ys = tf.data.Dataset.from_tensor_slices(y)
            
            # Multisource
            else:
                domains = self._check_domains(domains)
                self.n_sources_ = int(np.max(domains)+1)
                
                sizes = [np.sum(domains==dom)
                         for dom in range(self.n_sources_)]
                
                max_size = np.max(sizes)
                repeats = np.ceil(max_size/sizes)
                
                dataset_Xs = tf.data.Dataset.zip(tuple(
                    tf.data.Dataset.from_tensor_slices(X[domains==dom]).repeat(repeats[dom])
                    for dom in range(self.n_sources_))
                )

                dataset_ys = tf.data.Dataset.zip(tuple(
                    tf.data.Dataset.from_tensor_slices(y[domains==dom]).repeat(repeats[dom])
                    for dom in range(self.n_sources_))
                )
            
            dataset_src = tf.data.Dataset.zip((dataset_Xs, dataset_ys))            
        else:
            dataset_src = X
            
        ### 2.2 Target
        Xt, yt = self._get_target_data(Xt, yt)
        if not isinstance(Xt, tf.data.Dataset):
            if yt is None:
                check_array(Xt, ensure_2d=True, allow_nd=True)
                dataset_tgt = tf.data.Dataset.from_tensor_slices(Xt)

            else:
                check_arrays(Xt, yt)
            
                if len(yt.shape) <= 1:
                    yt = yt.reshape(-1, 1)
                
                dataset_Xt = tf.data.Dataset.from_tensor_slices(Xt)
                dataset_yt = tf.data.Dataset.from_tensor_slices(yt)
                dataset_tgt = tf.data.Dataset.zip((dataset_Xt, dataset_yt))
            
        else:
            dataset_tgt = Xt
            
        # 3. Initialize networks
        if not hasattr(self, "_is_fitted"):
            self._is_fitted = True
            self._initialize_networks()
            if isinstance(Xt, tf.data.Dataset):
                first_elem = next(iter(Xt))
                if not isinstance(first_elem, tuple):
                    shape = first_elem.shape
                else:
                    shape = first_elem[0].shape
                if self._check_for_batch(Xt):
                    shape = shape[1:]
            else:
                shape = Xt.shape[1:]
            self._initialize_weights(shape)
            
        
        # 3.5 Get datasets length
        self.length_src_ = self._get_length_dataset(dataset_src, domain="src")
        self.length_tgt_ = self._get_length_dataset(dataset_tgt, domain="tgt")
        
        
        # 4. Prepare validation dataset
        if validation_data is None and validation_split>0.:
            if shuffle:
                dataset_src = dataset_src.shuffle(buffer_size=self.length_src_,
                                                  reshuffle_each_iteration=False)
            frac = int(self.length_src_*validation_split)
            validation_data = dataset_src.take(frac)
            dataset_src = dataset_src.skip(frac)
            if not self._check_for_batch(validation_data):
                validation_data = validation_data.batch(validation_batch_size)
        
        if validation_data is not None:
            if isinstance(validation_data, tf.data.Dataset):
                if not self._check_for_batch(validation_data):
                    validation_data = validation_data.batch(validation_batch_size)
            
        
        # 5. Set datasets
        # Same length for src and tgt + complete last batch + shuffle
        if shuffle:
            if buffer_size is None:                
                dataset_src = dataset_src.shuffle(buffer_size=self.length_src_,
                                                  reshuffle_each_iteration=True)
                dataset_tgt = dataset_tgt.shuffle(buffer_size=self.length_tgt_,
                                                  reshuffle_each_iteration=True)
            else:
                dataset_src = dataset_src.shuffle(buffer_size=buffer_size,
                                                  reshuffle_each_iteration=True)
                dataset_tgt = dataset_tgt.shuffle(buffer_size=buffer_size,
                                                  reshuffle_each_iteration=True)
        
        max_size = max(self.length_src_, self.length_tgt_)
        max_size = np.ceil(max_size / batch_size) * batch_size
        repeat_src = np.ceil(max_size/self.length_src_)
        repeat_tgt = np.ceil(max_size/self.length_tgt_)

        dataset_src = dataset_src.repeat(repeat_src).take(max_size)
        dataset_tgt = dataset_tgt.repeat(repeat_tgt).take(max_size)

        self.total_steps_ = float(np.ceil(max_size/batch_size)*epochs)
        
        # 5. Pretraining
        if not hasattr(self, "pretrain_"):
            if not hasattr(self, "pretrain"):
                self.pretrain_ = False
            else:
                self.pretrain_ = self.pretrain
        
        if self.pretrain_:
                        
            if self._is_compiled:
                warnings.warn("The model has already been compiled. "
                              "To perform pretraining, the model will be "
                              "compiled again. Please make sure to pass "
                              "the compile parameters in __init__ to avoid errors.")
            
            compile_params = self._filter_params(super().compile, prefix="pretrain")
            self.compile(**compile_params)
            
            if not hasattr(self, "pretrain_history_"):
                self.pretrain_history_ = {}
                
            prefit_params = self._filter_params(super().fit, prefix="pretrain")
            
            pre_verbose = prefit_params.pop("verbose", verbose)
            pre_epochs = prefit_params.pop("epochs", epochs)
            pre_batch_size = prefit_params.pop("batch_size", batch_size)
            prefit_params.pop("validation_data", None)
            
            # !!! shuffle is already done
            dataset = tf.data.Dataset.zip((dataset_src, dataset_tgt))
            
            if not self._check_for_batch(dataset):
                dataset = dataset.batch(pre_batch_size)

            hist = super().fit(dataset, validation_data=validation_data,
                               epochs=pre_epochs, verbose=pre_verbose, **prefit_params)

            for k, v in hist.history.items():
                self.pretrain_history_[k] = self.pretrain_history_.get(k, []) + v
                
            self._initialize_pretain_networks()
        
        # 6. Compile
        if (not self._is_compiled) or (self.pretrain_):
            self.compile()
        
        if not hasattr(self, "history_"):
            self.history_ = {}

        # .7 Training
        dataset = tf.data.Dataset.zip((dataset_src, dataset_tgt))
        
        if not self._check_for_batch(dataset):
            dataset = dataset.batch(batch_size)

        self.pretrain_ = False
        
        hist = super().fit(dataset, validation_data=validation_data, **fit_params)
               
        for k, v in hist.history.items():
            self.history_[k] = self.history_.get(k, []) + v
        return self
    
    
    def compile(self,
                optimizer=None,
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                steps_per_execution=None,
                **kwargs):
        """
        Configures the model for training.

        Parameters
        ----------
        optimizer: str or `tf.keras.optimizer` instance
            Optimizer

        loss: str or `tf.keras.losses.Loss` instance
            Loss function. A loss function is any callable
            with the signature `loss = fn(y_true, y_pred)`,
            where `y_true` are the ground truth values, and
            `y_pred` are the model's predictions.
            `y_true` should have shape
            `(batch_size, d0, .. dN)` (except in the case of
            sparse loss functions such as
            sparse categorical crossentropy which expects integer arrays of shape
            `(batch_size, d0, .. dN-1)`).
            `y_pred` should have shape `(batch_size, d0, .. dN)`.
            The loss function should return a float tensor.
            If a custom `Loss` instance is
            used and reduction is set to `None`, return value has shape
            `(batch_size, d0, .. dN-1)` i.e. per-sample or per-timestep loss
            values; otherwise, it is a scalar. If the model has multiple outputs,
            you can use a different loss on each output by passing a dictionary
            or a list of losses. The loss value that will be minimized by the
            model will then be the sum of all individual losses, unless
            `loss_weights` is specified.

        metrics: list of str or list of `tf.keras.metrics.Metric` instance
            List of metrics to be evaluated by the model during training
            and testing. Typically you will use `metrics=['accuracy']`. A
            function is any callable with the signature `result = fn(y_true,
            y_pred)`. To specify different metrics for different outputs of a
            multi-output model, you could also pass a dictionary, such as
            `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
            You can also pass a list to specify a metric or a list of metrics
            for each output, such as `metrics=[['accuracy'], ['accuracy', 'mse']]`
            or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
            strings 'accuracy' or 'acc', we convert this to one of
            `tf.keras.metrics.BinaryAccuracy`,
            `tf.keras.metrics.CategoricalAccuracy`,
            `tf.keras.metrics.SparseCategoricalAccuracy` based on the loss
            function used and the model output shape. We do a similar
            conversion for the strings 'crossentropy' and 'ce' as well.

        loss_weights: List or dict of floats
            Scalars to weight the loss contributions of different model
            outputs. The loss value that will be minimized by the model will then
            be the *weighted sum* of all individual losses, weighted by the
            `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping to the model's
            outputs. If a dict, it is expected to map output names (strings)
            to scalar coefficients.

        weighted_metrics: list of metrics
            List of metrics to be evaluated and weighted by
            `sample_weight` or `class_weight` during training and testing.

        run_eagerly: bool (default=False)
            If `True`, this `Model`'s logic will not be wrapped
            in a `tf.function`. Recommended to leave
            this as `None` unless your `Model` cannot be run inside a
            `tf.function`. `run_eagerly=True` is not supported when using
            `tf.distribute.experimental.ParameterServerStrategy`.

        steps_per_execution: int (default=1)
            The number of batches to run during each
            `tf.function` call. Running multiple batches
            inside a single `tf.function` call can greatly improve performance
            on TPUs or small models with a large Python overhead.
            At most, one full epoch will be run each
            execution. If a number larger than the size of the epoch is passed,
            the execution will be truncated to the size of the epoch.
            Note that if `steps_per_execution` is set to `N`,
            `Callback.on_batch_begin` and `Callback.on_batch_end` methods
            will only be called every `N` batches
            (i.e. before/after each `tf.function` execution).

        **kwargs: key, value arguments
            Arguments supported for backwards compatibility only.
            
        Returns
        -------
        None: None
        """
        if hasattr(self, "_adapt_metrics") and metrics is None:
            metrics = self._adapt_metrics
                
        if not isinstance(metrics, (list, dict, type(None))):
            raise ValueError("`metrics` argument should be an instance "
                             "of dict or list")
        
        if isinstance(metrics, dict):
            metrics_disc = metrics.get("disc")
            metrics_task = metrics.get("task")
        else:
            metrics_disc = metrics
            metrics_task = metrics
        if metrics_disc is None:
            metrics_disc = []
        
        self.disc_metrics = [tf.keras.metrics.get(m) for m in metrics_disc]
        for metric, i in zip(self.disc_metrics,
                             range(len(self.disc_metrics))):
            if hasattr(metric, "name"):
                name = metric.name
            elif hasattr(metric, "__name__"):
                name = metric.__name__
            elif hasattr(metric, "__class__"):
                name = metric.__class__.__name__
            else:
                name = "met"
            if "_" in name:
                new_name = ""
                for split in name.split("_"):
                    if len(split) > 0:
                        new_name += split[0]
                name = new_name
            else:
                name = name[:3]
            metric.name = name
            
            if metric.name in ["acc", "Acc", "accuracy", "Accuracy"]:
                self.disc_metrics[i] = accuracy
                self.disc_metrics[i].name = "acc"
        
        compile_params = dict(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics_task,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
        )
        compile_params = {k: v for k, v in compile_params.items() if v is not None}
        
        compile_params = self._filter_params(super().compile, compile_params)
        
        if ((not "optimizer" in compile_params) or 
            (compile_params["optimizer"] is None)):
            compile_params["optimizer"] = RMSprop()
        else:
            if optimizer is None:
                if not isinstance(compile_params["optimizer"], str):
                    optim_params = self._filter_params(
                        compile_params["optimizer"].__init__)
                    if len(optim_params) > 0:
                        kwargs = compile_params["optimizer"].get_config()
                        kwargs.update(optim_params)
                        optimizer = compile_params["optimizer"].__class__(**kwargs)
                    else:
                        optimizer = compile_params["optimizer"]
                    compile_params["optimizer"] = optimizer
        
        if not "loss" in compile_params:
            compile_params["loss"] = "mse"
        
        self.task_loss_ = tf.keras.losses.get(compile_params["loss"])
        
        super().compile(
            **compile_params
        )
        
        # Set optimizer for encoder and discriminator
        if not hasattr(self, "optimizer_enc"):
            self.optimizer_enc = self.optimizer
        if not hasattr(self, "optimizer_disc"):
            self.optimizer_disc = self.optimizer
    
    
    def call(self, inputs):
        x = self.encoder_(inputs)
        return self.task_(x)
    
    
    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
        
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(Xs, training=True)
            loss = self.compiled_loss(
              ys, y_pred, regularization_losses=self.losses)
            loss = tf.reduce_mean(loss)

        # Run backwards pass.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(ys, y_pred)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics
    
        
    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        """
        Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for performance in
        large scale inputs. For small amount of inputs that fit in one batch,
        directly using `__call__()` is recommended for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `tf.keras.layers.BatchNormalization` that behaves differently during
        inference. Also, note the fact that test loss is not affected by
        regularization layers like noise and dropout.
        
        Parameters
        ----------
        x: array
            Input samples.
            
        batch_size: int (default=`None`)
            Number of samples per batch.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of dataset, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
            
        verbose: int (default=0)
            Verbosity mode, 0 or 1.
        
        steps: int (default=None)
            Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`. If x is a `tf.data`
            dataset and `steps` is None, `predict()` will
            run until the input dataset is exhausted.
            
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during prediction.
            See [callbacks](/api_docs/python/tf/keras/callbacks).

        max_queue_size: int (default=10)
            Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.

        workers: int (default=1)
            Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up when using
            process-based threading. If unspecified, `workers` will default
            to 1.
            
        use_multiprocessing: bool (default=False)
            Used for generator or `keras.utils.Sequence` input only.
            If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

        Returns
        -------
        y_pred : array
            Numpy array(s) of predictions.
        """
        return super().predict(x,
                    batch_size=batch_size,
                    verbose=verbose,
                    steps=steps,
                    callbacks=callbacks,
                    max_queue_size=max_queue_size,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing)
    
    
    def transform(self, X):
        """
        Return the encoded features of X.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        X_enc : array
            predictions of encoder network
        """
        return self.encoder_.predict(X)
    
    
    def predict_disc(self, X):
        """
        Return predictions of the discriminator on the encoded features.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_disc : array
            predictions of discriminator network
        """     
        return self.discriminator_.predict(self.transform(X))
    
    
    def predict_task(self, X):
        """
        Return predictions of the task on the encoded features.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_task : array
            predictions of task network
        """     
        return self.task_.predict(self.transform(X))
    
    
    def score(self, X, y, sample_weight=None):
        """
        Return the evaluation of the model on X, y.
        
        Call `evaluate` on tensorflow Model.
        
        Parameters
        ----------
        X : array
            input data
            
        y : array
            output data
            
        sample_weight : array (default=None)
            Sample weights
            
        Returns
        -------
        score : float
            Score.
        """
        if hasattr(X, "shape") and np.prod(X.shape) <= 10**8:
            score = self.evaluate(
                    X, y,
                    sample_weight=sample_weight,
                    batch_size=len(X)
                )
        else:
            score = self.evaluate(
                X, y,
                sample_weight=sample_weight
            )
        if isinstance(score, (tuple, list)):
            score = score[0]
        return score
    
    
#     def _check_validation_data(self, validation_data, batch_size, shuffle):
#         if isinstance(validation_data, tuple):
#             X_val = validation_data[0]
#             y_val = validation_data[1]
        
#             validation_data = tf.data.Dataset.zip(
#                 (tf.data.Dataset.from_tensor_slices(X_val),
#                  tf.data.Dataset.from_tensor_slices(y_val))
#             )
#             if shuffle:
#                 validation_data = validation_data.shuffle(buffer_size=1024).batch(batch_size)
#             else:
#                 validation_data = validation_data.batch(batch_size)
#         return validation_data

    
    def _get_legal_params(self, params):
        legal_params_fct = [self.__init__, super().fit, super().compile]
        
        if "optimizer" in params:
            optimizer = params["optimizer"]
        elif hasattr(self, "optimizer"):
            optimizer = self.optimizer
        else:
            optimizer = None
            
        if (optimizer is not None) and (not isinstance(optimizer, str)):
            legal_params_fct.append(optimizer.__init__)
        
        legal_params = ["domain", "val_sample_size", "optimizer_enc", "optimizer_disc"]
        for func in legal_params_fct:
            args = [
                p.name
                for p in inspect.signature(func).parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
            legal_params = legal_params + args
        
        if "pretrain" in legal_params:
            legal_params_fct = [super().fit, super().compile]
            if "pretrain__optimizer" in params:
                if not isinstance(params["pretrain__optimizer"], str):
                    legal_params_fct.append(params["pretrain__optimizer"].__init__)

            for func in legal_params_fct:
                args = [
                    p.name
                    for p in inspect.signature(func).parameters.values()
                    if p.name != "self" and p.kind != p.VAR_KEYWORD
                ]
                legal_params = legal_params + ["pretrain__"+name for name in args]
        return legal_params
    
    
    def _initialize_weights(self, shape_X):
        self(np.zeros((1,) + shape_X))
        if hasattr(self, "encoder_"):
            X_enc = self.encoder_(np.zeros((1,) + shape_X))
            if hasattr(self, "discriminator_"):
                self.discriminator_(X_enc)
                
    
    def _get_length_dataset(self, dataset, domain="src"):
        try:
            length = len(dataset)
        except:
            if self.verbose:
                print("Computing %s dataset size..."%domain)
            if not hasattr(self, "length_%s_"%domain):
                length = 0
                for _ in dataset:
                    length += 1
            else:
                length = getattr(self, "length_%s_"%domain)
            if self.verbose:
                print("Done!")
        return length
        
        
    def _check_for_batch(self, dataset):
        if "BatchDataset" in dataset.__class__.__name__:
            return True
        if hasattr(dataset, "_input_dataset"):
            return self._check_for_batch(dataset._input_dataset)
        elif hasattr(dataset, "_datasets"):
            checks = []
            for data in dataset._datasets:
                checks.append(self._check_for_batch(data))
            return np.all(checks)
        else:
            return False


    def _unpack_data(self, data):
        data_src = data[0]
        data_tgt = data[1]
        Xs = data_src[0]
        ys = data_src[1]
        if isinstance(data_tgt, tuple):
            Xt = data_tgt[0]
            yt = data_tgt[1]
            return Xs, Xt, ys, yt
        else:
            Xt = data_tgt
            return Xs, Xt, ys, None
    
    
    def _get_disc_metrics(self, ys_disc, yt_disc):
        disc_dict = {}
        return disc_dict
    
    
    def _initialize_networks(self):
        if self.encoder is None:
            self.encoder_ = get_default_encoder(name="encoder", state=self.random_state)
        else:
            self.encoder_ = check_network(self.encoder,
                                          copy=self.copy,
                                          name="encoder")
        if self.task is None:
            self.task_ = get_default_task(name="task", state=self.random_state)
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")
        if self.discriminator is None:
            self.discriminator_ = get_default_discriminator(name="discriminator", state=self.random_state)
        else:
            self.discriminator_ = check_network(self.discriminator,
                                                copy=self.copy,
                                                name="discriminator")
            
    def _initialize_pretain_networks(self):
        pass
