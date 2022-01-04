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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

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


base_doc_1 = """
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
        `_get_legal_params(params)`.
"""


def make_insert_doc(estimators=["estimator"]):
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
            splits[1] = (
                splits[1][:i-1]+
                doc_est+base_doc_1+
                splits[1][i-1:j+1]+
                base_doc_2+
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


    def score(self, X=None, y=None, sample_weight=None):
        """
        Return adaptation score.
        
        The normalized discrepancy distance is computed
        between the reweighted/transformed source input
        data and the target input data.
        
        Parameters
        ----------
        X : array (default=None)
            Not used, present here for sklearn consistency
            by convention. The training source data
            are used instead.
            
        y : array (default=None)
            Not used, present here for sklearn consistency
            by convention.
            
        sample_weight : array (default=None)
            Not used, present here for sklearn consistency
            by convention.
            
        Returns
        -------
        score : float
            Adaptation score.
        """
        if hasattr(self, "Xs_"):
            Xs = self.Xs_
            Xt = self.Xt_
            src_index = self.src_index_
            score = self.score_adapt(Xs, Xt, src_index)
        elif hasattr(self, "score_estimator"):
            score = self.score_estimator(X, y, sample_weight=sample_weight)
        else:
            raise NotFittedError("Adapt model is not fitted yet, "
                                 "please call 'fit' first.")
        return score


    def score_adapt(self, Xs, Xt, src_index=None):
        """
        Return adaptation score.
        
        The normalized discrepancy distance is computed
        between the reweighted/transformed source input
        data and the target input data.
        
        Parameters
        ----------
        Xs : array
            Source input data.
            
        Xt : array
            Source input data.
            
        src_index : array
            Index of Xs in the training data.
            
        Returns
        -------
        score : float
            Adaptation score.
        """
        Xs = np.array(Xs)
        Xt = np.array(Xt)
        if src_index is None:
            src_index = np.arange(len(Xs))
        if hasattr(self, "transform"):
            args = inspect.getfullargspec(self.transform).args
            if "domain" in args:
                Xt = self.transform(Xt, domain="tgt")
                Xs = self.transform(Xs, domain="src")
            else:
                Xt = self.transform(Xt)
                Xs = self.transform(Xs)
        elif hasattr(self, "predict_weights"):
            sample_weight = self.predict_weights()
            sample_weight = sample_weight[src_index]
            
            sample_weight = check_sample_weight(sample_weight, Xs)
            sample_weight /= sample_weight.sum()
            
            set_random_seed(self.random_state)
            bootstrap_index = np.random.choice(
            len(Xs), size=len(Xs), replace=True, p=sample_weight)
            Xs = Xs[bootstrap_index]
        else:
            raise ValueError("The Adapt model should implement"
                             " a transform or predict_weights methods")
        Xs = np.array(Xs)
        Xt = np.array(Xt)
        return normalized_linear_discrepancy(Xs, Xt)
    
    
    def _check_params(self, params):
        legal_params = self._get_legal_params(params)
        for key in params:
            if not key in legal_params:
                raise ValueError("%s is not a legal params for %s model. "
                                 "Legal params are: %s"%
                                 (key, self.__class__.__name__, str(legal_params)))
    
    
    def _filter_params(self, func, override={}, prefix=""):
        kwargs = {}
        args = inspect.getfullargspec(func).args
        for key, value in self.__dict__.items():
            new_key = key.replace(prefix+"__", "")
            if new_key in args and prefix in key:
                kwargs[new_key] = value
        kwargs.update(override)
        return kwargs


    def _save_validation_data(self, Xs, Xt):
        Xs = np.array(Xs)
        Xt = np.array(Xt)
        if hasattr(self, "val_sample_size"):
            set_random_seed(self.random_state)
            size = min(self.val_sample_size, len(Xs))
            src_index = np.random.choice(len(Xs), size, replace=False)
            self.Xs_ = Xs[src_index]
            size = min(self.val_sample_size, len(Xt))
            tgt_index = np.random.choice(len(Xt), size, replace=False)
            self.Xt_ = Xt[tgt_index]
        else:
            self.Xs_ = Xs
            self.Xt_ = Xt
            self.src_index_ = np.arange(len(Xs))
    
    
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
        

#     def get_params(self, deep=True):
#         """
#         Get parameters for this estimator.
        
#         Parameters
#         ----------
#         deep : bool (default=True)
#             If True, will return the parameters for this estimator and
#             contained subobjects that are estimators.
        
#         Returns
#         -------
#         params : dict
#             Parameter names mapped to their values.
#         """
#         out = dict()
#         legal_params = self._get_legal_params(self.__dict__)
#         params_names = set(self.__dict__) & set(legal_params)
#         for key in params_names:
#             value = getattr(self, key)
#             if deep and hasattr(value, "get_params"):
#                 deep_items = value.get_params().items()
#                 out.update((key + "__" + k, val) for k, val in deep_items)
#             out[key] = value
#         return out
    
    
    def fit(self, X, y, Xt=None, yt=None, domains=None, **fit_params):
        """
        Fit Adapt Model.

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
            Target input data. If None, the `Xt` argument
            given in `init` is used.
            
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
        if yt is not None:
            Xt, yt = check_arrays(Xt, yt)
        else:
            Xt = check_array(Xt)
        set_random_seed(self.random_state)

        self._save_validation_data(X, Xt)
 
        if hasattr(self, "fit_weights"):
            if self.verbose:
                print("Fit weights...")
            self.weights_ = self.fit_weights(Xs=X, Xt=Xt,
                                             ys=y, yt=yt,
                                             domains=domains)
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
        Fit estimator.
        
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
        X, y = check_arrays(X, y)
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
                            optimizer = compile_params["optimizer"].__class__(**optim_params)
                        else:
                            optimizer = compile_params["optimizer"]
                        compile_params["optimizer"] = optimizer
                self.estimator_.compile(**compile_params)

        fit_params = self._filter_params(self.estimator_.fit, fit_params)
        
        if "sample_weight" in inspect.getfullargspec(self.estimator_.fit).args:
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
        Return estimator predictions.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_pred : array
            prediction of estimator.
        """      
        X = check_array(X, ensure_2d=True, allow_nd=True)
        predict_params = self._filter_params(self.estimator_.predict,
                                            predict_params)
        return self.estimator_.predict(X, **predict_params)


    def predict(self, X, **predict_params):
        """
        Return estimator predictions after
        adaptation.
        
        Parameters
        ----------
        X : array
            input data
            
        Returns
        -------
        y_pred : array
            prediction of the Adapt Model.
        """
        X = check_array(X, ensure_2d=True, allow_nd=True)
        if hasattr(self, "transform"):
            if "domain" in predict_params:
                domain = predict_params.pop("domain")
                X = self.transform(X, domain=domain)
            else:
                X = self.transform(X)
        return self.predict_estimator(X, **predict_params)


    def score_estimator(self, X, y, sample_weight=None):
        """
        Return the estimator score.
        
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
            
        Returns
        -------
        score : float
            estimator score.
        """
        X, y = check_arrays(X, y)
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
                
        legal_params = ["domain", "val_sample_size"]
        for func in legal_params_fct:
            args = list(inspect.getfullargspec(func).args)
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
        print("getting")
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
        print("setting")
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
        
        return dict(weights=weights,
                    config=config,
                    klass=klass)


    def _from_config_keras_model(self, dict_):
        weights = dict_["weights"]
        config = dict_["config"]
        klass = dict_["klass"]

        model = klass.from_config(config)

        if weights is not None:
            model.set_weights(weights)
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
    
    
    def fit(self, X, y, Xt=None, yt=None, domains=None, **fit_params):
        """
        Fit Model. Note that ``fit`` does not reset
        the model but extend the training.

        Parameters
        ----------
        X : array or Tensor
            Source input data.

        y : array or Tensor
            Source output data.
            
        Xt : array (default=None)
            Target input data. If None, the `Xt` argument
            given in `init` is used.

        yt : array (default=None)
            Target input data. If None, the `Xt` argument
            given in `init` is used.
            
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
        
        # 1. Initialize networks
        if not hasattr(self, "_is_fitted"):
            self._is_fitted = True
            self._initialize_networks()
            self._initialize_weights(X.shape[1:])
        
        # 2. Prepare dataset
        Xt, yt = self._get_target_data(Xt, yt)

        check_arrays(X, y)
        if len(y.shape) <= 1:
            y = y.reshape(-1, 1)
            
        if yt is None:
            yt = y
            check_array(Xt)
        else:
            check_arrays(Xt, yt)
        
        if len(yt.shape) <= 1:
            yt = yt.reshape(-1, 1)
            
        self._save_validation_data(X, Xt)
        
        domains = fit_params.pop("domains", None)
        
        if domains is None:
            domains = np.zeros(len(X))
        
        domains = self._check_domains(domains)

        self.n_sources_ = int(np.max(domains)+1)
        
        sizes = np.array(
            [np.sum(domains==dom) for dom in range(self.n_sources_)]+
            [len(Xt)])
        
        max_size = np.max(sizes)
        repeats = np.ceil(max_size/sizes)
        
        dataset_X = tf.data.Dataset.zip(tuple(
            tf.data.Dataset.from_tensor_slices(X[domains==dom]).repeat(repeats[dom])
            for dom in range(self.n_sources_))+
            (tf.data.Dataset.from_tensor_slices(Xt).repeat(repeats[-1]),)
        )
                                        
        dataset_y = tf.data.Dataset.zip(tuple(
            tf.data.Dataset.from_tensor_slices(y[domains==dom]).repeat(repeats[dom])
            for dom in range(self.n_sources_))+
            (tf.data.Dataset.from_tensor_slices(yt).repeat(repeats[-1]),)
        )
        
        
        # 3. Get Fit params
        fit_params = self._filter_params(super().fit, fit_params)
        
        verbose = fit_params.get("verbose", 1)
        epochs = fit_params.get("epochs", 1)
        batch_size = fit_params.pop("batch_size", 32)
        shuffle = fit_params.pop("shuffle", True)
        
        # 4. Pretraining
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
            pre_shuffle = prefit_params.pop("shuffle", shuffle)
            
            if pre_shuffle:
                dataset = tf.data.Dataset.zip((dataset_X, dataset_y)).shuffle(buffer_size=1024).batch(pre_batch_size)
            else:
                dataset = tf.data.Dataset.zip((dataset_X, dataset_y)).batch(pre_batch_size)
                
            hist = super().fit(dataset, epochs=pre_epochs, verbose=pre_verbose, **prefit_params)

            for k, v in hist.history.items():
                self.pretrain_history_[k] = self.pretrain_history_.get(k, []) + v
                
            self._initialize_pretain_networks()
            
        # 5. Training
        if (not self._is_compiled) or (self.pretrain_):
            self.compile()
        
        if not hasattr(self, "history_"):
            self.history_ = {}

        if shuffle:
            dataset = tf.data.Dataset.zip((dataset_X, dataset_y)).shuffle(buffer_size=1024).batch(batch_size)
        else:
            dataset = tf.data.Dataset.zip((dataset_X, dataset_y)).batch(batch_size)
            
        self.pretrain_ = False
        self.steps_ = tf.Variable(0.)
        self.total_steps_ = float(np.ceil(max_size/batch_size)*epochs)
        
        hist = super().fit(dataset, **fit_params)
               
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
        
        print(compile_params)
        
        compile_params = self._filter_params(super().compile, compile_params)
        
        print(compile_params)
        
        if ((not "optimizer" in compile_params) or 
            (compile_params["optimizer"] is None)):
            compile_params["optimizer"] = "rmsprop"
        else:
            if optimizer is None:
                if not isinstance(compile_params["optimizer"], str):
                    optim_params = self._filter_params(
                        compile_params["optimizer"].__init__)
                    if len(optim_params) > 0:
                        optimizer = compile_params["optimizer"].__class__(**optim_params)
                    else:
                        optimizer = compile_params["optimizer"]
                    compile_params["optimizer"] = optimizer
        
        if not "loss" in compile_params:
            compile_params["loss"] = "mse"
        
        self.task_loss_ = tf.keras.losses.get(compile_params["loss"])
        
        super().compile(
            **compile_params
        )
    
    
    def call(self, inputs):
        x = self.encoder_(inputs)
        return self.task_(x)
    
    
    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
        
        # Single source
        Xs = Xs[0]
        ys = ys[0]
        
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(Xs, training=True)
            loss = self.compiled_loss(
              ys, y_pred, regularization_losses=self.losses)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
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
        
        legal_params = ["domain", "val_sample_size"]
        for func in legal_params_fct:
            args = list(inspect.getfullargspec(func).args)
            legal_params = legal_params + args
        
        if "pretrain" in legal_params:
            legal_params_fct = [super().fit, super().compile]
            if "pretrain__optimizer" in params:
                if not isinstance(params["pretrain__optimizer"], str):
                    legal_params_fct.append(params["pretrain__optimizer"].__init__)

            for func in legal_params_fct:
                args = list(inspect.getfullargspec(func).args)
                legal_params = legal_params + ["pretrain__"+name for name in args]
        return legal_params
    
    
    def _initialize_weights(self, shape_X):
        self(np.zeros((1,) + shape_X))
        if hasattr(self, "encoder_"):
            X_enc = self.encoder_(np.zeros((1,) + shape_X))
            if hasattr(self, "discriminator_"):
                self.discriminator_(X_enc)
    
    
    def _check_pretrain(self, params):
        self.pretrain = False
        for param in params:
            if "pretrain__" in param:
                self.pretrain = True
                break


    def _unpack_data(self, data):
        data_X = data[0]
        data_y = data[1]
        Xs = data_X[:-1]
        Xt = data_X[-1]
        ys = data_y[:-1]
        yt = data_y[-1]
        return Xs, Xt, ys, ys
    
    
    def _get_disc_metrics(self, ys_disc, yt_disc):
        disc_dict = {}
        return disc_dict
    
    
    def _initialize_networks(self):
        if self.encoder is None:
            self.encoder_ = get_default_encoder(name="encoder")
        else:
            self.encoder_ = check_network(self.encoder,
                                          copy=self.copy,
                                          name="encoder")
        if self.task is None:
            self.task_ = get_default_task(name="task")
        else:
            self.task_ = check_network(self.task,
                                       copy=self.copy,
                                       name="task")
        if self.discriminator is None:
            self.discriminator_ = get_default_discriminator(name="discriminator")
        else:
            self.discriminator_ = check_network(self.discriminator,
                                                copy=self.copy,
                                                name="discriminator")
            
    def _initialize_pretain_networks(self):
        pass