import numpy as np

class FE(object):
    """
    FE: Frustratingly Easy Domain Adaptation
        
    Reference: Daume III, H.
    "Frustratingly easy domain adaptation".
    In ACL, 2007.
    
    Parameters
    ----------
    get_estimator: callable, optional
        Constructor for the estimator.
        
    kwargs: key, value arguments, optional
        Additional arguments for constructor
    """
    def __init__(self, get_estimator, **kwargs):
        self.get_estimator = get_estimator
        self.kwargs = kwargs

    
    def fit(self, X, y, src_index, tgt_index, **fit_params):
        """
        Fit FE
        
        Parameters
        ----------
        X, y: numpy arrays
            Input data
            
        src_index: iterable
            indexes of source labeled data in X, y
            
        fit_params: key, value arguments
            Arguments to pass to the fit method (epochs, batch_size...)
            
        Returns
        -------
        self 
        """
        
        Xs = X[src_index]
        ys = y[src_index]
        Xt = X[tgt_index]
        yt = y[tgt_index]
            
        self.estimator_ = self.get_estimator(**self.kwargs)
        
        X_augmented_src = np.concatenate((Xs, np.zeros(Xs.shape), Xs), axis=1)
        X_augmented_tgt = np.concatenate((np.zeros(Xt.shape), Xt, Xt), axis=1)
        
        X = np.concatenate((X_augmented_src, X_augmented_tgt))
        y = np.concatenate((ys, yt))
        
        self.estimator_.fit(X, y, **fit_params)
        return self
    
    
    def predict(self, X):
        """
        Predict method: return the prediction of estimator
        on the augmented features
        
        Parameters
        ----------
        X: array
            input data
            
        domain: str, optional (default="target")
            Choose between source and target
            augmented feature space.
            
        Returns
        -------
        y_pred: array
            prediction of task network
        """
        if domain == "target":
            X_augmented = np.concatenate((np.zeros(X.shape), X, X), axis=1)
        elif domain == "source":
            X_augmented = np.concatenate((X, np.zeros(X.shape), X), axis=1)
        else:
            raise ValueError("Choose between source or target for domain name")
        return self.estimator_.predict(X_augmented)