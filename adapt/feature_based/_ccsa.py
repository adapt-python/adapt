import numpy as np
import tensorflow as tf

from adapt.base import BaseAdaptDeep, make_insert_doc
from adapt.utils import set_random_seed


EPS = np.finfo(np.float32).eps

def pairwise_y(X, Y):
    batch_size_x = tf.shape(X)[0]
    batch_size_y = tf.shape(Y)[0]
    dim = tf.reduce_prod(tf.shape(X)[1:])
    X = tf.reshape(X, (batch_size_x, dim))
    Y = tf.reshape(Y, (batch_size_y, dim))
    X = tf.tile(tf.expand_dims(X, -1), [1, 1, batch_size_y])
    Y = tf.tile(tf.expand_dims(Y, -1), [1, 1, batch_size_x])
    return tf.reduce_sum(tf.abs(X-tf.transpose(Y)), 1)/2.


def pairwise_X(X, Y):
    X2 = tf.tile(tf.reduce_sum(tf.square(X), axis=1, keepdims=True), [1, tf.shape(Y)[0]])
    Y2 = tf.tile(tf.reduce_sum(tf.square(Y), axis=1, keepdims=True), [1, tf.shape(X)[0]])
    XY = tf.matmul(X, tf.transpose(Y))
    return X2 + tf.transpose(Y2) - 2*XY


@make_insert_doc(["encoder", "task"], supervised=True)
class CCSA(BaseAdaptDeep):
    """
    CCSA : Classification and Contrastive Semantic Alignment
    
    CCSA is a supervised feature based method for classification.
    
    It aims at producing an encoded space where the distances between
    source and target pairs from the same label are minized, whereas
    the distances between pairs from different labels are maximized.
    
    The optimization can be written as follows:
    
    .. math::
    
        \mathcal{L}_{CCSA} = \\gamma \mathcal{L}_{task}(h \circ g) +
        (1-\\gamma) (\mathcal{L}_{SA}(g) + \mathcal{L}_{S}(g))

    Where:
    
    .. math::
    
        \mathcal{L}_{SA}(g) = \sum_{i, j; \; y^s_i = y^t_j} || g(x^s_i) - g(x^t_j) ||^2
        
    .. math::
    
        \mathcal{L}_{S}(g) = \sum_{i, j; \; y^s_i \\neq y^t_j} \max(0, m - || g(x^s_i) - g(x^t_j) ||^2)
    
    With:
    
    - :math:`(x^s_i, y^s_i)` the labeled source data (:math:`y^s_i` gives the label)
    - :math:`(x^t_i, y^t_i)` the labeled target data
    - :math:`g, h` are respectively the **encoder** and the **task** networks
    - :math:`\\gamma` is the trade-off parameter.
    - :math:`m` is the margin parameter.
    
    Parameters
    ----------
    margin : float (default=1.)
        Margin for the inter-class separation.
        The higher the margin, the more the classes
        will be separated in the encoded space.
        
    gamma : float  (default=0.5)
        Trade-off parameter. ``0<gamma<1``
        If ``gamma`` is close to 1 more
        importance are given to the task. If
        gamma is close to 0, more importance
        are given to the contrastive loss.
    
    Attributes
    ----------    
    encoder_ : tensorflow Model
        encoder network.
        
    task_ : tensorflow Model
        task network.
        
    history_ : dict
        history of the losses and metrics across the epochs.
        If ``yt`` is given in ``fit`` method, target metrics
        and losses are recorded too.
        
    See also
    --------
    CDAN
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.utils import make_classification_da
    >>> from adapt.feature_based import CCSA
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> model = CCSA(margin=1., gamma=0.5, Xt=Xt, metrics=["acc"], random_state=0)
    >>> model.fit(Xs, ys, epochs=100, verbose=0)
    >>> model.score(Xt, yt)
    1/1 [==============================] - 0s 180ms/step - loss: 0.1550 - acc: 0.8900
    0.15503168106079102
        
    References
    ----------
    .. [1] `[1] <https://arxiv.org/abs/1709.10190>`_ S. Motiian, M. Piccirilli, \
D. A Adjeroh, and G. Doretto. "Unified deep supervised domain adaptation and \
generalization". In ICCV 2017.
    """
    
    def __init__(self,
                 encoder=None,
                 task=None,
                 Xt=None,
                 yt=None,
                 margin=1.,
                 gamma=0.5,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):

        names = self._get_param_names()
        kwargs = {k: v for k, v in locals().items() if k in names}
        kwargs.update(params)
        super().__init__(**kwargs)
        
        
    def train_step(self, data):
        # Unpack the data.
        Xs, Xt, ys, yt = self._unpack_data(data)
        
        # Check that yt is not None
        if yt is None:
            raise ValueError("The target labels `yt` is `None`, CCSA is a supervised"
                             " domain adaptation method and need `yt` to be specified.")
        
        # Check shape of ys
        if len(ys.get_shape()) <= 1 or ys.get_shape()[1] == 1:
            self._ys_is_1d = True
        else:
            self._ys_is_1d = False

        # loss
        with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape:            
            # Forward pass
            Xs_enc = self.encoder_(Xs, training=True)
            ys_pred = self.task_(Xs_enc, training=True)
            
            # Change type
            ys = tf.cast(ys, ys_pred.dtype)
            yt = tf.cast(yt, ys_pred.dtype)
            
            Xt_enc = self.encoder_(Xt, training=True)
            
            dist_y = pairwise_y(ys, yt)            
            dist_X = pairwise_X(Xs_enc, Xt_enc)
            
            if self._ys_is_1d:
                dist_y *= 2.
            
            contrastive_loss = tf.reduce_sum(dist_y * tf.maximum(0., self.margin - dist_X), 1) / (tf.reduce_sum(dist_y, 1) + EPS)
            contrastive_loss += tf.reduce_sum((1-dist_y) * dist_X, 1) / (tf.reduce_sum(1-dist_y, 1) + EPS)
            contrastive_loss = tf.reduce_mean(contrastive_loss)
            contrastive_loss *= 0.5
            
            # Reshape
            ys_pred = tf.reshape(ys_pred, tf.shape(ys))
            
            # Compute the loss value
            task_loss = tf.reduce_mean(self.task_loss_(ys, ys_pred))
                        
            enc_loss = self.gamma * task_loss + (1-self.gamma) * contrastive_loss
            
            task_loss += sum(self.task_.losses)
            enc_loss += sum(self.encoder_.losses)
            
        # Compute gradients
        trainable_vars_task = self.task_.trainable_variables
        trainable_vars_enc = self.encoder_.trainable_variables
        
        gradients_task = task_tape.gradient(task_loss, trainable_vars_task)
        gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
        self.optimizer_enc.apply_gradients(zip(gradients_enc, trainable_vars_enc))
        
        # Update metrics
        self.compiled_metrics.update_state(ys, ys_pred)
        self.compiled_loss(ys, ys_pred)
        # Return a dict mapping metric names to current value
        logs = {m.name: m.result() for m in self.metrics}
        logs.update({"contrast": contrastive_loss})
        return logs