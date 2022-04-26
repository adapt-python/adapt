import numpy as np
import tensorflow as tf

from adapt.utils import make_classification_da
from adapt.feature_based import CCSA

np.random.seed(0)
tf.random.set_seed(0)

task = tf.keras.Sequential()
task.add(tf.keras.layers.Dense(50, activation="relu"))
task.add(tf.keras.layers.Dense(2, activation="softmax"))

ind = np.random.choice(100, 10)
Xs, ys, Xt, yt = make_classification_da()


def test_ccsa():
    ccsa = CCSA(task=task, loss="categorical_crossentropy",
                optimizer="adam", metrics=["acc"], gamma=0.1, random_state=0)
    ccsa.fit(Xs, tf.one_hot(ys, 2).numpy(), Xt=Xt[ind],
             yt=tf.one_hot(yt, 2).numpy()[ind], epochs=100, verbose=0)
    assert np.mean(ccsa.predict(Xt).argmax(1) == yt) > 0.8

    ccsa = CCSA(task=task, loss="categorical_crossentropy",
                optimizer="adam", metrics=["acc"], gamma=1., random_state=0)
    ccsa.fit(Xs, tf.one_hot(ys, 2).numpy(), Xt=Xt[ind],
             yt=tf.one_hot(yt, 2).numpy()[ind], epochs=100, verbose=0)

    assert np.mean(ccsa.predict(Xt).argmax(1) == yt) < 0.9