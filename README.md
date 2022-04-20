# ADAPT

[![PyPI version](https://badge.fury.io/py/adapt.svg)](https://pypi.org/project/adapt)
[![Build Status](https://github.com/adapt-python/adapt/workflows/build/badge.svg)](https://github.com/adapt-python/adapt/actions)
[![Python Version](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8|%203.9-blue)](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8|%203.9-blue)
[![Codecov Status](https://codecov.io/gh/adapt-python/adapt/branch/master/graph/badge.svg?token=IWQXMYGY2Q)](https://codecov.io/gh/adapt-python/adapt)

**A**wesome **D**omain **A**daptation **P**ython **T**oolbox

---

ADAPT is an open source library providing numerous tools to perform Transfer Learning and Domain Adaptation.

The purpose of the ADAPT library is to facilitate the access to transfer leanring algorithms for a large public, including industrial players. ADAPT is specifically designed for [Scikit-learn](https://scikit-learn.org/stable/) and [Tensorflow](https://www.tensorflow.org/) users with a "user-friendly" approach. All objects in ADAPT implement the ***fit***, ***predict*** and ***score*** methods like any scikit-learn object. A very detailed documentation with several examples is provided:

:arrow_right: [Documentation](https://adapt-python.github.io/adapt/)

<table>
  <tr valign="top">
    <td width="50%" >
        <a href="doc/examples/Sample_bias_example.html">
            <br>
            <b>Sample bias correction</b>
            <br>
            <br>
            <img src="src_docs/_static/images/sample_bias_corr_img.png">
        </a>
    </td>
    <td width="50%">
        <a href="doc/examples/Flowers_example.html">
            <br>
            <b>Model-based Transfer</b>
            <br>
            <br>
            <img src="src_docs/_static/images/finetuned.png">
        </a>
    </td>
  </tr>
  <tr valign="top">
    <td width="50%">
        <a href="doc/examples/Flowers_example.html">
            <br>
            <b>Deep Domain Adaptation</b>
            <br>
            <br>
            <img src="src_docs/_static/images/office_item.png">
        </a>
    </td>
    <td width="50%">
        <a href="https://adapt-python.github.io/adapt/examples/Multi_fidelity.html">
            <br>
            <b>Multi-Fidelity Transfer</b>
            <br>
            <br>
            <img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/multifidelity_setup.png">
        </a>
    </td>
  </tr>
</table>

## Installation and Usage

This package is available on [Pypi](https://pypi.org/project/adapt) and can be installed with the following command line: 

```
pip install adapt
```

The following dependencies are required and will be installed with the library:
- `numpy`
- `scipy`
- `tensorflow` (>= 2.0)
- `scikit-learn`
- `cvxopt`

If for some reason, these packages failed to install, you can do it manually with:

```
pip install numpy scipy tensorflow scikit-learn cvxopt
```

Finally import the module in your python scripts with:

```python
import adapt
```

An usage example is given in the [Qick-Start](#Quick-Start) below.


## Quick Start

```python
import numpy as np
from adapt.feature_based import DANN
np.random.seed(0)

# Xs and Xt are shifted along the second feature.
Xs = np.concatenate((np.random.random((100, 1)),
                     np.zeros((100, 1))), 1)
Xt = np.concatenate((np.random.random((100, 1)),
                     np.ones((100, 1))), 1)
ys = 0.2 * Xs[:, 0]
yt = 0.2 * Xt[:, 0]

# With lambda set to zero, no adaptation is performed.
model = DANN(lambda_=0., random_state=0)
model.fit(Xs, ys, Xt=Xt, epochs=100, verbose=0)
print(model.evaluate(Xt, yt)) # This gives the target score at the last training epoch.
>>> 0.0231

# With lambda set to 0.1, the shift is corrected, the target score is then improved.
model = DANN(lambda_=0.1, random_state=0)
model.fit(Xs, ys, Xt=Xt, epochs=100, verbose=0)
model.evaluate(Xt, yt)
>>> 0.0011
```


## Content

ADAPT package is divided in three sub-modules containing the following domain adaptation methods:

### Feature-based methods

<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/feature_based.png">

- [FE](https://adapt-python.github.io/adapt/generated/adapt.feature_based.FE.html) (*Frustratingly Easy Domain Adaptation*) [[paper]](https://arxiv.org/pdf/0907.1815.pdf)
- [mSDA](https://adapt-python.github.io/adapt/generated/adapt.feature_based.mSDA.html) (*marginalized Stacked Denoising Autoencoder*) [[paper]](https://arxiv.org/ftp/arxiv/papers/1206/1206.4683.pdf)
- [DANN](https://adapt-python.github.io/adapt/generated/adapt.feature_based.DANN.html) (*Discriminative Adversarial Neural Network*) [[paper]](https://jmlr.org/papers/volume17/15-239/15-239.pdf)
- [ADDA](https://adapt-python.github.io/adapt/generated/adapt.feature_based.ADDA.html) (*Adversarial Discriminative Domain Adaptation*) [[paper]](https://arxiv.org/pdf/1702.05464.pdf)
- [CORAL](https://adapt-python.github.io/adapt/generated/adapt.feature_based.CORAL.html) (*CORrelation ALignment*) [[paper]](https://arxiv.org/pdf/1511.05547.pdf)
- [DeepCORAL](https://adapt-python.github.io/adapt/generated/adapt.feature_based.DeepCORAL.html) (*Deep CORrelation ALignment*) [[paper]](https://arxiv.org/pdf/1607.01719.pdf)
- [MCD](https://adapt-python.github.io/adapt/generated/adapt.feature_based.MCD.html) (*Maximum Classifier Discrepancy*) [[paper]](https://arxiv.org/pdf/1712.02560.pdf)
- [MDD](https://adapt-python.github.io/adapt/generated/adapt.feature_based.MDD.html) (*Margin Disparity Discrepancy*) [[paper]](https://arxiv.org/pdf/1904.05801.pdf)
- [WDGRL](https://adapt-python.github.io/adapt/generated/adapt.feature_based.WDGRL.html) (*Wasserstein Distance Guided Representation Learning*) [[paper]](https://arxiv.org/pdf/1707.01217.pdf)

### Instance-based methods

<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/instance_based.png">

- [KMM](https://adapt-python.github.io/adapt/generated/adapt.instance_based.KMM.html) (*Kernel Mean Matching*) [[paper]](https://proceedings.neurips.cc/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf)
- [KLIEP](https://adapt-python.github.io/adapt/generated/adapt.instance_based.KLIEP.html) (*Kullback–Leibler Importance Estimation Procedure*) [[paper]](https://proceedings.neurips.cc/paper/2007/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper.pdf)
- [TrAdaBoost](https://adapt-python.github.io/adapt/generated/adapt.instance_based.TrAdaBoost.html) (*Transfer AdaBoost*) [[paper]](https://cse.hkust.edu.hk/~qyang/Docs/2007/tradaboost.pdf)
- [TrAdaBoostR2](https://adapt-python.github.io/adapt/generated/adapt.instance_based.TrAdaBoostR2.html) (*Transfer AdaBoost for Regression*) [[paper]](https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf)
- [TwoStageTrAdaBoostR2](https://adapt-python.github.io/adapt/generated/adapt.instance_based.TwoStageTrAdaBoostR2.html) (*Two Stage Transfer AdaBoost for Regression*) [[paper]](https://www.cs.utexas.edu/~dpardoe/papers/ICML10.pdf)

### Parameter-based methods

<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/parameter_based.png">

- [RegularTransferLR](https://adapt-python.github.io/adapt/generated/adapt.parameter_based.RegularTransferLR.html) (*Regular Transfer with Linear Regression*) [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/07/2004-chelba-emnlp.pdf)
- [RegularTransferLC](https://adapt-python.github.io/adapt/generated/adapt.parameter_based.RegularTransferLC.html) (*Regular Transfer with Linear Classification*) [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/07/2004-chelba-emnlp.pdf)
- [RegularTransferNN](https://adapt-python.github.io/adapt/generated/adapt.parameter_based.RegularTransferNN.html) (*Regular Transfer with Neural Network*) [[paper]](https://hal.inria.fr/hal-00911179v1/document)


## Reference

If you use this library in your research, please cite ADAPT using the following reference: https://arxiv.org/pdf/2107.03049.pdf

```
@article{de2021adapt,
	  title={ADAPT: Awesome Domain Adaptation Python Toolbox},
	  author={de Mathelin, Antoine and Deheeger, Fran{\c{c}}ois and Richard, Guillaume and Mougeot, Mathilde and Vayatis, Nicolas},
	  journal={arXiv preprint arXiv:2107.03049},
	  year={2021}
	}
```

## Acknowledgement

This work has been funded by Michelin and the Industrial Data Analytics and Machine Learning chair from ENS Paris-Saclay, Borelli center.

[<img src="src_docs/_static/images/michelin.png" width=200px alt="Michelin">](https://www.michelin.com/) <img src="src_docs/_static/images/idaml.jpg" width=200px alt="IDAML"> <img src="src_docs/_static/images/borelli.jpg" alt="Centre Borelli" width=150px>
