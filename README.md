# ADAPT

[![PyPI version](https://badge.fury.io/py/adapt.svg)](https://pypi.org/project/adapt)
[![Build Status](https://github.com/adapt-python/adapt/workflows/build/badge.svg)](https://github.com/adapt-python/adapt/actions)
[![Python Version](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue)](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7-blue)
[![Codecov Status](https://codecov.io/gh/adapt-python/adapt/branch/master/graph/badge.svg?token=IWQXMYGY2Q)](https://codecov.io/gh/adapt-python/adapt)

**A**wesome **D**omain **A**daptation **P**ython **T**oolbox

ADAPT is a python library which provides several domain adaptation methods implemented with [Tensorflow](https://www.tensorflow.org/) and [Scikit-learn](https://scikit-learn.org/stable/).

## Documentation Website

Find the details of all implemented methods as well as illustrative examples here: [ADAPT Documentation Website](https://adapt-python.github.io/adapt/)

## Installation

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
model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
print(model.history_["task_t"][-1]) # This gives the target score at the last training epoch.
>>> 0.0240

# With lambda set to 0.1, the shift is corrected, the target score is then improved.
model = DANN(lambda_=0.1, random_state=0)
model.fit(Xs, ys, Xt, yt, epochs=100, verbose=0)
print(model.history_["task_t"][-1])
>>> 0.0022
```

## Examples

| Two Moons  | Classification | Regression  |
| :-------------: | :-------------: | :-------------: |
| [<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/two_moons_setup.png">](https://adapt-python.github.io/adapt/examples/Two_moons.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Tz-TIkHI8ashHP90Im6D3tMjZ3lkR7s6?usp=sharing) | [<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/classification_setup.png">](https://adapt-python.github.io/adapt/examples/Classification.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ANQUix9Y6V4RXu-vAaCFGmU979d5m4bO?usp=sharing)  | [<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/regression_setup.png">](https://adapt-python.github.io/adapt/examples/Regression.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1adhqoV6b0uEavLDmMfkiwtRjam0DrXux?usp=sharing) |

| Sample Bias   | Multi-Fidelity | Rotation |
| :-------------: | :-------------: | :-------------: |
| [<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/sample_bias_2d_setup.png">](https://adapt-python.github.io/adapt/examples/sample_bias_2d.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hbg2kDXKjKzeQKJSwxzaV7pwbmORhyA3?usp=sharing) | [<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/multifidelity_setup.png">](https://adapt-python.github.io/adapt/examples/Multi_fidelity.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Cc9TVY_Tl_boVzZDNisQnqe6Qx78svqe?usp=sharing)  | [<img src="https://raw.githubusercontent.com/adapt-python/adapt/a490a5c4cefb80d6222bc831a8cc25b2f65221ce/docs/_static/images/rotation_setup.png">](https://adapt-python.github.io/adapt/examples/Rotation.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XePW12UF80PKzvLu9cyRJKWQoZIxk_J2?usp=sharing) |


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

## Acknowledgement

This work has been funded by Michelin and the Industrial Data Analytics and Machine Learning chair from ENS Paris-Saclay, Borelli center.

[<img src="https://www.michelin.com/wp-content/themes/michelin/public/img/michelin-logo.svg" width=200px alt="Michelin">](https://www.michelin.com/) [<img src="https://www.centreborelli.fr/wp-content/uploads/2021/01/Logotype_IDAML.png" width=200px alt="IDAML">](https://www.centreborelli.fr/partenariats/chaires/chaires-industrielles-2/) [<img src="https://www.centreborelli.fr/wp-content/uploads/2020/07/logotype_centre_borelli_site_web.png" alt="Centre Borelli" width=150px>](https://www.centreborelli.fr)
