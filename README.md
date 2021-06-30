# ADAPT

[![PyPI version](https://badge.fury.io/py/adaptation.svg)](https://pypi.org/project/adaptation)
[![Build Status](https://github.com/adapt-python/adapt/workflows/build/badge.svg)](https://github.com/adapt-python/adapt/actions)
[![Python Version](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7-blue)](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7-blue)
[![Codecov Status](https://codecov.io/gh/adapt-python/adapt/branch/master/graph/badge.svg?token=IWQXMYGY2Q)](https://codecov.io/gh/adapt-python/adapt)

**A**wesome **D**omain **A**daptation **P**ackage **T**oolbox

ADAPT is a python library which provides several domain adaptation methods usefull to improve machine learning models.

## Documentation Website

Find the details of all implemented methods as well as illustrative examples here: [ADAPT Documentation Website](https://adapt-python.github.io/adapt/)

## Installation

This package is available on [Pypi](https://pypi.org/project/adaptation) and can be installed with the following command line:

`pip install adaptation`

The following dependencies are required and will be installed with the library:
- `numpy`
- `scipy`
- `tensorflow` (>= 2.0)
- `scikit-learn`
- `cvxopt`

If for some reason, these packages failed to install, you can do it manually with:

`pip install numpy scipy tensorflow scikit-learn cvxopt`

Finally import the module in your python scripts with:

```python
import adapt
```

## Content

ADAPT package is divided in three sub-modules containing the following domain adaptation methods:

### Feature-based methods

- [FE](https://adapt-python.github.io/adapt/generated/adapt.feature_based.FE.html) (*Frustratingly Easy Domain Adaptation*)
- [mSDA](https://adapt-python.github.io/adapt/generated/adapt.feature_based.mSDA.html) (*marginalized Stacked Denoising Autoencoder*)
- [DANN](https://adapt-python.github.io/adapt/generated/adapt.feature_based.DANN.html) (*Discriminative Adversarial Neural Network*)
- [ADDA](https://adapt-python.github.io/adapt/generated/adapt.feature_based.ADDA.html) (*Adversarial Discriminative Domain Adaptation*)
- [CORAL](https://adapt-python.github.io/adapt/generated/adapt.feature_based.CORAL.html) (*CORrelation ALignment*)
- [DeepCORAL](https://adapt-python.github.io/adapt/generated/adapt.feature_based.DeepCORAL.html) (*Deep CORrelation ALignment*)

### Instance-based methods

- [KMM](https://adapt-python.github.io/adapt/generated/adapt.instance_based.KMM.html) (*Kernel Mean Matching*)
- [KLIEP](https://adapt-python.github.io/adapt/generated/adapt.instance_based.KLIEP.html) (*Kullback–Leibler Importance Estimation Procedure*)
- [TrAdaBoost](https://adapt-python.github.io/adapt/generated/adapt.instance_based.TrAdaBoost.html) (*Transfer AdaBoost*)
- [TrAdaBoostR2](https://adapt-python.github.io/adapt/generated/adapt.instance_based.TrAdaBoostR2.html) (*Transfer AdaBoost for Regression*)
- [TwoStageTrAdaBoostR2](https://adapt-python.github.io/adapt/generated/adapt.instance_based.TwoStageTrAdaBoostR2.html) (*Two Stage Transfer AdaBoost for Regression*)

### Parameter-based methods

- [RegularTransferLR](https://adapt-python.github.io/adapt/generated/adapt.parameter_based.RegularTransferLR.html) (*Regular Transfer with Linear Regression*)
- [RegularTransferLC](https://adapt-python.github.io/adapt/generated/adapt.parameter_based.RegularTransferLC.html) (*Regular Transfer with Linear Classification*)
- [RegularTransferNN](https://adapt-python.github.io/adapt/generated/adapt.parameter_based.RegularTransferNN.html) (*Regular Transfer with Neural Network*)


## Examples

Examples for regression and classification DA on synthetic datasets are available here:

Classification | Regression         
:-------------------------:|:-------------------------:
[<img src="docs/_static/images/classification_setup.png" width="600px" height="350px">](https://adapt-python.github.io/adapt/classification_example.html) | [<img src="docs/_static/images/regression_setup.png" width="600px" height="300px">](https://adapt-python.github.io/adapt/regression_example.html)


## Acknowledgement

Part of this work has been funded by the Industrial Data Analytics and Machine Learning chair from ENS Paris-Saclay, Borelli center.
