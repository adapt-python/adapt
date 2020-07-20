# [ADAPT](https://antoinedemathelin.github.io/adapt/_build/html/index.html)

[![PyPI version](https://badge.fury.io/py/adaptation.svg)](https://badge.fury.io/py/adaptation)
[![Build Status](https://github.com/antoinedemathelin/adapt/workflows/build/badge.svg)](https://github.com/antoinedemathelin/adapt/actions)
[![Codecov Status](https://codecov.io/gh/antoinedemathelin/adapt/branch/master/graph/badge.svg?token=IWQXMYGY2Q)](https://codecov.io/gh/antoinedemathelin/adapt)

**A**wesome **D**omain **A**daptation **P**ackage **T**oolbox

ADAPT is a python library which provides several domain adaptation methods usefull to improve machine learning models.

## Documentation Website

Find the details of all implemented methods as well as illustrative examples here: [ADAPT Documentation Website](https://antoinedemathelin.github.io/adapt/_build/html/index.html)

## Installation

This package is available on [Pypi](https://badge.fury.io/py/adaptation) and can be installed with the following command line:

`pip install adaptation`

The following dependencies are required and will be installed with the library:
- `numpy`
- `scipy`
- `tensorflow` (>= 2.0)
- `scikit-learn`

If for some reason, these packages failed to install, you can do it manually with:

`pip install numpy scipy tensorflow scikit-learn`

Finally import the module in your python scripts with:

```python
import adapt
```

## Content

ADAPT package is divided in three sub-modules containing the following domain adaptation methods:

### Feature-based methods

- [FE](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.feature_based.FE.html) (*Frustratingly Easy Domain Adaptation*)
- [mSDA](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.feature_based.mSDA.html) (*marginalized Stacked Denoising Autoencoder*)
- [DANN](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.feature_based.DANN.html) (*Discriminative Adversarial Neural Network*)
- [ADDA](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.feature_based.ADDA.html) (*Adversarial Discriminative Domain Adaptation*)
- [CORAL](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.feature_based.CORAL.html) (*CORrelation ALignment*)
- [DeepCORAL](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.feature_based.DeepCORAL.html) (*Deep CORrelation ALignment*)

### Instance-based methods

- [KMM](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.instance_based.KMM.html) (*Kernel Mean Matching*)
- [KLIEP](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.instance_based.KLIEP.html) (*Kullback–Leibler Importance Estimation Procedure*)
- [TrAdaBoost](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.instance_based.TrAdaBoost.html) (*Transfer AdaBoost*)
- [TrAdaBoostR2](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.instance_based.TrAdaBoostR2.html) (*Transfer AdaBoost for Regression*)
- [TwoStageTrAdaBoostR2](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.instance_based.TwoStageTrAdaBoostR2.html) (*Two Stage Transfer AdaBoost for Regression*)

### Parameter-based methods

- [RegularTransferLR](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.parameter_based.RegularTransferLR.html) (*Regular Transfer with Linear Regression*)
- [RegularTransferLC](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.parameter_based.RegularTransferLC.html) (*Regular Transfer with Linear Classification*)
- [RegularTransferNN](https://antoinedemathelin.github.io/adapt/_build/html/generated/adapt.parameter_based.RegularTransferNN.html) (*Regular Transfer with Neural Network*)


## Examples

Examples for regression and classification DA on synthetic datasets are available here:

Classification | Regression         
:-------------------------:|:-------------------------:
[<img src="docs/_build/_static/images/classification_setup.png" width="600px" height="350px">](https://antoinedemathelin.github.io/adapt/_build/html/classification_example.html) | [<img src="docs/_build/_static/images/regression_setup.png" width="600px" height="300px">](https://antoinedemathelin.github.io/adapt/_build/html/regression_example.html)
