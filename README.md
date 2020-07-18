# ADAPT

[![PyPI version](https://badge.fury.io/py/adaptation.svg)](https://badge.fury.io/py/adaptation)
[![Build Status](https://github.com/antoinedemathelin/adapt/workflows/build/badge.svg)](https://github.com/antoinedemathelin/adapt/actions)
[![Codecov Status](https://codecov.io/gh/antoinedemathelin/adapt/branch/master/graph/badge.svg?token=IWQXMYGY2Q)](https://codecov.io/gh/antoinedemathelin/adapt)

**A**wesome **D**omain **A**daptation **P**ackage **T**oolbox

ADAPT is a python library which provides several domain adaptation methods usefull to improve machine learning models.

## Documentation Website

Find the details of all implemented methods as well as illustrative examples on: 

## Installation

This package is available on [Pypi](https://badge.fury.io/py/adaptation) and can be installed with the following command line:

`pip install adaptation`

The following dependencies are required and will be installed with the library:
- `tensorflow` (>= 2.0)
- `scikit-learn`
- `cvxopt`

If for some reason, these packages failed to install, you can do it manually with:

`pip install tensorflow scikit-learn cvxopt`

Finally import the module in your python scripts with:

```python
import adapt
```

## Content

ADAPT package is divided in three sub-modules containing the following domain adaptation methods:

### Feature-based methods

- FE (*Frustratingly Easy Domain Adaptation*)
- mSDA
- DANN
- ADDA
- CORAL
- DeepCORAL

### Instance-based methods

- KMM
- KLIEP
- TrAdaBoost
- TrAdaBoostR2
- TwoStageTrAdaBoostR2

### Parameter-based methods

- RegularTransferLR
- RegularTransferLC
- RegularTransferNN


## Examples

Examples for regression and classification DA on synthetic datasets are available here:

Classification | Regression         
:-------------------------:|:-------------------------:
<img src="docs/source/_static/images/classification_setup.png" width="600px" height="300px"> | <img src="docs/source/_static/images/regression_setup.png" width="600px" height="300px">
