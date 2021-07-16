"""
Feature-Based Methods Module
"""

from ._fe import FE
from ._coral import CORAL
from ._msda import mSDA
from ._deep import DANN, ADDA, DeepCORAL, MCD, MDD

__all__ = ["FE", "CORAL", "DeepCORAL", "ADDA", "DANN", "mSDA", "MCD", "MDD"]
