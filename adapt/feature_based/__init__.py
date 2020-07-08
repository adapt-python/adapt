"""
Feature-Based Methods Module
"""

from ._fe import FE
from ._coral import CORAL, DeepCORAL
from ._adda import ADDA
from ._dann import DANN
from ._msda import mSDA

__all__ = ["FE", "CORAL", "DeepCORAL", "ADDA", "DANN", "mSDA"]
