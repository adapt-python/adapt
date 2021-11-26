"""
Feature-Based Methods Module
"""

from ._fe import FE
from ._coral import CORAL
from ._msda import mSDA
from ._deep import BaseDeepFeature
from ._dann import DANN
from ._adda import ADDA
from ._deepcoral import DeepCORAL
from ._mcd import MCD
from ._mdd import MDD
from ._wdgrl import WDGRL
from ._cdan import CDAN

__all__ = ["FE", "CORAL", "DeepCORAL", "ADDA", "DANN", "mSDA", "MCD", "MDD", "WDGRL", "BaseDeepFeature", "CDAN"]
