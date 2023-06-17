"""
Feature-Based Methods Module
"""

from ._fa import FA
from ._coral import CORAL
from ._dann import DANN
from ._adda import ADDA
from ._deepcoral import DeepCORAL
from ._mcd import MCD
from ._mdd import MDD
from ._wdgrl import WDGRL
from ._cdan import CDAN
from ._sa import SA
from ._fmmd import fMMD
from ._ccsa import CCSA
from ._tca import TCA
from ._pred import PRED

__all__ = ["FA", "CORAL", "DeepCORAL", "ADDA", "DANN",
           "MCD", "MDD", "WDGRL", "CDAN", "SA", "fMMD", "CCSA", "TCA", "PRED"]
