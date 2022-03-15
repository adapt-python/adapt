"""
Parameter-Based Methods Module
"""

from ._regular import RegularTransferLR, RegularTransferLC, RegularTransferNN
from ._finetuning import FineTuning

__all__ = ["RegularTransferLR", "RegularTransferLC", "RegularTransferNN", "FineTuning"]