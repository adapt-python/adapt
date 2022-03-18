"""
Parameter-Based Methods Module
"""

from ._regular import RegularTransferLR, RegularTransferLC, RegularTransferNN
from ._finetuning import FineTuning
from ._transfer_tree import TransferTreeClassifier

__all__ = ["RegularTransferLR",
           "RegularTransferLC",
           "RegularTransferNN",
           "FineTuning",
           "TransferTreeClassifier"]