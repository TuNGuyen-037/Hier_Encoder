# losses/__init__.py
from .cross_entropy import SequenceCrossEntropyLoss
from .ranking_loss import BPRLoss

__all__ = [
    "SequenceCrossEntropyLoss",
    "BPRLoss"
]
