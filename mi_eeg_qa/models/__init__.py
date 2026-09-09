"""Decoders with predict_proba interfaces."""

from .csp_lda import CSPLDAClassifier
from .torch_models import EEGNet, EEGNetClassifier, ShallowConvNet, ShallowConvNetClassifier

__all__ = [
    "CSPLDAClassifier",
    "EEGNet",
    "EEGNetClassifier",
    "ShallowConvNet",
    "ShallowConvNetClassifier",
]
