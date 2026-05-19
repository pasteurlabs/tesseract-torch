# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch compatibility layer for Tesseract.

Provides :func:`apply_tesseract`, which wraps any Tesseract as a differentiable
PyTorch operation supporting both reverse-mode (``.backward()``) and forward-mode
(``torch.autograd.forward_ad``) automatic differentiation.
"""

from . import _version
from .function import apply_tesseract

__version__ = _version.get_versions()["version"]
del _version

__all__ = ["apply_tesseract"]
