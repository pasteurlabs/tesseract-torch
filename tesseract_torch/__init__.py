# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch compatibility layer for Tesseract.

Provides :func:`apply_tesseract`, which wraps any Tesseract as a differentiable
PyTorch operation supporting both reverse-mode (``.backward()``) and forward-mode
(``torch.autograd.forward_ad``) automatic differentiation.
"""

try:
    from ._version import __version__ as scm_version
except ImportError:
    import warnings

    warnings.warn(
        "Unable to import version information from _version.py. "
        "This is likely due to the package not being installed. "
        "Using default version '0.0.0+unknown'.",
        ImportWarning,
        stacklevel=1,
    )
    scm_version = "0.0.0+unknown"

from .function import apply_tesseract

__version__ = scm_version

__all__ = ["apply_tesseract"]
