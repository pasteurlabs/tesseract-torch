# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from tesseract_core import Tesseract

here = Path(__file__).parent


@pytest.fixture(scope="module")
def vectoradd_tess() -> Tesseract:
    """Load the simple vector-addition test Tesseract."""
    return Tesseract.from_tesseract_api(
        here / "vectoradd_tesseract" / "tesseract_api.py"
    )


@pytest.fixture(scope="module")
def nonlinear_tess() -> Tesseract:
    """Cubic with a cross term, so the Jacobian depends on the input point."""
    return Tesseract.from_tesseract_api(
        here / "nonlinear_tesseract" / "tesseract_api.py"
    )


@pytest.fixture(scope="module")
def nested_tess() -> Tesseract:
    """Load the nested-schema test Tesseract."""
    return Tesseract.from_tesseract_api(here / "nested_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="module")
def forwardonly_tess() -> Tesseract:
    """Load a Tesseract that only has an apply endpoint (no JVP/VJP)."""
    return Tesseract.from_tesseract_api(
        here / "forwardonly_tesseract" / "tesseract_api.py"
    )


@pytest.fixture(scope="module")
def dict_tess() -> Tesseract:
    """Tesseract with dict-valued differentiable fields on both sides."""
    return Tesseract.from_tesseract_api(here / "dict_tesseract" / "tesseract_api.py")
