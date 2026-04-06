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
def nested_tess() -> Tesseract:
    """Load the nested-schema test Tesseract."""
    return Tesseract.from_tesseract_api(here / "nested_tesseract" / "tesseract_api.py")
