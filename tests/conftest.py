# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
from tesseract_core import Tesseract

here = Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


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


@pytest.fixture(scope="module")
def list_tess() -> Tesseract:
    """List-valued differentiable field, one coefficient per position."""
    return Tesseract.from_tesseract_api(here / "list_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="module")
def dict_key_tess() -> Tesseract:
    """Dict-valued differentiable field, for keys that are not plain words."""
    return Tesseract.from_tesseract_api(
        here / "dict_key_tesseract" / "tesseract_api.py"
    )


# ---------------------------------------------------------------------------
# GPU (cuda_ipc) serving
# ---------------------------------------------------------------------------
#
# Cross-process CUDA IPC needs the Tesseract (producer) and the test process
# (consumer) to be *separate* processes sharing the GPU, because a process
# cannot open an IPC handle it exported itself. ``from_source`` runs the
# Tesseract in a subprocess on this interpreter, which shares the GPU and IPC
# namespace with the test process without needing Docker's ``--ipc=host``.


@pytest.fixture(scope="module")
def served_gpu_tesseract():
    """A served GPU Tesseract with cuda_ipc enabled. Skips without a CUDA GPU."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")

    with Tesseract.from_source(
        here / "gpu_tesseract" / "tesseract_api.py",
        python_executable=sys.executable,
        gpu_transport="cuda_ipc",
    ) as tess:
        yield tess
