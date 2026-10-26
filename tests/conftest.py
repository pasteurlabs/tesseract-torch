# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from tesseract_core import Tesseract

here = Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


# ---------------------------------------------------------------------------
# GPU requirements
# ---------------------------------------------------------------------------
#
# The GPU tests skip without a CUDA GPU, and the ones against the CuPy producer
# also without CuPy, unless TESSERACT_TORCH_GPU_REQUIRED is set (as in the GPU
# CI job), in which case they fail instead.

GPU_PRODUCERS = ["torch", "cupy"]


def _skip_or_fail(reason: str) -> None:
    if os.environ.get("TESSERACT_TORCH_GPU_REQUIRED"):
        pytest.fail(f"{reason}, but TESSERACT_TORCH_GPU_REQUIRED is set")
    pytest.skip(reason)


def _require_gpu(producer: str = "torch") -> None:
    import torch

    if not torch.cuda.is_available():
        _skip_or_fail("no CUDA GPU available")
    if producer == "cupy" and importlib.util.find_spec("cupy") is None:
        _skip_or_fail("CuPy is not installed")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # Before any fixture is set up, so that none of them needs its own check
    if item.get_closest_marker("gpu") is not None:
        _require_gpu()


@contextlib.contextmanager
def _env(**env: str) -> Iterator[None]:
    """Set environment variables for the duration of the block."""
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        yield


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
def batched_tess() -> Tesseract:
    """Like ``nonlinear_tess``, but every array field accepts a batch dimension."""
    return Tesseract.from_tesseract_api(here / "batched_tesseract" / "tesseract_api.py")


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
# The GPU Tesseract
# ---------------------------------------------------------------------------
#
# ``gpu_tesseract`` reads TESSERACT_TORCH_TEST_ARRAY_MODULE when it is loaded.
# In process it computes with PyTorch, and served with each of ``GPU_PRODUCERS``.
#
# Cross-process CUDA IPC needs the Tesseract (producer) and the test process
# (consumer) to be *separate* processes sharing the GPU, because a process
# cannot open an IPC handle it exported itself. ``from_source`` runs the
# Tesseract in a subprocess on this interpreter, which shares the GPU and IPC
# namespace with the test process without needing Docker's ``--ipc=host``.

GPU_TESSERACT_API = here / "gpu_tesseract" / "tesseract_api.py"


@pytest.fixture(scope="module")
def local_gpu_tesseract():
    """The GPU Tesseract computing with PyTorch, loaded in-process."""
    with _env(TESSERACT_TORCH_TEST_ARRAY_MODULE="torch"):
        return Tesseract.from_tesseract_api(GPU_TESSERACT_API)


@pytest.fixture(scope="module")
def local_cuda_ipc_gpu_tesseract():
    """The GPU Tesseract in-process, created to take CUDA tensors as they are."""
    with _env(TESSERACT_TORCH_TEST_ARRAY_MODULE="torch"):
        return Tesseract.from_tesseract_api(GPU_TESSERACT_API, gpu_transport="cuda_ipc")


def _serve_gpu_tesseract(producer: str, *, allow_host_copies: bool = False):
    """Serve the GPU Tesseract with cuda_ipc, computing with ``producer``.

    Unless ``allow_host_copies`` is set, the served Tesseract fails a call whose
    inputs arrive in host memory or whose GPU outputs would be copied to the
    host.
    """
    _require_gpu(producer)
    env = {"TESSERACT_TORCH_TEST_ARRAY_MODULE": producer}
    if not allow_host_copies:
        env["TESSERACT_TORCH_TEST_REJECT_HOST_INPUTS"] = "1"
        env["TESSERACT_FORBID_DEVICE_HOST_COPY"] = "1"
    with contextlib.ExitStack() as stack:
        # Set only while spawning the server, which inherits them, so the test
        # process itself is unaffected
        with _env(**env):
            tess = stack.enter_context(
                Tesseract.from_source(
                    GPU_TESSERACT_API,
                    python_executable=sys.executable,
                    gpu_transport="cuda_ipc",
                )
            )
        yield tess


@pytest.fixture(scope="module")
def served_torch_gpu_tesseract():
    """The GPU Tesseract computing with PyTorch, served with cuda_ipc."""
    yield from _serve_gpu_tesseract("torch")


@pytest.fixture(scope="module")
def served_cupy_gpu_tesseract():
    """The GPU Tesseract computing with CuPy, served with cuda_ipc."""
    yield from _serve_gpu_tesseract("cupy")


@pytest.fixture(scope="module", params=GPU_PRODUCERS)
def served_gpu_tesseract(request):
    """The GPU Tesseract served with cuda_ipc, once per GPU producer."""
    return request.getfixturevalue(f"served_{request.param}_gpu_tesseract")


@pytest.fixture(scope="module", params=GPU_PRODUCERS)
def served_gpu_tesseract_allowing_host_copies(request):
    """Like ``served_gpu_tesseract``, for calls that copy through the host."""
    yield from _serve_gpu_tesseract(request.param, allow_host_copies=True)
