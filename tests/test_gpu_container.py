# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests against a Tesseract served from a Docker container.

The other GPU tests reach Tesseracts in-process and in a subprocess. Here the
Tesseract runs in a container, so tensors cross a container boundary. The tests
need Docker with the NVIDIA container runtime and are marked ``gpu``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract

from tesseract_torch import apply_tesseract

pytestmark = pytest.mark.gpu

here = Path(__file__).parent


@pytest.fixture(scope="module")
def gpu_container_image() -> str:
    """Build the GPU Tesseract into an image once."""
    from tesseract_core.sdk.engine import build_tesseract

    # The image name comes from the Tesseract's config, the tag from here
    build_tesseract(here / "gpu_tesseract", "test")
    return "tesseract-torch-gpu:test"


@pytest.mark.parametrize("client", ["served", "from_url"])
def test_container_round_trip_stays_on_device(gpu_container_image, client):
    """CUDA tensors cross the container boundary over cuda_ipc, host-copy free.

    The client that served the container requests cuda_ipc, and a ``from_url``
    client of it requests nothing and picks cuda_ipc once it checked that it
    works. The container computes with CuPy and fails a call whose inputs
    arrive in host memory or whose outputs would be copied to the host, so
    passing rules out a host copy on either side.
    """
    with Tesseract.from_image(
        gpu_container_image,
        gpus=["all"],
        gpu_transport="cuda_ipc",
        environment={
            "TESSERACT_TORCH_TEST_ARRAY_MODULE": "cupy",
            "TESSERACT_TORCH_TEST_REJECT_HOST_INPUTS": "1",
            "TESSERACT_FORBID_DEVICE_HOST_COPY": "1",
        },
    ) as served:
        tess = served if client == "served" else Tesseract.from_url(served._client.url)
        a = torch.arange(64, dtype=torch.float32, device="cuda", requires_grad=True)
        b = torch.ones(64, dtype=torch.float32, device="cuda")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = apply_tesseract(tess, {"a": a, "b": b})
        assert out["c"].is_cuda
        np.testing.assert_allclose(
            out["c"].detach().cpu().numpy(), 2 * np.arange(64) + 1.0
        )
        out["c"].sum().backward()
        assert a.grad.is_cuda
        np.testing.assert_allclose(a.grad.cpu().numpy(), np.full(64, 2.0))
