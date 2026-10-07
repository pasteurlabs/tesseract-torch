# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests against a Tesseract served from a Docker container.

The other GPU tests reach Tesseracts in-process and in a subprocess. Here the
Tesseract runs in a container, so tensors cross a container boundary, and a
container that cannot see the GPU stands in for a server that shares no GPU with
the client, such as one on another host. The tests need Docker with the NVIDIA
container runtime and are marked ``gpu``.
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
    """Build the container test Tesseract once. Skips without a CUDA GPU."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")
    from tesseract_core.sdk.engine import build_tesseract

    # The image name comes from the Tesseract's config, the tag from here
    build_tesseract(here / "gpu_container_tesseract", "test")
    return "tesseract-torch-gpu-container:test"


@pytest.mark.parametrize("client", ["served", "from_url"])
def test_container_round_trip_stays_on_device(gpu_container_image, client):
    """CUDA tensors cross the container boundary over cuda_ipc, host-copy free.

    The client that served the container requests cuda_ipc, and a ``from_url``
    client of it requests nothing and picks cuda_ipc once it checked that it
    works. The container forbids host copies of GPU arrays, so passing rules out
    a host copy on its side.
    """
    with Tesseract.from_image(
        gpu_container_image,
        gpus=["all"],
        gpu_transport="cuda_ipc",
        environment={"TESSERACT_FORBID_DEVICE_HOST_COPY": "1"},
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


def test_container_without_a_shared_gpu_gets_host_copies(gpu_container_image):
    """A container offering cuda_ipc that it cannot use leads to host copies.

    A ``from_url`` client, which requests no transport, falls back to host
    copies with a warning and still returns its outputs on the input's device.
    The client that requested cuda_ipc gets an error instead of a silent host
    copy.
    """
    with Tesseract.from_image(
        gpu_container_image,
        gpus=["all"],
        gpu_transport="cuda_ipc",
        environment={"CUDA_VISIBLE_DEVICES": ""},
    ) as served:
        remote = Tesseract.from_url(served._client.url)
        a = torch.arange(8, dtype=torch.float32, device="cuda", requires_grad=True)
        b = torch.ones(8, dtype=torch.float32, device="cuda")
        with pytest.warns(UserWarning, match="copied to the host instead"):
            out = apply_tesseract(remote, {"a": a, "b": b})
        assert out["c"].is_cuda
        np.testing.assert_allclose(
            out["c"].detach().cpu().numpy(), 2 * np.arange(8) + 1.0
        )
        out["c"].sum().backward()
        assert a.grad.is_cuda
        np.testing.assert_allclose(a.grad.cpu().numpy(), np.full(8, 2.0))

        with pytest.raises(RuntimeError, match="does not work between"):
            apply_tesseract(served, {"a": a, "b": b})
