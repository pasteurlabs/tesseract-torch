# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-direct dispatch tests: ``apply_tesseract(..., cuda_ipc=True)`` end to end.

With ``cuda_ipc=True``, a served (HTTP) Tesseract exchanges CUDA tensors via
CUDA IPC handles instead of a host round-trip, keeping data on the device.

These require a real GPU and a served (subprocess) GPU Tesseract, since CUDA
IPC is cross-process and cannot be self-opened. Marked ``gpu``; the
``served_gpu_tesseract`` fixture skips where no CUDA GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tesseract_torch import apply_tesseract

pytestmark = pytest.mark.gpu


def test_apply_matches_analytic(served_gpu_tesseract):
    n = 512
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.ones(n, dtype=torch.float32, device="cuda")
    out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b}, cuda_ipc=True)
    c = out["c"]
    assert c.is_cuda
    np.testing.assert_allclose(
        c.cpu().numpy(), a.cpu().numpy() * 2.0 + b.cpu().numpy(), rtol=1e-6, atol=0
    )


def test_apply_matches_default_host_path(served_gpu_tesseract):
    """The cuda_ipc path must match the default (host-copy) path exactly."""
    a = torch.linspace(-5, 5, 257, dtype=torch.float32, device="cuda")
    b = torch.linspace(10, -10, 257, dtype=torch.float32, device="cuda")

    ipc = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b}, cuda_ipc=True)["c"]
    default = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})["c"]

    assert ipc.is_cuda
    np.testing.assert_array_equal(ipc.cpu().numpy(), default.cpu().numpy())


def test_grad_through_cuda_ipc(served_gpu_tesseract):
    """Reverse-mode AD dispatches through the same cuda_ipc path (vjp)."""
    n = 512
    a = torch.arange(n, dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.ones(n, dtype=torch.float32, device="cuda")

    out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b}, cuda_ipc=True)
    out["c"].sum().backward()

    assert a.grad.is_cuda
    # d/da sum(a*2 + b) = 2
    np.testing.assert_allclose(a.grad.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)


def test_jvp_through_cuda_ipc(served_gpu_tesseract):
    """Forward-mode AD dispatches through the same cuda_ipc path (jvp)."""
    import torch.autograd.forward_ad as fwAD

    n = 512
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.ones(n, dtype=torch.float32, device="cuda")
    ta = torch.ones(n, dtype=torch.float32, device="cuda")
    tb = torch.zeros(n, dtype=torch.float32, device="cuda")

    with fwAD.dual_level():
        a_dual = fwAD.make_dual(a, ta)
        b_dual = fwAD.make_dual(b, tb)
        out = apply_tesseract(
            served_gpu_tesseract, {"a": a_dual, "b": b_dual}, cuda_ipc=True
        )
        _primal, tangent = fwAD.unpack_dual(out["c"])

    assert tangent.is_cuda
    # c = a*scale + b, scale=2 => dc = scale*da + db = 2*ta + tb = 2.
    np.testing.assert_allclose(tangent.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)


def test_serial_reuse(served_gpu_tesseract):
    """Back-to-back serial dispatches must not corrupt each other's results.

    The server releases the previously exported buffer at the start of each
    request, so repeated calls must each still return correct data.
    """
    for i in range(20):
        a = torch.full((512,), float(i), dtype=torch.float32, device="cuda")
        b = torch.full((512,), float(2 * i), dtype=torch.float32, device="cuda")
        out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b}, cuda_ipc=True)
        np.testing.assert_allclose(
            out["c"].cpu().numpy(),
            np.full((512,), i * 2.0 + 2 * i, dtype=np.float32),
            rtol=1e-6,
        )


def test_grad_with_nondiff_array_input(served_gpu_tesseract):
    """A non-differentiable array input must not force an unwanted host copy.

    ``mask`` is passed as a CUDA tensor but is not declared ``Differentiable``
    in the schema, so it is routed as a static input; the gradient is
    requested only for ``a``. The real gradient wrt ``a`` must still come
    back correctly, on-device.
    """
    n = 8
    a = torch.arange(n, dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.ones(n, dtype=torch.float32, device="cuda")
    mask = torch.full((n,), 3.0, dtype=torch.float32, device="cuda")

    out = apply_tesseract(
        served_gpu_tesseract, {"a": a, "b": b, "mask": mask}, cuda_ipc=True
    )
    out["c"].sum().backward()

    assert a.grad.is_cuda
    # c = (a*scale + b)*mask, scale=2 => d/da sum(c) = 2*mask.
    np.testing.assert_allclose(
        a.grad.cpu().numpy(), np.full((n,), 2.0 * 3.0), rtol=1e-6
    )
