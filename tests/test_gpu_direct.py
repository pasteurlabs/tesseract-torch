# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of ``apply_tesseract`` over the ``cuda_ipc`` transport.

The ``served_gpu_tesseract`` fixture is created with ``gpu_transport="cuda_ipc"``,
so tests that call ``apply_tesseract`` without naming a transport also cover the
default picking it up. All tests are marked ``gpu``, and the fixture skips where
no CUDA GPU is available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract

from tesseract_torch import apply_tesseract

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("n", [1, 8, 1000, 100_003])
def test_apply_matches_analytic(served_gpu_tesseract, n):
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.ones(n, dtype=torch.float32, device="cuda") * 3.0
    out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})
    c = out["c"]
    assert c.is_cuda
    np.testing.assert_allclose(
        c.cpu().numpy(), a.cpu().numpy() * 2.0 + b.cpu().numpy(), rtol=1e-6, atol=0
    )


def test_apply_matches_host_path(served_gpu_tesseract):
    """The cuda_ipc path must match the host-copy path exactly.

    ``gpu_transport="none"`` overrides the transport the Tesseract was created
    with, in both directions, so the baseline comes back as a host tensor.
    """
    a = torch.linspace(-5, 5, 257, dtype=torch.float32, device="cuda")
    b = torch.linspace(10, -10, 257, dtype=torch.float32, device="cuda")

    ipc = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})["c"]
    host = apply_tesseract(
        served_gpu_tesseract, {"a": a, "b": b}, gpu_transport="none"
    )["c"]

    assert ipc.is_cuda
    assert not host.is_cuda
    np.testing.assert_array_equal(ipc.cpu().numpy(), host.numpy())


def test_explicit_transport_on_client_without_one(served_gpu_tesseract):
    """A plain ``from_url`` client advertises no transport but can name one."""
    a = torch.arange(8, dtype=torch.float32, device="cuda")
    b = torch.ones(8, dtype=torch.float32, device="cuda")

    with Tesseract.from_url(served_gpu_tesseract._client.url) as tess:
        assert tess.supported_gpu_transports == ()
        out = apply_tesseract(tess, {"a": a, "b": b}, gpu_transport="cuda_ipc")

    assert out["c"].is_cuda
    np.testing.assert_allclose(
        out["c"].cpu().numpy(), a.cpu().numpy() * 2.0 + 1.0, rtol=1e-6
    )


def test_grad_through_cuda_ipc(served_gpu_tesseract):
    """Reverse-mode AD dispatches through the same cuda_ipc path (vjp)."""
    n = 512
    a = torch.arange(n, dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.ones(n, dtype=torch.float32, device="cuda")

    out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})
    out["c"].sum().backward()

    assert a.grad.is_cuda
    # d/da sum(a*2 + b) = 2
    np.testing.assert_allclose(a.grad.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)


def test_vmap_through_cuda_ipc(served_gpu_tesseract):
    """``torch.vmap`` batches the cuda_ipc path, on-device outputs included."""
    a = torch.randn(4, 64, device="cuda", requires_grad=True)
    b = torch.ones(64, device="cuda")

    out = torch.vmap(
        lambda x: apply_tesseract(
            served_gpu_tesseract,
            {"a": x, "b": b},
            vmap_method="sequential",
        )
    )(a)
    expected = a.detach() * 2.0 + b

    assert out["c"].is_cuda
    torch.testing.assert_close(out["c"], expected)
    torch.testing.assert_close(out["c_sum"].cpu().reshape(4), expected.sum(dim=1).cpu())

    out["c"].sum().backward()
    assert a.grad.is_cuda
    torch.testing.assert_close(a.grad, torch.full_like(a, 2.0))


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
        out = apply_tesseract(served_gpu_tesseract, {"a": a_dual, "b": b_dual})
        _primal, tangent = fwAD.unpack_dual(out["c"])

    assert tangent.is_cuda
    # c = a*scale + b, scale=2 => dc = scale*da + db = 2*ta + tb = 2.
    np.testing.assert_allclose(tangent.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)


def test_jvp_with_nondiff_output(served_gpu_tesseract):
    """A non-differentiable output must not break the jvp path or force a host copy.

    ``c_sum`` is a non-differentiable output, so the jvp endpoint returns no
    tangent for it and it comes back as-is (a host array) rather than a dual
    tensor. Its presence must not disturb the differentiable output ``c``, whose
    tangent must still come back correct and on-device under ``cuda_ipc``.
    """
    import torch.autograd.forward_ad as fwAD

    n = 8
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = torch.ones(n, dtype=torch.float32, device="cuda")
    ta = torch.ones(n, dtype=torch.float32, device="cuda")
    tb = torch.zeros(n, dtype=torch.float32, device="cuda")

    with fwAD.dual_level():
        a_dual = fwAD.make_dual(a, ta)
        b_dual = fwAD.make_dual(b, tb)
        out = apply_tesseract(served_gpu_tesseract, {"a": a_dual, "b": b_dual})
        _primal, tangent = fwAD.unpack_dual(out["c"])

    assert tangent.is_cuda
    # c = a*scale + b, scale=2 => dc = scale*da + db = 2*ta + tb = 2.
    np.testing.assert_allclose(tangent.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)
    # The non-differentiable c_sum output still comes back, matching the analytic
    # sum(c) = sum(a*2 + b).
    expected_c_sum = (a * 2.0 + b).sum().item()
    np.testing.assert_allclose(
        np.asarray(out["c_sum"]).reshape(()), expected_c_sum, rtol=1e-6
    )


def test_serial_reuse(served_gpu_tesseract):
    """Back-to-back serial dispatches must not corrupt each other's results.

    The server releases the previously exported buffer at the start of each
    request, so repeated calls must each still return correct data.
    """
    for i in range(20):
        a = torch.full((512,), float(i), dtype=torch.float32, device="cuda")
        b = torch.full((512,), float(2 * i), dtype=torch.float32, device="cuda")
        out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})
        np.testing.assert_allclose(
            out["c"].cpu().numpy(),
            np.full((512,), i * 2.0 + 2 * i, dtype=np.float32),
            rtol=1e-6,
        )


def test_grad_with_nondiff_array_input(served_gpu_tesseract):
    """A non-differentiable array input must not force an unwanted host copy.

    ``mask`` is passed as a CUDA tensor but is not declared ``Differentiable``
    in the schema, so it is passed through the autograd function without a
    gradient; the gradient is requested only for ``a``. The real gradient wrt ``a`` must still come
    back correctly, on-device.
    """
    n = 8
    a = torch.arange(n, dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.ones(n, dtype=torch.float32, device="cuda")
    mask = torch.full((n,), 3.0, dtype=torch.float32, device="cuda")

    out = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b, "mask": mask})
    out["c"].sum().backward()

    assert a.grad.is_cuda
    # c = (a*scale + b)*mask, scale=2 => d/da sum(c) = 2*mask.
    np.testing.assert_allclose(
        a.grad.cpu().numpy(), np.full((n,), 2.0 * 3.0), rtol=1e-6
    )
