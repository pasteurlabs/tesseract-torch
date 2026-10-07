# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of ``apply_tesseract`` over the ``cuda_ipc`` transport.

The ``served_gpu_tesseract`` fixture is created with ``gpu_transport="cuda_ipc"``,
which ``apply_tesseract`` uses for CUDA tensors. The ``local_*`` tests load the
same Tesseract in-process, where one created with ``gpu_transport="cuda_ipc"``
hands its endpoints CUDA tensors as they are. All tests are marked ``gpu``, and
the fixtures skip where no CUDA GPU is available.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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

    A view requesting ``gpu_transport="none"`` overrides the transport the
    Tesseract was created with, in both directions. Which transport a call
    uses must not change its result, so both outputs land on the input's
    device, and the non-differentiable one is a NumPy array either way.
    """
    a = torch.linspace(-5, 5, 257, dtype=torch.float32, device="cuda")
    b = torch.linspace(10, -10, 257, dtype=torch.float32, device="cuda")

    ipc = apply_tesseract(served_gpu_tesseract, {"a": a, "b": b})
    host_view = served_gpu_tesseract.with_encoding(gpu_transport="none")
    host = apply_tesseract(host_view, {"a": a, "b": b})

    assert ipc["c"].is_cuda and host["c"].is_cuda
    np.testing.assert_array_equal(ipc["c"].cpu().numpy(), host["c"].cpu().numpy())
    assert isinstance(ipc["c_sum"], np.ndarray)
    assert isinstance(host["c_sum"], np.ndarray)
    np.testing.assert_array_equal(ipc["c_sum"], host["c_sum"])


def test_from_url_client_uses_cuda_ipc_once_checked(served_gpu_tesseract):
    """A plain ``from_url`` client requests no transport, but uses cuda_ipc once it checked it works."""
    a = torch.arange(8, dtype=torch.float32, device="cuda")
    b = torch.ones(8, dtype=torch.float32, device="cuda")

    with Tesseract.from_url(served_gpu_tesseract._client.url) as tess:
        assert tess.current_encoding.gpu_transport is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert tess.resolve_gpu_transport() == "cuda_ipc"
        out = apply_tesseract(tess, {"a": a, "b": b})

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


@pytest.mark.parametrize(
    ("tess_fixture", "received_type"),
    [
        ("local_gpu_tesseract", np.ndarray),
        ("local_cuda_ipc_gpu_tesseract", torch.Tensor),
    ],
)
def test_local_client_receives_cuda_tensors_if_created_with_transport(
    request, monkeypatch, tess_fixture, received_type
):
    """An in-process client gets CUDA tensors as-is only if created with a transport.

    Otherwise its endpoint sees host NumPy arrays. Either way the endpoint
    computes on the GPU, so the output alone cannot tell the two apart, and the
    payload the client receives is checked instead.
    """
    local_gpu_tesseract = request.getfixturevalue(tess_fixture)
    client = local_gpu_tesseract._client
    received = []
    run_tesseract = client.run_tesseract

    def recording_run_tesseract(endpoint, payload=None, *args, **kwargs):
        if endpoint == "apply":
            received.append(payload["inputs"]["a"])
        return run_tesseract(endpoint, payload, *args, **kwargs)

    monkeypatch.setattr(client, "run_tesseract", recording_run_tesseract)

    a = torch.arange(8, dtype=torch.float32, device="cuda")
    b = torch.ones(8, dtype=torch.float32, device="cuda")
    out = apply_tesseract(local_gpu_tesseract, {"a": a, "b": b})

    assert out["c"].is_cuda
    assert isinstance(received[0], received_type)
    if received_type is torch.Tensor:
        assert received[0].is_cuda
    np.testing.assert_allclose(
        out["c"].cpu().numpy(), a.cpu().numpy() * 2.0 + 1.0, rtol=1e-6
    )


def test_local_client_derivatives_with_transport(local_cuda_ipc_gpu_tesseract):
    """Vjp and jvp of an in-process client created with a transport stay on-device."""
    local_gpu_tesseract = local_cuda_ipc_gpu_tesseract
    import torch.autograd.forward_ad as fwAD

    n = 64
    a = torch.arange(n, dtype=torch.float32, device="cuda", requires_grad=True)
    b = torch.ones(n, dtype=torch.float32, device="cuda")

    out = apply_tesseract(local_gpu_tesseract, {"a": a, "b": b})
    out["c"].sum().backward()
    assert a.grad.is_cuda
    np.testing.assert_allclose(a.grad.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)

    with fwAD.dual_level():
        a_dual = fwAD.make_dual(a.detach(), torch.ones_like(b))
        b_dual = fwAD.make_dual(b, torch.zeros_like(b))
        out = apply_tesseract(local_gpu_tesseract, {"a": a_dual, "b": b_dual})
        _primal, tangent = fwAD.unpack_dual(out["c"])
    assert tangent.is_cuda
    np.testing.assert_allclose(tangent.cpu().numpy(), np.full((n,), 2.0), rtol=1e-6)


def test_falls_back_to_host_when_cuda_ipc_does_not_work(monkeypatch):
    """A Tesseract offering cuda_ipc that does not work gets host copies, with a warning.

    Hiding the GPU from the server stands in for every reason cuda_ipc can fail,
    e.g. a server on another host. A ``from_url`` client, which requests no
    transport, falls back to host copies and still returns its outputs on the
    input's device, while the client that requested cuda_ipc gets an error
    instead of a silent host copy.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")
    api_path = Path(__file__).parent / "vectoradd_tesseract" / "tesseract_api.py"
    # Only the server is spawned without the GPU. The test process reads
    # CUDA_VISIBLE_DEVICES when it first touches CUDA, so restore it right away.
    with monkeypatch.context() as m:
        m.setenv("CUDA_VISIBLE_DEVICES", "")
        served = Tesseract.from_source(
            api_path, python_executable=sys.executable, gpu_transport="cuda_ipc"
        )
        served.serve()
    try:
        remote = Tesseract.from_url(served._client.url)
        a = torch.arange(8, dtype=torch.float32, device="cuda", requires_grad=True)
        b = torch.ones(8, dtype=torch.float32, device="cuda")
        with pytest.warns(UserWarning, match="copied to the host instead"):
            out = apply_tesseract(remote, {"a": a, "b": b})
        assert out["c"].is_cuda
        np.testing.assert_allclose(out["c"].detach().cpu().numpy(), np.arange(8) + 1.0)
        out["c"].sum().backward()
        assert a.grad.is_cuda
        np.testing.assert_allclose(a.grad.cpu().numpy(), np.ones(8))

        with pytest.raises(RuntimeError, match="does not work between"):
            apply_tesseract(served, {"a": a, "b": b})
    finally:
        served.teardown()
