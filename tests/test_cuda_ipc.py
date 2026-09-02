# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CUDA IPC fast path (``apply_tesseract(..., cuda_ipc=True)``).

These exercise the pure-Python plumbing (the ``_cuda_ipc_mode`` client toggle
and the ``_to_tensor`` decode gate) without needing a GPU or a served
Tesseract. Full end-to-end coverage (a real ``HTTPClient`` talking CUDA IPC to
a GPU container) lives in tesseract-core's own end-to-end test suite; here we
only need to verify tesseract-torch drives that API correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract
from tesseract_core.sdk.tesseract import HTTPClient

from tesseract_torch import apply_tesseract
from tesseract_torch.function import (
    _cuda_ipc_mode,
    _supports_cuda_ipc,
    _tensor_to_numpy_or_cuda,
    _to_tensor,
)


def _fake_tesseract(client: object | None) -> Tesseract:
    """A ``Tesseract`` instance with an arbitrary (possibly fake) ``_client``.

    Bypasses ``__init__`` (which warns that direct construction is deprecated
    and always builds a real ``HTTPClient``) since these tests only need a
    real ``Tesseract`` *instance* -- for the typeguard-checked
    ``_cuda_ipc_mode(tesseract: Tesseract)`` annotation -- with a specific,
    possibly non-``HTTPClient``, ``_client`` swapped in.
    """
    tess = object.__new__(Tesseract)
    tess._client = client
    return tess


# ---------------------------------------------------------------------------
# _to_tensor: the DLPack decode gate must not swallow plain NumPy arrays
# ---------------------------------------------------------------------------


def test_to_tensor_numpy_roundtrip():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = _to_tensor(a)
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_readonly_numpy_array():
    """A read-only NumPy array must go through the copy fallback, not DLPack.

    Plain ``np.ndarray`` also implements ``__dlpack__`` (since NumPy 1.22), so
    gating the DLPack branch on that attribute alone would misroute host
    arrays -- and DLPack has no way to signal read-only to an older consumer,
    so a read-only array would fail outright. The gate must instead be
    ``__cuda_array_interface__``, which only CUDA arrays expose.
    """
    a = np.array([1.0, 2.0], dtype=np.float32)
    a.flags.writeable = False
    t = _to_tensor(a)
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([1.0, 2.0]))


def test_to_tensor_tensor_passthrough():
    """Already-a-tensor inputs (e.g. the JVP no-tangent shortcut) pass through."""
    t = torch.tensor([1.0, 2.0])
    assert _to_tensor(t) is t


def test_to_tensor_adopts_cuda_array_interface_via_dlpack():
    """An object exposing ``__cuda_array_interface__`` decodes via DLPack.

    Mimics ``tesseract_core.runtime.cuda_ipc.IpcDeviceArray`` without needing
    a real GPU: any object with that attribute (plus ``__dlpack__``) must be
    routed through DLPack rather than ``np.asarray``.
    """

    class FakeIpcDeviceArray:
        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor
            # Real IpcDeviceArray exposes this; only its presence is checked.
            self.__cuda_array_interface__ = {
                "shape": tuple(tensor.shape),
                "typestr": "<f4",
                "data": (0, False),
                "strides": None,
                "version": 3,
            }

        def __dlpack__(self, *args: object, **kwargs: object) -> object:
            return torch.utils.dlpack.to_dlpack(self._tensor)

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

    fake = FakeIpcDeviceArray(torch.tensor([9.0, 8.0]))
    t = _to_tensor(fake)
    assert isinstance(t, torch.Tensor)
    assert torch.allclose(t, torch.tensor([9.0, 8.0]))


# ---------------------------------------------------------------------------
# _tensor_to_numpy_or_cuda: CPU tensors still go through NumPy
# ---------------------------------------------------------------------------


def test_tensor_to_numpy_or_cuda_cpu_tensor_returns_numpy():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = _tensor_to_numpy_or_cuda(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_tensor_to_numpy_or_cuda_cpu_tensor_returns_numpy_even_with_cuda_ipc():
    """``cuda_ipc=True`` only changes behavior for CUDA tensors."""
    t = torch.tensor([1.0, 2.0, 3.0])
    out = _tensor_to_numpy_or_cuda(t, cuda_ipc=True)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_to_numpy_or_cuda_cuda_tensor_default_is_host_copy():
    """Without ``cuda_ipc=True``, a CUDA tensor still takes the host round-trip.

    The default (non-``cuda_ipc``) client encoder calls ``np.asanyarray`` on
    whatever it's handed, which cannot read GPU memory -- so passing a raw
    CUDA tensor through here when cuda_ipc is off would break the client, not
    speed it up.
    """
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    out = _tensor_to_numpy_or_cuda(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_to_numpy_or_cuda_cuda_tensor_with_cuda_ipc_passes_through():
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    out = _tensor_to_numpy_or_cuda(t, cuda_ipc=True)
    assert isinstance(out, torch.Tensor)
    assert out.is_cuda
    assert out.is_contiguous()


def test_tensor_to_numpy_or_cuda_rejects_functional_tensors():
    """torch.func transforms wrap tensors without backing storage."""
    from torch import func

    def f(x: torch.Tensor) -> np.ndarray:
        return _tensor_to_numpy_or_cuda(x)

    with pytest.raises(RuntimeError, match=r"torch\.func transforms"):
        func.grad(f)(torch.tensor(1.0))


# ---------------------------------------------------------------------------
# _supports_cuda_ipc: which clients cuda_ipc mode can actually apply to
# ---------------------------------------------------------------------------


def _fake_http_client(output_format: str = "json+base64") -> HTTPClient:
    """A real ``HTTPClient`` that never talks to the network in these tests.

    ``HTTPClient.__init__`` only sets attributes and opens a ``requests.Session``
    (no connection), so this is cheap and side-effect-free.
    """
    return HTTPClient("http://fake-tesseract.invalid", output_format=output_format)


def test_supports_cuda_ipc_true_for_http_client():
    tess = _fake_tesseract(client=_fake_http_client())
    assert _supports_cuda_ipc(tess) is True


def test_supports_cuda_ipc_false_for_local_client_shaped_object():
    """A client without ``_output_format`` (e.g. LocalClient) is unsupported."""
    tess = _fake_tesseract(client=object())
    assert _supports_cuda_ipc(tess) is False


def test_supports_cuda_ipc_false_when_no_client():
    tess = _fake_tesseract(client=None)
    assert _supports_cuda_ipc(tess) is False


# ---------------------------------------------------------------------------
# _cuda_ipc_mode: the HTTPClient toggle itself (assumes the caller already
# checked _supports_cuda_ipc)
# ---------------------------------------------------------------------------


def test_cuda_ipc_mode_toggles_and_restores_http_client():
    """Verify the toggle restores the default ``Accept`` header.

    A fresh ``HTTPClient``'s ``requests.Session`` always has a default
    ``Accept: */*`` header, so the "prior" value the toggle restores is that
    default, not an absent header.
    """
    tess = _fake_tesseract(client=_fake_http_client("json+base64"))
    prior_accept = tess._client._session.headers["Accept"]
    with _cuda_ipc_mode(tess):
        assert tess._client._output_format == "json+cuda_ipc"
        assert tess._client._session.headers["Accept"] == "application/json+cuda_ipc"
    assert tess._client._output_format == "json+base64"
    assert tess._client._session.headers["Accept"] == prior_accept


def test_cuda_ipc_mode_preserves_prior_accept_header():
    client = _fake_http_client("json+base64")
    client._session.headers["Accept"] = "application/json+base64"
    tess = _fake_tesseract(client=client)
    with _cuda_ipc_mode(tess):
        assert client._session.headers["Accept"] == "application/json+cuda_ipc"
    assert client._session.headers["Accept"] == "application/json+base64"


def test_cuda_ipc_mode_restores_on_exception():
    tess = _fake_tesseract(client=_fake_http_client("json+base64"))
    prior_accept = tess._client._session.headers["Accept"]
    with pytest.raises(ValueError, match="boom"), _cuda_ipc_mode(tess):
        raise ValueError("boom")
    assert tess._client._output_format == "json+base64"
    assert tess._client._session.headers["Accept"] == prior_accept


# ---------------------------------------------------------------------------
# apply_tesseract(..., cuda_ipc=True): local (in-process) client is a no-op
# ---------------------------------------------------------------------------


def test_cuda_ipc_flag_is_noop_for_local_client(vectoradd_tess):
    """``cuda_ipc=True`` against a LocalClient must behave exactly as without it.

    tests/vectoradd_tesseract is loaded via ``from_tesseract_api``, i.e. an
    in-process LocalClient that already shares memory -- ``_cuda_ipc_mode``
    is a no-op for it (see test_cuda_ipc_mode_noop_for_local_client), so this
    just confirms the flag doesn't break the ordinary CPU path.
    """
    a = torch.tensor([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = apply_tesseract(vectoradd_tess, {"a": a, "b": b}, cuda_ipc=True)
    assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))


# ---------------------------------------------------------------------------
# GPU tensor handling (real CUDA hardware, still a LocalClient)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCudaTensorWithoutCudaIpc:
    """Without ``cuda_ipc=True``, CUDA tensors still take the host round-trip."""

    def test_cuda_tensor_forward_default(self, vectoradd_tess):
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert torch.allclose(result["c"].cpu(), torch.tensor([5.0, 7.0, 9.0]))

    def test_cuda_tensor_forward_with_cuda_ipc_flag_on_local_client(
        self, vectoradd_tess
    ):
        """``cuda_ipc=True`` against a LocalClient must still host-copy.

        ``_supports_cuda_ipc`` is False for a LocalClient (see
        test_supports_cuda_ipc_false_for_local_client_shaped_object), so the
        request is resolved to inactive and the CUDA tensor still takes the
        host round-trip -- passing it through raw would reach the in-process
        endpoint's NumPy-based code, which cannot read GPU memory and would
        raise. This is a regression test for exactly that failure mode.
        """
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b}, cuda_ipc=True)
        assert torch.allclose(result["c"].cpu(), torch.tensor([5.0, 7.0, 9.0]))
