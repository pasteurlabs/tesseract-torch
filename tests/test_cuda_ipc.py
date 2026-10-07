# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for how ``apply_tesseract`` picks and drives a GPU transport.

These exercise the pure-Python plumbing (which transport a call uses, when it
goes through an encoding view, and the ``_to_tensor`` decode gate) without
needing a GPU. Full end-to-end coverage (a real ``HTTPClient`` talking CUDA IPC
to a served GPU Tesseract) lives in ``test_gpu_direct`` and tesseract-core's own
suite; here we only need to verify tesseract-torch drives that API correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tesseract_core import Tesseract

from tesseract_torch import apply_tesseract
from tesseract_torch.function import (
    _resolve_gpu_transport,
    _tensor_to_numpy_or_cuda,
    _to_tensor,
    _with_gpu_transport,
)


@pytest.fixture
def vectoradd_api_path() -> Path:
    return Path(__file__).parent / "vectoradd_tesseract" / "tesseract_api.py"


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
    """An input that is already a tensor passes through untouched."""
    t = torch.tensor([1.0, 2.0])
    assert _to_tensor(t) is t


def test_to_tensor_adopts_cuda_array_interface_via_dlpack():
    """An object exposing ``__cuda_array_interface__`` decodes via DLPack.

    Mimics ``tesseract_core.runtime.cuda.ipc.IpcDeviceArray`` without needing
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


def test_tensor_to_numpy_or_cuda_cpu_tensor_returns_numpy():
    t = torch.tensor([1.0, 2.0, 3.0])
    out = _tensor_to_numpy_or_cuda(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_tensor_to_numpy_or_cuda_cpu_tensor_returns_numpy_even_on_device():
    """``on_device=True`` only changes behavior for CUDA tensors."""
    t = torch.tensor([1.0, 2.0, 3.0])
    out = _tensor_to_numpy_or_cuda(t, on_device=True)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_to_numpy_or_cuda_cuda_tensor_default_is_host_copy():
    """Without ``on_device=True``, a CUDA tensor still takes the host round-trip.

    The default client encoder calls ``np.asanyarray`` on whatever it's handed,
    which cannot read GPU memory -- so passing a raw CUDA tensor through here
    when no device transport is active would break the client, not speed it up.
    """
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    out = _tensor_to_numpy_or_cuda(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_to_numpy_or_cuda_cuda_tensor_on_device_passes_through():
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    out = _tensor_to_numpy_or_cuda(t, on_device=True)
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


def test_cpu_tensors_take_the_host_path(vectoradd_api_path):
    """Only a call with CUDA tensors asks the Tesseract for a transport.

    CPU tensors have nothing to keep on a device, so even a Tesseract that
    accepts GPU arrays gets them as NumPy arrays, and its NumPy-based endpoint
    works with them.
    """
    tess = Tesseract.from_tesseract_api(vectoradd_api_path, gpu_transport="cuda_ipc")
    assert _resolve_gpu_transport(tess, torch.device("cpu")) == "none"
    assert _resolve_gpu_transport(tess, torch.device("cuda")) == "cuda_ipc"

    a = torch.tensor([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = apply_tesseract(tess, {"a": a, "b": b})
    assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))


def test_in_process_transport_is_the_one_it_was_created_with(vectoradd_tess):
    assert _resolve_gpu_transport(vectoradd_tess, torch.device("cuda")) == "none"


def test_unsupported_transport_is_rejected(vectoradd_api_path, monkeypatch):
    """A transport the GPU path cannot drive must not be sent CUDA tensors."""
    tess = Tesseract.from_tesseract_api(vectoradd_api_path)
    monkeypatch.setattr(tess, "resolve_gpu_transport", lambda: "nixl")
    with pytest.raises(ValueError, match="tesseract-torch cannot drive"):
        _resolve_gpu_transport(tess, torch.device("cuda"))


def test_with_gpu_transport_views_only_when_needed(vectoradd_api_path):
    """A call uses the Tesseract itself unless it needs another transport.

    Served with cuda_ipc, a call that needs ``"none"`` goes through a view
    requesting it, and every other combination through the Tesseract itself.
    """
    in_process = Tesseract.from_tesseract_api(vectoradd_api_path)
    assert _with_gpu_transport(in_process, "cuda_ipc") is in_process

    with Tesseract.from_source(
        vectoradd_api_path, python_executable=sys.executable, gpu_transport="cuda_ipc"
    ) as served:
        assert _with_gpu_transport(served, "cuda_ipc") is served
        host = _with_gpu_transport(served, "none")
        assert host is not served
        assert host.current_encoding.gpu_transport == "none"

        remote = Tesseract.from_url(served._client.url)
        assert _with_gpu_transport(remote, "none") is remote
        assert _with_gpu_transport(
            remote, "cuda_ipc"
        ).current_encoding.gpu_transport == ("cuda_ipc")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCudaTensorWithoutDeviceTransport:
    """Without a device transport, CUDA tensors still take the host round-trip."""

    def test_cuda_tensor_forward_default(self, vectoradd_tess):
        """The output still lands on the input's device."""
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert result["c"].is_cuda
        assert torch.allclose(result["c"].cpu(), torch.tensor([5.0, 7.0, 9.0]))
