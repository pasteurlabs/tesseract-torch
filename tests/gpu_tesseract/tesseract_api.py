# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Tesseract behind the GPU tests, computing with PyTorch, CuPy or NumPy.

``TESSERACT_TORCH_TEST_ARRAY_MODULE`` selects the array module ``xp``. The GPU
tests run against ``torch`` and against ``cupy``, which stands in for any other
GPU framework and differs from PyTorch in how it adopts device inputs and
allocates outputs. ``numpy`` (the default) is for a server that cannot see the
GPU. On the GPU, inputs that arrive in host memory are copied to the device, or
fail the call if ``TESSERACT_TORCH_TEST_REJECT_HOST_INPUTS`` is set.

The math is ``c = (a * scale + b) * mask``. The non-differentiable input
``mask`` (default all-ones) and output ``c_sum`` exercise the placeholder paths
for non-differentiable fields.
"""

import os
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

_ARRAY_MODULE = os.environ.get("TESSERACT_TORCH_TEST_ARRAY_MODULE", "numpy")
_REJECT_HOST_INPUTS = bool(os.environ.get("TESSERACT_TORCH_TEST_REJECT_HOST_INPUTS"))

if _ARRAY_MODULE == "numpy":
    xp = np

    def _adopt(x: Any) -> Any:
        return np.asarray(x)

elif _ARRAY_MODULE == "torch":
    import torch as xp

    def _adopt(x: Any) -> Any:
        if isinstance(x, xp.Tensor):
            return x.cuda()
        if isinstance(x, np.ndarray):
            return xp.tensor(x, device="cuda")
        # A device array, such as a cuda_ipc input
        return xp.from_dlpack(x)

elif _ARRAY_MODULE == "cupy":
    import cupy as xp

    def _adopt(x: Any) -> Any:
        # Takes device memory as it is and copies host arrays to the GPU
        return xp.asarray(x)

else:
    raise ValueError(f"Unknown TESSERACT_TORCH_TEST_ARRAY_MODULE {_ARRAY_MODULE!r}")


def _asxp(x: Any) -> Any:
    """Adopt an input array with ``xp``, without copying a GPU-resident one."""
    if _REJECT_HOST_INPUTS and xp is not np and isinstance(x, np.ndarray):
        raise RuntimeError("Expected a GPU-resident input, got a NumPy array.")
    return _adopt(x)


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Differentiable[Array[(None,), Float32]] = Field(description="Vector b")
    scale: Float32 = Field(default=2.0, description="Scalar scale")
    mask: Array[(None,), Float32] | None = Field(
        default=None,
        description="Non-differentiable elementwise mask; defaults to all-ones.",
    )


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="(a*scale + b)*mask")
    c_sum: Array[(1,), Float32] = Field(
        description="Non-differentiable diagnostic: sum(c), shape (1,)."
    )


def _mask_for(inputs, a):
    return _asxp(inputs.mask) if inputs.mask is not None else xp.ones_like(a)


def apply(inputs: InputSchema) -> OutputSchema:
    a = _asxp(inputs.a)
    c = (a * float(inputs.scale) + _asxp(inputs.b)) * _mask_for(inputs, a)
    return OutputSchema(c=c, c_sum=c.sum().reshape(1))


def abstract_eval(abstract_inputs):
    return {
        "c": abstract_inputs.a,
        "c_sum": {"shape": (1,), "dtype": "float32"},
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    a = _asxp(inputs.a)
    mask = _mask_for(inputs, a)
    out = xp.zeros_like(a)
    if "a" in tangent_vector:
        out = out + _asxp(tangent_vector["a"]) * float(inputs.scale) * mask
    if "b" in tangent_vector:
        out = out + _asxp(tangent_vector["b"]) * mask
    return {"c": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    a = _asxp(inputs.a)
    ct = _asxp(cotangent_vector["c"]) * _mask_for(inputs, a)
    out = {}
    if "a" in vjp_inputs:
        out["a"] = ct * float(inputs.scale)
    if "b" in vjp_inputs:
        out["b"] = ct
    return out
