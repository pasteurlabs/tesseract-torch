# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU-resident Tesseract for exercising the cuda_ipc dispatch path.

``apply`` computes on ``torch`` CUDA tensors so its outputs stay in GPU memory,
which is what lets the runtime export them via CUDA IPC (no device->host copy).
The core math is a simple elementwise ``c = (a * scale + b) * mask`` so parity
against a NumPy reference is trivial to check.

``mask`` is a *non-differentiable* array input (contrast ``a`` / ``b``, which
are ``Differentiable``). It defaults to all-ones so callers that omit it get
the plain ``a * scale + b``, but when supplied it exercises the derivative
code's handling of a non-differentiable, non-static array input.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Vector a")
    b: Differentiable[Array[(None,), Float32]] = Field(description="Vector b")
    scale: Float32 = Field(default=np.float32(2.0), description="Scalar scale")
    mask: Array[(None,), Float32] | None = Field(
        default=None,
        description="Non-differentiable elementwise mask; defaults to all-ones.",
    )


class OutputSchema(BaseModel):
    c: Differentiable[Array[(None,), Float32]] = Field(description="(a*scale + b)*mask")
    c_sum: Array[(1,), Float32] = Field(
        description="Non-differentiable diagnostic: sum(c), shape (1,)."
    )


def _to_cuda(x):
    import torch

    # x may arrive as a numpy array (base64 inputs) or an IpcDeviceArray
    # (cuda_ipc inputs, exposing __cuda_array_interface__/__dlpack__).
    # from_dlpack keeps a cuda_ipc array on-device; as_tensor moves a numpy
    # array onto the device.
    if hasattr(x, "__dlpack__"):
        return torch.utils.dlpack.from_dlpack(x).cuda()
    return torch.as_tensor(np.asarray(x)).cuda()


def _compute_c(inputs):
    import torch

    a = _to_cuda(inputs.a)
    b = _to_cuda(inputs.b)
    scale = float(inputs.scale)
    c = a * scale + b  # stays on GPU
    mask = _to_cuda(inputs.mask) if inputs.mask is not None else torch.ones_like(a)
    return c * mask


def apply(inputs: InputSchema) -> OutputSchema:
    c = _compute_c(inputs)
    # c_sum is a non-differentiable output: it makes the derivative endpoints
    # emit a placeholder for it, exercising the GPU-direct return path for a
    # non-differentiable output slot.
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
    import torch

    scale = float(inputs.scale)
    a = _to_cuda(inputs.a)
    mask = _to_cuda(inputs.mask) if inputs.mask is not None else torch.ones_like(a)
    out = torch.zeros_like(a)
    if "a" in tangent_vector:
        out = out + _to_cuda(tangent_vector["a"]) * scale * mask
    if "b" in tangent_vector:
        out = out + _to_cuda(tangent_vector["b"]) * mask
    return {"c": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    import torch

    scale = float(inputs.scale)
    a = _to_cuda(inputs.a)
    mask = _to_cuda(inputs.mask) if inputs.mask is not None else torch.ones_like(a)
    ct = _to_cuda(cotangent_vector["c"]) * mask
    out = {}
    if "a" in vjp_inputs:
        out["a"] = ct * scale
    if "b" in vjp_inputs:
        out["b"] = ct
    return out
