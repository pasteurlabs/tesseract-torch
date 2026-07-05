# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
from torch.utils._pytree import tree_map


class Vector_and_Scalar(BaseModel):
    v: Differentiable[Array[(None,), Float32]] = Field(
        description="An arbitrary vector"
    )
    s: Differentiable[Float32] = Field(description="A scalar", default=1.0)

    def scale(self) -> Differentiable[Array[(None,), Float32]]:
        return self.s * self.v


class InputSchema(BaseModel):
    a: Vector_and_Scalar = Field(
        description="An arbitrary vector and a scalar to multiply it by"
    )
    b: Vector_and_Scalar = Field(
        description="An arbitrary vector and a scalar to multiply it by, "
        "must be of same shape as a"
    )
    norm_ord: int = Field(
        description="Order of norm (see numpy.linalg.norm)",
        default=2,
    )

    @model_validator(mode="after")
    def validate_shape_inputs(self) -> Self:
        if self.a.v.shape != self.b.v.shape:
            raise ValueError(
                f"a.v and b.v must have the same shape. "
                f"Got {self.a.v.shape} and {self.b.v.shape} instead."
            )
        return self


class Result_and_Norm(BaseModel):
    result: Differentiable[Array[(None,), Float32]] = Field(
        description="Vector s_a·a + s_b·b"
    )
    normed_result: Differentiable[Array[(None,), Float32]] = Field(
        description="Normalized Vector s_a·a + s_b·b/|s_a·a + s_b·b|"
    )


class OutputSchema(BaseModel):
    vector_add: Result_and_Norm
    vector_min: Result_and_Norm


to_tensor = lambda x: torch.tensor(x) if isinstance(x, np.generic | np.ndarray) else x


def evaluate(inputs: dict) -> dict:
    """Core differentiable computation."""
    a_scaled = inputs["a"]["s"] * inputs["a"]["v"]
    b_scaled = inputs["b"]["s"] * inputs["b"]["v"]
    add_result = a_scaled + b_scaled
    min_result = a_scaled - b_scaled

    def safe_norm(x, ord):
        return torch.pow(torch.pow(torch.abs(x), ord).sum() + 1e-8, 1.0 / ord)

    return {
        "vector_add": {
            "result": add_result,
            "normed_result": add_result / safe_norm(add_result, ord=inputs["norm_ord"]),
        },
        "vector_min": {
            "result": min_result,
            "normed_result": min_result / safe_norm(min_result, ord=inputs["norm_ord"]),
        },
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Multiplies a vector `a` by `s`, and sums the result to `b`."""
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    return evaluate(tensor_inputs)


#
# PyTorch-handled gradient endpoints
#


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    jvp_inputs = tuple(jvp_inputs)
    tangent_vector = {key: tangent_vector[key] for key in jvp_inputs}

    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    pos_tangent = tree_map(to_tensor, tangent_vector).values()
    pos_inputs = flatten_with_paths(tensor_inputs, jvp_inputs).values()

    filtered_pos_eval = filter_func(
        evaluate, tensor_inputs, jvp_outputs, input_paths=jvp_inputs
    )

    return torch.func.jvp(filtered_pos_eval, tuple(pos_inputs), tuple(pos_tangent))[1]


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    vjp_inputs = tuple(vjp_inputs)
    cotangent_vector = {key: cotangent_vector[key] for key in vjp_outputs}

    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    tensor_cotangent = tree_map(to_tensor, cotangent_vector)
    pos_inputs = flatten_with_paths(tensor_inputs, vjp_inputs).values()

    filtered_pos_func = filter_func(
        evaluate, tensor_inputs, vjp_outputs, input_paths=vjp_inputs
    )

    _, vjp_func = torch.func.vjp(filtered_pos_func, *pos_inputs)
    vjp_vals = vjp_func(tensor_cotangent)
    return dict(zip(vjp_inputs, vjp_vals, strict=True))


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    jac_inputs = tuple(jac_inputs)
    tensor_inputs = tree_map(to_tensor, inputs.model_dump())
    pos_inputs = flatten_with_paths(tensor_inputs, jac_inputs).values()

    filtered_pos_eval = filter_func(
        evaluate, tensor_inputs, jac_outputs, input_paths=jac_inputs
    )

    def filtered_pos_eval_flat(*args):
        res = filtered_pos_eval(*args)
        return tuple(res[k] for k in jac_outputs)

    jac = torch.autograd.functional.jacobian(filtered_pos_eval_flat, tuple(pos_inputs))

    jac_dict = {}
    for dy, dys in zip(jac_outputs, jac, strict=True):
        jac_dict[dy] = {}
        for dx, dxs in zip(jac_inputs, dys, strict=True):
            jac_dict[dy][dx] = dxs

    return jac_dict
