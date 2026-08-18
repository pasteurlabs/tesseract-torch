# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-valued differentiable fields on both the input and output side.

Coefficients differ per field so a mispathed gradient shows up in the value,
not only in the requested path.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    x: Differentiable[Array[(3,), Float32]] = Field(description="plain array field")
    params: dict[str, Differentiable[Array[(3,), Float32]]] = Field(
        description="named parameter groups, all leaves differentiable"
    )


class OutputSchema(BaseModel):
    y: Differentiable[Array[(3,), Float32]] = Field(description="2*x + 10*params.p")
    outs: dict[str, Differentiable[Array[(3,), Float32]]] = Field(
        description="dict-valued differentiable output"
    )


def _arrays(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs["x"], dtype=np.float32)
    p = np.asarray(inputs["params"]["p"], dtype=np.float32)
    return x, p


def apply(inputs: InputSchema) -> OutputSchema:
    x, p = _arrays(inputs.model_dump())
    return {"y": 2.0 * x + 10.0 * p, "outs": {"a": 3.0 * x, "b": 7.0 * x}}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    zero = np.zeros(3, dtype=np.float32)

    def ct(name: str) -> np.ndarray:
        v = cotangent_vector.get(name)
        return zero if v is None else np.asarray(v, dtype=np.float32)

    d_x = 2.0 * ct("y") + 3.0 * ct("outs.{a}") + 7.0 * ct("outs.{b}")
    d_p = 10.0 * ct("y")

    out: dict[str, Any] = {}
    for path in vjp_inputs:
        if path == "x":
            out["x"] = d_x
        elif path == "params.{p}":
            out["params.{p}"] = d_p
    return out


def abstract_eval(abstract_inputs: Any) -> dict:
    shape = {"shape": (3,), "dtype": "float32"}
    return {"y": shape, "outs": {"a": shape, "b": shape}}
