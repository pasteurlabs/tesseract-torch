# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract whose Jacobian depends on where it is evaluated.

Every other fixture here is affine, so its Jacobian is a constant and a
gradient taken at the wrong linearization point is indistinguishable from
one taken at the right place. This one is cubic with a cross term, so the
evaluation point shows up in the answer.

    y = a**3 + b * a          dy/da = 3a**2 + b      dy/db = a

Float64 so the tests can run torch.autograd.gradcheck against it.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float64]] = Field(description="vector a")
    b: Differentiable[Array[(None,), Float64]] = Field(description="vector b")


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(description="a**3 + b*a")


def _arrays(inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(inputs["a"], dtype=np.float64),
        np.asarray(inputs["b"], dtype=np.float64),
    )


def apply(inputs: InputSchema) -> OutputSchema:
    a, b = _arrays(inputs.model_dump())
    return {"y": a**3 + b * a}


def _diagonals(inputs: InputSchema) -> tuple[np.ndarray, np.ndarray]:
    """dy/da and dy/db, both diagonal because the map is elementwise."""
    a, b = _arrays(inputs.model_dump())
    return 3.0 * a**2 + b, a


def jacobian(
    inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]
) -> dict[str, dict[str, Any]]:
    d_a, d_b = _diagonals(inputs)
    cols = {"a": np.diag(d_a), "b": np.diag(d_b)}
    return {"y": {name: cols[name] for name in jac_inputs}}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    d_a, d_b = _diagonals(inputs)
    out = np.zeros_like(d_a)
    if "a" in jvp_inputs:
        out = out + d_a * np.asarray(tangent_vector["a"], dtype=np.float64)
    if "b" in jvp_inputs:
        out = out + d_b * np.asarray(tangent_vector["b"], dtype=np.float64)
    return {"y": out}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    d_a, d_b = _diagonals(inputs)
    ct = np.asarray(cotangent_vector["y"], dtype=np.float64)
    grads = {"a": d_a * ct, "b": d_b * ct}
    return {name: grads[name] for name in vjp_inputs}


def abstract_eval(abstract_inputs: Any) -> dict:
    raw = abstract_inputs.model_dump()["a"]
    shape = raw["shape"] if isinstance(raw, dict) else np.asarray(raw).shape
    return {"y": {"shape": tuple(shape), "dtype": "float64"}}
