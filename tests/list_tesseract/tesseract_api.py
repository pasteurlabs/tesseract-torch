# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract with a list-valued differentiable field.

Each entry gets its own coefficient, so a gradient routed to the wrong
position shows up in the value rather than only in the requested path.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32

COEFFS = (2.0, 5.0)


class InputSchema(BaseModel):
    xs: list[Differentiable[Array[(3,), Float32]]]


class OutputSchema(BaseModel):
    total: Differentiable[Array[(3,), Float32]]


def apply(inputs: InputSchema) -> OutputSchema:
    xs = inputs.model_dump()["xs"]
    total = sum(c * np.asarray(x, np.float32) for c, x in zip(COEFFS, xs, strict=True))
    return {"total": total}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    ct = np.asarray(cotangent_vector["total"], np.float32)
    out = {}
    for path in vjp_inputs:
        idx = int(path.removeprefix("xs.[").removesuffix("]"))
        out[path] = COEFFS[idx] * ct
    return out


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    total = np.zeros(3, np.float32)
    for path, tangent in tangent_vector.items():
        idx = int(path.removeprefix("xs.[").removesuffix("]"))
        total = total + COEFFS[idx] * np.asarray(tangent, np.float32)
    return {"total": total}


def abstract_eval(abstract_inputs: Any) -> dict:
    return {"total": {"shape": (3,), "dtype": "float32"}}
