# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract with a list-valued differentiable field.

Entry i is weighed by ``coefficient(i)``, distinct per position and defined
for any length, so a gradient delivered to the wrong position shows up in the
value rather than only in the requested path.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32


def coefficient(index: int) -> float:
    """Weight of the entry at *index*."""
    return 2.0 + index


def _index_of(path: str) -> int:
    """Position addressed by a wire path such as ``xs.[2]``."""
    return int(path.removeprefix("xs.[").removesuffix("]"))


class InputSchema(BaseModel):
    xs: list[Differentiable[Array[(3,), Float32]]]


class OutputSchema(BaseModel):
    total: Differentiable[Array[(3,), Float32]]


def apply(inputs: InputSchema) -> OutputSchema:
    xs = inputs.model_dump()["xs"]
    total = sum(coefficient(i) * np.asarray(x, np.float32) for i, x in enumerate(xs))
    return {"total": total}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    ct = np.asarray(cotangent_vector["total"], np.float32)
    return {path: coefficient(_index_of(path)) * ct for path in vjp_inputs}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    total = np.zeros(3, np.float32)
    for path, tangent in tangent_vector.items():
        total = total + coefficient(_index_of(path)) * np.asarray(tangent, np.float32)
    return {"total": total}


def abstract_eval(abstract_inputs: Any) -> dict:
    return {"total": {"shape": (3,), "dtype": "float32"}}
