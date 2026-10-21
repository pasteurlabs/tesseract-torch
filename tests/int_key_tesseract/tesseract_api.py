# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dicts keyed by int on both sides, next to a differentiable array."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]] = Field(description="scaled input")
    weights: dict[int, float] = Field(
        description="added to the scale of x, which starts at 1", default={}
    )
    offsets: dict[int, Array[(None,), Float64]] = Field(
        description="added to y", default={}
    )


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(
        description="(1 + sum(weights)) * x + sum(offsets)"
    )
    by_index: dict[int, float] = Field(description="the entries of y, by position")


def _scale(inputs: InputSchema) -> float:
    return 1.0 + sum(inputs.weights.values())


def apply(inputs: InputSchema) -> OutputSchema:
    y = _scale(inputs) * np.asarray(inputs.x)
    for offset in inputs.offsets.values():
        y = y + np.asarray(offset)
    return OutputSchema(y=y, by_index={i: float(v) for i, v in enumerate(y)})


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    return {"x": _scale(inputs) * np.asarray(cotangent_vector["y"])}
