# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A batch-polymorphic Tesseract for the ``torch.vmap`` tests.

Every array field is ``Array[..., Float64]``, so a leading batch dimension is
valid input, and the endpoints broadcast like NumPy. The Jacobian depends on
the evaluation point, so a gradient taken at the wrong batch element shows up.

    y = a**3 + b * a              dy/da = 3a**2 + b      dy/db = a
    extras.z = n * (a + c) + shift    not differentiable

The non-differentiable outputs sit in a nested model with no differentiable
field, which ``apply_tesseract`` otherwise keeps whole as a single value, and
so does the optional ``opts.shift`` input.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64


class Options(BaseModel):
    shift: Array[..., Float64] = Field(description="added to extras.z")


class InputSchema(BaseModel):
    a: Differentiable[Array[..., Float64]] = Field(description="array a")
    b: Differentiable[Array[..., Float64]] = Field(description="array b")
    c: Array[..., Float64] = Field(description="non-differentiable offset")
    opts: Options | None = Field(default=None, description="optional shift")
    n: int = Field(default=1, description="scale of extras.z")


class Extras(BaseModel):
    z: Array[..., Float64] = Field(description="n * (a + c) + shift")
    positive: bool = Field(description="whether every entry of a is positive")


class OutputSchema(BaseModel):
    y: Differentiable[Array[..., Float64]] = Field(description="a**3 + b*a")
    extras: Extras = Field(description="non-differentiable outputs")
    tags: list[str] = Field(default_factory=list, description="always empty")


def apply(inputs: InputSchema) -> OutputSchema:
    a, b, c = inputs.a, inputs.b, inputs.c
    shift = inputs.opts.shift if inputs.opts is not None else 0.0
    z = inputs.n * (a + c) + shift
    extras = Extras(z=z, positive=bool(np.all(a > 0)))
    return OutputSchema(y=a**3 + b * a, extras=extras)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
) -> dict[str, Any]:
    a, b = inputs.a, inputs.b
    y = np.zeros(np.broadcast_shapes(a.shape, b.shape))
    if "a" in jvp_inputs:
        y = y + (3.0 * a**2 + b) * tangent_vector["a"]
    if "b" in jvp_inputs:
        y = y + a * tangent_vector["b"]
    return {"y": y}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
) -> dict[str, Any]:
    a, b = inputs.a, inputs.b
    ct = cotangent_vector["y"]
    grads = {"a": (3.0 * a**2 + b) * ct, "b": a * ct}
    return {name: grads[name] for name in vjp_inputs}
