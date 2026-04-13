# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract with only an apply endpoint (no JVP or VJP)."""

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    a: Differentiable[Array[(None,), Float32]] = Field(description="Input vector")


class OutputSchema(BaseModel):
    b: Differentiable[Array[(None,), Float32]] = Field(description="Output vector")


def apply(inputs: InputSchema) -> OutputSchema:
    """Doubles the input vector."""
    return OutputSchema(b=inputs.a * 2)
