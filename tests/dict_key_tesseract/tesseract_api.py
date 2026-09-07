# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Tesseract with a dict-valued differentiable field.

The schema constrains the keys to nothing beyond ``str``, so a caller may use
whichever ones it wants to exercise. Every gradient is the same fixed multiple
of the cotangent, which puts a mispathed one in the value rather than only in
the requested path.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32


class InputSchema(BaseModel):
    params: dict[str, Differentiable[Array[(3,), Float32]]]


class OutputSchema(BaseModel):
    result: Differentiable[Array[(3,), Float32]]


def apply(inputs: InputSchema) -> OutputSchema:
    p = inputs.model_dump()["params"]
    return {"result": sum(2.0 * np.asarray(v, np.float32) for v in p.values())}


def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    ct = np.asarray(cotangent_vector["result"], np.float32)
    return {k: 2.0 * ct for k in vjp_inputs}


def abstract_eval(abstract_inputs: Any) -> dict:
    return {"result": {"shape": (3,), "dtype": "float32"}}
