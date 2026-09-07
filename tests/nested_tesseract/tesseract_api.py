# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32


class Scalars(BaseModel):
    a: Differentiable[Float32] = Field(description="Scalar value a.", default=0.0)
    b: Float32 = Field(description="Scalar value b.")


class Vectors(BaseModel):
    v: Differentiable[Array[(None,), Float32]] = Field(description="Vector value v.")
    w: Array[(3,), Float32] = Field(
        description="Vector value w.", default=[0.0, 1.0, 2.0]
    )


class OtherStuff(BaseModel):
    s: str = Field(description="String value s.")
    i: int = Field(description="Integer value i.")
    f: float = Field(description="Float value f.")


class InputSchema(BaseModel):
    scalars: Scalars
    vectors: Vectors
    other_stuff: OtherStuff


class OutputSchema(BaseModel):
    scalars: Scalars
    vectors: Vectors


def apply(inputs: InputSchema) -> OutputSchema:
    a = inputs.scalars.a
    b = inputs.scalars.b
    v = inputs.vectors.v
    w = inputs.vectors.w

    new_a = a * 10 + b
    new_v = v * 10 + w

    scalars = Scalars(a=new_a, b=b)
    vectors = Vectors(v=new_v, w=w)
    return OutputSchema(scalars=scalars, vectors=vectors)


#
# Optional endpoints
#


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector,
):
    out = {dy: 0.0 for dy in jvp_outputs}
    if "scalars.a" in jvp_inputs and "scalars.a" in jvp_outputs:
        out["scalars.a"] = 10.0 * tangent_vector["scalars.a"]
    if "vectors.v" in jvp_inputs and "vectors.v" in jvp_outputs:
        out["vectors.v"] = 10.0 * tangent_vector["vectors.v"]
    return out


def _record_vjp_outputs(vjp_outputs: set[str]) -> None:
    """Append the sorted ``vjp_outputs`` received, as JSON, to a record file.

    Set via ``$NESTED_VJP_RECORD``. Lets a test assert that ``backward()``
    requests a VJP for only the outputs a loss actually used.
    """
    import json
    import os

    path = os.environ.get("NESTED_VJP_RECORD")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(json.dumps(sorted(vjp_outputs)) + "\n")


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector,
):
    # The runtime requires a gradient for every requested input wire. An input
    # not reachable from any REQUESTED output gets a correctly shaped zero, not
    # a bare 0.0 scalar: the wrapper drops unused outputs from vjp_outputs, so
    # e.g. a loss on scalars.a leaves vectors.v out and vectors.v's input
    # gradient is a zero vector of v's shape. (The map is diagonal:
    # scalars.a<-scalars.a, vectors.v<-vectors.v.)
    import numpy as np

    _record_vjp_outputs(vjp_outputs)

    v_shape = np.asarray(inputs.vectors.v).shape
    zeros = {"scalars.a": np.float32(0.0), "vectors.v": np.zeros(v_shape, np.float32)}
    out = {dx: zeros[dx] for dx in vjp_inputs}
    if "scalars.a" in vjp_inputs and "scalars.a" in vjp_outputs:
        out["scalars.a"] = 10.0 * cotangent_vector["scalars.a"]
    if "vectors.v" in vjp_inputs and "vectors.v" in vjp_outputs:
        out["vectors.v"] = 10.0 * cotangent_vector["vectors.v"]
    return out


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    jac = {dy: {dx: [0.0, 0.0, 0.0] for dx in jac_inputs} for dy in jac_outputs}

    if "scalars.a" in jac_inputs and "scalars.a" in jac_outputs:
        jac["scalars.a"]["scalars.a"] = 10.0
    if "vectors.v" in jac_inputs and "vectors.v" in jac_outputs:
        jac["vectors.v"]["vectors.v"] = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]

    return jac


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    return {
        "scalars": {
            "a": abstract_inputs.scalars.a,
            "b": abstract_inputs.scalars.b,
        },
        "vectors": {
            "v": abstract_inputs.vectors.v,
            "w": abstract_inputs.vectors.w,
        },
    }
