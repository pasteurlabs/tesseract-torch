# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths
from torch.utils._pytree import tree_map


class PointCloud(BaseModel):
    positions: Differentiable[Array[(None, 3), Float32]] = Field(
        description="Cartesian coordinates of the points, one row per point"
    )
    weights: Differentiable[Array[(None,), Float32]] = Field(
        description="Weight of each point, one entry per row of positions"
    )


class Settings(BaseModel):
    label: str = Field(
        description="Name reported back alongside the statistics", default="cloud"
    )
    scale: float = Field(
        description="Factor applied to the positions before anything is computed",
        default=1.0,
    )


class InputSchema(BaseModel):
    cloud: PointCloud = Field(description="The weighted point cloud to summarize")
    settings: Settings = Field(
        description="Plain, non-differentiable options", default_factory=Settings
    )

    @model_validator(mode="after")
    def validate_one_weight_per_point(self) -> Self:
        n_points = self.cloud.positions.shape[0]
        if self.cloud.weights.shape != (n_points,):
            raise ValueError(
                f"cloud.weights must have one entry per point, got shape "
                f"{self.cloud.weights.shape} for {n_points} points."
            )
        return self


class Statistics(BaseModel):
    barycenter: Differentiable[Array[(3,), Float32]] = Field(
        description="Weighted mean position"
    )
    radius_of_gyration: Differentiable[Float32] = Field(
        description="Weighted root mean square distance from the barycenter"
    )


class Summary(BaseModel):
    label: str = Field(description="The label passed in via settings")
    n_points: int = Field(description="Number of points in the cloud")


class OutputSchema(BaseModel):
    statistics: Statistics
    summary: Summary


to_tensor = lambda x: torch.tensor(x) if isinstance(x, np.generic | np.ndarray) else x


def evaluate(inputs: dict) -> dict:
    """Core differentiable computation."""
    positions = inputs["settings"]["scale"] * inputs["cloud"]["positions"]
    weights = inputs["cloud"]["weights"]
    total_weight = weights.sum()

    barycenter = (weights[:, None] * positions).sum(dim=0) / total_weight
    offsets = positions - barycenter
    radius_of_gyration = torch.sqrt(
        (weights * (offsets * offsets).sum(dim=1)).sum() / total_weight
    )

    return {
        "statistics": {
            "barycenter": barycenter,
            "radius_of_gyration": radius_of_gyration,
        },
        "summary": {
            "label": inputs["settings"]["label"],
            "n_points": positions.shape[0],
        },
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Summarize a weighted point cloud by its barycenter and radius of gyration."""
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
