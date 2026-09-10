# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable PyTorch wrapper for Tesseract operations.

This module registers a Tesseract as a first-class differentiable primitive in
PyTorch's autograd graph.  The forward pass dispatches to ``tesseract.apply()``,
the backward pass to ``tesseract.vector_jacobian_product()``, and the
forward-mode JVP to ``tesseract.jacobian_vector_product()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tesseract_core import Tesseract

# A leaf's path, one entry per schema level. Kept as segments rather than a
# dotted string because a dict key is free to contain a dot. An int segment is
# a list position, which is what keeps it apart from a dict key that merely
# looks like one: {"0": v} carries "0" here, [v] carries 0.
type KeyType = tuple[str | int, ...]


def _to_tensor(arr: Any) -> torch.Tensor:
    """Convert a numpy array to a float32 tensor, copying if read-only."""
    a = np.asarray(arr)
    if not a.flags.writeable:
        a = a.copy()
    return torch.as_tensor(a)


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array, preserving dtype."""
    # torch.func transforms (vjp, jvp, grad, vmap) wrap tensors in a C++
    # FunctionalTensorWrapper that has no backing storage.  These tensors
    # report type(t)==torch.Tensor (no Python subclass), so there is no
    # isinstance check we can use.  Instead we probe data_ptr(), the same
    # public precondition that .numpy() relies on, to raise an actionable
    # error instead of the confusing default message ("Cannot access data
    # pointer of Tensor that doesn't have storage").
    try:
        t.data_ptr()
    except RuntimeError:
        raise RuntimeError(
            "apply_tesseract does not support torch.func transforms "
            "(torch.func.vjp, torch.func.jvp, torch.func.grad, etc.). "
            "Use the standard autograd API instead:\n"
            "  - Reverse mode: result['y'].backward() or torch.autograd.grad()\n"
            "  - Forward mode: torch.autograd.forward_ad (dual tensors)"
        ) from None
    return t.detach().cpu().numpy()


def _get_differentiable_arrays(
    openapi_schema: dict,
    component: str,
) -> set[str]:
    """Extract differentiable array dotted-paths from the OpenAPI schema."""
    schema = openapi_schema["components"]["schemas"].get(component, {})
    return set(schema.get("differentiable_arrays", {}))


# ---------------------------------------------------------------------------
# Pytree helpers - flatten / unflatten nested dicts using dotted paths
# ---------------------------------------------------------------------------


def _flatten_pytree(
    tree: Any,
    prefix: KeyType = (),
    *,
    recurse_into: set[str] | None = None,
) -> list[tuple[KeyType, Any]]:
    """Flatten a nested container into ``(path_parts, leaf_value)`` pairs.

    The path stays a tuple of segments rather than a dotted string because a
    dict key is free to contain a dot, and joining loses where the key ends.
    A list entry contributes its position as an int.

    Only recurses into sub-containers a path in *recurse_into* reaches past.
    Everything else is a leaf, which is how a container the schema does not
    mark differentiable stays whole. ``None`` recurses everywhere.
    """
    children = _children(tree)
    if children is None:
        return [(prefix, tree)]
    items: list[tuple[KeyType, Any]] = []
    for key, child in children:
        path = (*prefix, key)
        if _should_recurse(path, child, recurse_into):
            items.extend(_flatten_pytree(child, path, recurse_into=recurse_into))
        else:
            items.append((path, child))
    return items


def _children(value: Any) -> list[tuple[str | int, Any]] | None:
    """A container's (segment, child) pairs, or None if it is a leaf.

    An array is a leaf however sequence-like it looks, since the schema means
    it as one value rather than a list of them.
    """
    if isinstance(value, dict):
        return list(value.items())
    if isinstance(value, torch.Tensor | np.ndarray):
        return None
    if isinstance(value, list | tuple):
        return list(enumerate(value))
    return None


def _segment_matches(template: str, segment: str | int) -> bool:
    """True when a template segment covers a concrete one.

    ``{}`` stands for any dict key and ``[]`` for any list position, so which
    wildcard applies depends on what the segment is.
    """
    if isinstance(segment, int):
        return template == _LIST_WILDCARD
    return template in (_DICT_WILDCARD, segment)


def _should_recurse(
    path: KeyType,
    value: Any,
    known_paths: set[str] | None,
) -> bool:
    """True when some declared path reaches past *path* into this container.

    A container is only worth opening if a leaf is declared below it, and the
    path so far has to match that declaration segment by segment. An empty one
    holds nothing to reach, so it stays a leaf.
    """
    if not _children(value):
        return False
    if known_paths is None:
        return True
    for known in known_paths:
        declared = known.split(".")
        if len(declared) <= len(path):
            continue
        if all(_segment_matches(t, s) for t, s in zip(declared, path, strict=False)):
            return True
    return False


# ---------------------------------------------------------------------------
# Wildcard paths
# ---------------------------------------------------------------------------
#
# tesseract-core encodes a container-valued differentiable field as a template
# rather than a concrete path: ``dict[str, Differentiable[...]]`` is declared as
# ``params.{}``. The wire name the runtime accepts puts the concrete key in the
# braces, ``params.{p}``, because its own path regex compiles the sentinel to
# ``\{[^.]+\}``. Neither ``params.p`` nor ``params.{}`` is accepted.
#
# So a concrete leaf path and the name used to talk to the Tesseract are not
# the same string, and both are needed: the concrete path rebuilds the input
# pytree, the wire name addresses the endpoint.

_DICT_WILDCARD = "{}"
_LIST_WILDCARD = "[]"


def _is_templated(path: str) -> bool:
    """True when *path* carries a container wildcard segment."""
    parts = path.split(".")
    return _DICT_WILDCARD in parts or _LIST_WILDCARD in parts


def _wire_name(concrete_parts: KeyType, templates: set[str]) -> str | None:
    """Map a concrete leaf path to the name the Tesseract expects.

    Returns ``None`` when no declared path covers it, which is how the caller
    tells a differentiable leaf from a static one.
    """
    joined = ".".join(str(part) for part in concrete_parts)
    if joined in templates:
        return joined

    for template in templates:
        template_parts = template.split(".")
        if len(template_parts) != len(concrete_parts):
            continue
        resolved: list[str] = []
        for tpl, concrete in zip(template_parts, concrete_parts, strict=True):
            if tpl == _DICT_WILDCARD and isinstance(concrete, str):
                resolved.append("{" + concrete + "}")
            elif tpl == _LIST_WILDCARD and isinstance(concrete, int):
                resolved.append(f"[{concrete}]")
            elif tpl != concrete:
                break
            else:
                resolved.append(str(concrete))
        else:
            return ".".join(resolved)
    return None


def _resolve_output_names(
    flat_result: dict[KeyType, Any], templates: list[str]
) -> list[tuple[KeyType, str]]:
    """Pair each concrete output leaf with its wire name, in a stable order.

    Concrete keys of a dict-valued output are not knowable until ``apply``
    returns, so the differentiable output list is resolved here rather than
    derived from the schema up front.
    """
    resolved: list[tuple[KeyType, str]] = []
    for concrete in sorted(flat_result):
        wire = _wire_name(concrete, set(templates))
        if wire is not None:
            resolved.append((concrete, wire))
    return resolved


def _unflatten_pytree(flat: dict[KeyType, Any]) -> dict[str, Any]:
    """Reconstruct a nested container from ``{path_parts: value}``.

    Segments are built into dicts first and the ones holding list positions are
    turned back into lists afterwards. A schema will not accept a numeric-keyed
    dict where it wants a list, and only a list position ever arrives as an int,
    so this cannot mistake a dict key that looks like a number for one.
    """
    tree: dict[str | int, Any] = {}
    for parts, value in flat.items():
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return _relist(tree)


def _relist(node: Any) -> Any:
    """Turn every dict built from list positions back into a list."""
    if not isinstance(node, dict):
        return node
    rebuilt = {key: _relist(value) for key, value in node.items()}
    if rebuilt and all(isinstance(key, int) for key in rebuilt):
        return [rebuilt[key] for key in sorted(rebuilt)]
    return rebuilt


# ---------------------------------------------------------------------------
# Core autograd function
# ---------------------------------------------------------------------------


class _TesseractFunction(torch.autograd.Function):
    """Low-level autograd function wrapping a Tesseract.

    This is an implementation detail.  Users should call :func:`apply_tesseract`.
    """

    @staticmethod
    def forward(
        tesseract: Tesseract,
        diff_input_paths: list[KeyType],
        diff_input_wires: list[str],
        diff_output_templates: list[str],
        all_paths: set[str],
        static_inputs: dict[KeyType, Any],
        non_diff_result_holder: list[Any],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the Tesseract forward pass, returning differentiable outputs.

        The full (flat) result dict and the resolved output names are stashed
        in *non_diff_result_holder* so the caller can reconstruct
        non-differentiable outputs without a second ``apply()`` call.

        Output names are resolved here rather than passed in: a dict-valued
        differentiable output is declared as a template, and its concrete keys
        only exist once ``apply`` has returned.
        """
        flat_inputs = dict(static_inputs)
        for path, tensor in zip(diff_input_paths, tensors, strict=True):
            flat_inputs[path] = _tensor_to_numpy(tensor)

        result = tesseract.apply(_unflatten_pytree(flat_inputs))
        flat_result = dict(_flatten_pytree(result, recurse_into=all_paths))

        resolved_outputs = _resolve_output_names(flat_result, diff_output_templates)

        # Stash full result + resolved output names for the caller
        non_diff_result_holder.append(flat_result)
        non_diff_result_holder.append(resolved_outputs)

        return tuple(_to_tensor(flat_result[c]) for c, _ in resolved_outputs)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        """Save forward-pass metadata for use in backward / jvp."""
        # Do not materialise zero cotangents for outputs the loss never used:
        # let those arrive as None in backward() so we can drop them from the
        # VJP request entirely, rather than paying the Tesseract to compute a
        # gradient it will multiply by zero. (A seam with several
        # differentiable outputs otherwise runs one autograd.grad per output on
        # every backward, even for outputs with no incoming gradient.)
        ctx.set_materialize_grads(False)
        (
            tesseract,
            diff_input_paths,
            diff_input_wires,
            diff_output_templates,  # noqa: RUF059
            all_paths,  # noqa: RUF059
            static_inputs,
            holder,
            *tensors,
        ) = inputs
        ctx.tesseract = tesseract
        ctx.diff_input_paths = diff_input_paths
        # Wire names address the endpoint; concrete paths rebuild the pytree.
        ctx.diff_input_wires = diff_input_wires

        # This conversion is also the torch.func rejection guard: under those
        # transforms the tensors arrive storage-less and it raises a documented
        # RuntimeError. Keep it ahead of anything that reads the holder, which
        # is unpopulated on that path and would surface a bare IndexError first.
        saved_inputs: dict[KeyType, Any] = dict(static_inputs)
        for path, tensor in zip(diff_input_paths, tensors, strict=True):
            saved_inputs[path] = _tensor_to_numpy(tensor)
        ctx.saved_inputs = saved_inputs

        # Output names are resolved in forward(), since a dict-valued output
        # has no concrete keys until apply() has returned.
        ctx.diff_output_wires = [wire for _, wire in holder[1]]

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Reverse-mode AD via the Tesseract's VJP endpoint."""
        # With set_materialize_grads(False) an output the loss did not use
        # arrives as None. Request the VJP only for the outputs that carry an
        # incoming cotangent, so a seam with several differentiable outputs is
        # never asked to compute (and we never pay to transport) a gradient
        # that would be multiplied by zero.
        active_wires: list[str] = []
        cotangent_vector: dict[str, Any] = {}
        for wire, grad in zip(ctx.diff_output_wires, grad_outputs, strict=True):
            if grad is None:
                continue
            active_wires.append(wire)
            cotangent_vector[wire] = _tensor_to_numpy(grad)

        # No output carried a cotangent: the Tesseract cannot contribute any
        # input gradient, so skip the VJP call entirely and return None for
        # every input. Autograd normally prunes a node whose outputs are all
        # off the backward path before calling backward(), so this is a
        # defensive guard rather than a path a normal .backward() reaches.
        if not active_wires:
            # None for the seven non-tensor arguments, then one per input.
            return (None,) * (7 + len(ctx.diff_input_wires))

        vjp_result = ctx.tesseract.vector_jacobian_product(
            inputs=_unflatten_pytree(ctx.saved_inputs),
            vjp_inputs=list(ctx.diff_input_wires),
            vjp_outputs=active_wires,
            cotangent_vector=cotangent_vector,
        )

        grad_inputs: list[torch.Tensor | None] = []
        for wire in ctx.diff_input_wires:
            g = vjp_result.get(wire)
            grad_inputs.append(_to_tensor(g) if g is not None else None)

        # None for (tesseract, diff_input_paths, diff_input_wires,
        #           diff_output_templates, all_paths, static_inputs, holder)
        return (None, None, None, None, None, None, None, *grad_inputs)

    @staticmethod
    def jvp(
        ctx: Any,
        *tangents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Forward-mode AD via the Tesseract's JVP endpoint."""
        # tangents: (tesseract, diff_input_paths, diff_input_wires,
        #            diff_output_templates, all_paths, static_inputs, holder,
        #            *tensor_tangents)
        tensor_tangents = tangents[7:]

        tangent_vector: dict[str, Any] = {}
        jvp_inputs: list[str] = []
        for wire, t in zip(ctx.diff_input_wires, tensor_tangents, strict=True):
            if t is not None:
                tangent_vector[wire] = _tensor_to_numpy(t)
                jvp_inputs.append(wire)

        # apply_tesseract only routes tensors on a differentiable path into the
        # autograd Function, so torch calls jvp only when at least one carries a
        # forward tangent. jvp_inputs is therefore never empty here.
        assert jvp_inputs, "jvp called with no forward tangents"

        jvp_result = ctx.tesseract.jacobian_vector_product(
            inputs=_unflatten_pytree(ctx.saved_inputs),
            jvp_inputs=jvp_inputs,
            jvp_outputs=list(ctx.diff_output_wires),
            tangent_vector=tangent_vector,
        )

        return tuple(_to_tensor(jvp_result[wire]) for wire in ctx.diff_output_wires)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_tesseract(
    tesseract: Tesseract,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Call a Tesseract as a differentiable PyTorch operation.

    Infers which inputs/outputs are differentiable from the Tesseract's schema.
    Torch tensors provided for differentiable fields participate in autograd;
    all other values are passed through as static inputs.

    Supports both reverse-mode (``.backward()``) and forward-mode
    (``torch.autograd.forward_ad``) differentiation.

    Args:
        tesseract: A Tesseract instance.
        inputs: Nested dict matching the Tesseract's input schema.  Provide
            ``torch.Tensor`` for array fields you want gradients through,
            and plain Python / NumPy values for everything else.

    Returns:
        Nested dict matching the Tesseract's output schema, with
        differentiable array outputs as ``torch.Tensor`` (with ``grad_fn``
        when inputs require grad) and non-differentiable outputs as-is
        (NumPy arrays or scalars).

    Example::

        # Flat schema
        result = apply_tesseract(quadratic, {"x": x, "A": A, "b": b})
        result["y"].sum().backward()

        # Nested schema
        result = apply_tesseract(meshstats, {
            "mesh": {"n_points": 3, ..., "points": points_tensor}
        })
        result["statistics"]["barycenter"].sum().backward()
    """
    openapi = tesseract.openapi_schema
    diff_in_paths = _get_differentiable_arrays(openapi, "ApplyInputSchema")
    diff_out_paths = _get_differentiable_arrays(openapi, "ApplyOutputSchema")
    diff_out_templates = sorted(diff_out_paths)

    # All known dotted paths guide pytree flattening so we recurse into
    # sub-models but not into opaque dict fields.
    all_paths = diff_in_paths | diff_out_paths

    flat_inputs = _flatten_pytree(inputs, recurse_into=all_paths)

    # Partition into differentiable tensors vs static values. A declared path
    # may be a template, so match rather than compare: ``params.{}`` covers the
    # concrete leaf ``params.p`` and is addressed on the wire as ``params.{p}``.
    diff_paths: list[KeyType] = []
    diff_wires: list[str] = []
    diff_tensors: list[torch.Tensor] = []
    static: dict[KeyType, Any] = {}

    for path, value in flat_inputs:
        wire = (
            _wire_name(path, diff_in_paths) if isinstance(value, torch.Tensor) else None
        )
        if wire is not None:
            diff_paths.append(path)
            diff_wires.append(wire)
            diff_tensors.append(value)
        elif isinstance(value, torch.Tensor):
            static[path] = _tensor_to_numpy(value)
        else:
            static[path] = value

    # Mutable holder so forward() can pass the full result dict back to us
    # without going through autograd's return values.
    result_holder: list[Any] = []

    output_tensors = _TesseractFunction.apply(
        tesseract,
        diff_paths,
        diff_wires,
        diff_out_templates,
        all_paths,
        static,
        result_holder,
        *diff_tensors,
    )

    # Reconstruct full output pytree. Names were resolved inside forward(),
    # since a dict-valued output has no concrete keys until apply() returns.
    flat_result = dict(result_holder[0])
    resolved_outputs = result_holder[1]
    for (concrete, _wire), tensor in zip(resolved_outputs, output_tensors, strict=True):
        flat_result[concrete] = tensor

    return _unflatten_pytree(flat_result)
