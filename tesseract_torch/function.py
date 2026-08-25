# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable PyTorch wrapper for Tesseract operations.

This module registers a Tesseract as a first-class differentiable primitive in
PyTorch's autograd graph.  The forward pass dispatches to ``tesseract.apply()``,
the backward pass to ``tesseract.vector_jacobian_product()``, and the
forward-mode JVP to ``tesseract.jacobian_vector_product()``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

import numpy as np
import torch
from tesseract_core import Tesseract


def _supports_cuda_ipc(tesseract: Tesseract) -> bool:
    """Whether ``tesseract``'s client can be switched to ``cuda_ipc`` mode.

    True only for an ``HTTPClient`` (has ``_output_format``) -- e.g.
    ``LocalClient`` already shares process memory and has no ``cuda_ipc``
    concept, so a CUDA tensor handed to it raw would reach its in-process
    endpoint code untouched and fail there instead of being exported by IPC
    handle.
    """
    client = getattr(tesseract, "_client", None)
    return client is not None and hasattr(client, "_output_format")


@contextlib.contextmanager
def _cuda_ipc_mode(tesseract: Tesseract) -> Generator[None]:
    """Temporarily switch a served Tesseract's HTTP client to ``cuda_ipc`` mode.

    Caller must check :func:`_supports_cuda_ipc` first; this assumes it does.

    Scoped to a single call so a Tesseract shared across CPU-tensor and
    CUDA-tensor calls is not permanently switched over; mirrors
    ``tesseract_jax.tesseract_compat.Jaxeract.cuda_ipc``.
    """
    client = tesseract._client
    session = getattr(client, "_session", None)
    prev_fmt = client._output_format
    had_accept = session is not None and "Accept" in session.headers
    prev_accept = session.headers.get("Accept") if session is not None else None

    client._output_format = "json+cuda_ipc"
    if session is not None:
        session.headers["Accept"] = "application/json+cuda_ipc"
    try:
        yield
    finally:
        client._output_format = prev_fmt
        if session is not None:
            if had_accept:
                session.headers["Accept"] = prev_accept
            else:
                session.headers.pop("Accept", None)


def _to_tensor(arr: Any) -> torch.Tensor:
    """Convert a decoded Tesseract array to a tensor, copying if read-only.

    ``arr`` is a NumPy array (host encodings), an ``IpcDeviceArray``
    (``cuda_ipc`` encoding, a fresh device buffer owned by this process,
    adopted zero-copy via DLPack), or already a ``torch.Tensor`` (a saved
    CUDA input echoed back untouched, e.g. the JVP no-tangent shortcut).

    The DLPack branch is gated on ``__cuda_array_interface__`` rather than
    ``__dlpack__`` alone: plain ``np.ndarray`` also implements ``__dlpack__``
    (since NumPy 1.22), and routing a *read-only* one through it fails
    (DLPack has no way to signal read-only to an older consumer), where the
    ``np.asarray``/copy fallback below handles it correctly.
    """
    if isinstance(arr, torch.Tensor):
        return arr
    if hasattr(arr, "__cuda_array_interface__"):
        return torch.utils.dlpack.from_dlpack(arr)
    a = np.asarray(arr)
    if not a.flags.writeable:
        a = a.copy()
    return torch.as_tensor(a)


def _tensor_to_numpy_or_cuda(t: torch.Tensor, *, cuda_ipc: bool = False) -> Any:
    """Convert a torch tensor to a numpy array, or pass a CUDA tensor through.

    A CUDA tensor already exposes ``__cuda_array_interface__``, so when
    ``cuda_ipc`` is enabled for this call it can be handed to the Tesseract
    client as-is and exported by IPC handle instead of copied to host. It
    must be contiguous, since ``cuda_ipc`` transfers a flat byte range with
    no strides. Without ``cuda_ipc``, a CUDA tensor still needs the host copy
    below: the client's default (non-``cuda_ipc``) encoder calls
    ``np.asanyarray`` on it, which cannot read GPU memory.

    torch.func transforms (vjp, jvp, grad, vmap) wrap tensors in a C++
    FunctionalTensorWrapper that has no backing storage.  These tensors
    report type(t)==torch.Tensor (no Python subclass), so there is no
    isinstance check we can use.  Instead we probe data_ptr(), the same
    public precondition that .numpy() relies on, to raise an actionable
    error instead of the confusing default message ("Cannot access data
    pointer of Tensor that doesn't have storage").
    """
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
    if cuda_ipc and t.is_cuda:
        return t.detach().contiguous()
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
    tree: dict[str, Any],
    prefix: str = "",
    *,
    recurse_into: set[str] | None = None,
) -> list[tuple[str, Any]]:
    """Flatten a nested dict into ``(dotted_path, leaf_value)`` pairs.

    Only recurses into sub-dicts whose dotted prefix is a strict prefix of at
    least one path in *recurse_into*.  All other dicts are treated as opaque
    leaf values (e.g. ``dict[str, Array]`` schema fields).

    If *recurse_into* is ``None``, every nested dict is recursed into.
    """
    items: list[tuple[str, Any]] = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and _should_recurse(path, value, recurse_into):
            items.extend(_flatten_pytree(value, path, recurse_into=recurse_into))
        else:
            items.append((path, value))
    return items


def _should_recurse(
    path: str,
    value: dict,
    known_paths: set[str] | None,
) -> bool:
    """Return True when *path* is a prefix of a known leaf path."""
    if not value:
        return False
    if known_paths is None:
        return True
    dot_prefix = path + "."
    return any(p.startswith(dot_prefix) for p in known_paths)


# ---------------------------------------------------------------------------
# Wildcard paths
# ---------------------------------------------------------------------------
#
# tesseract-core encodes a container-valued differentiable field as a template
# rather than a concrete path: ``dict[str, Differentiable[...]]`` is declared as
# ``params.{}``. The wire name the runtime accepts puts the concrete key in the
# braces, ``params.{p}``, because its own path regex compiles the sentinel to
# ``\{[\w \-]+\}``. Neither ``params.p`` nor ``params.{}`` is accepted.
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


def _wire_name(concrete_path: str, templates: set[str]) -> str | None:
    """Map a concrete leaf path to the name the Tesseract expects.

    Returns ``None`` when no declared path covers it, which is how the caller
    tells a differentiable leaf from a static one.
    """
    if concrete_path in templates:
        return concrete_path

    concrete_parts = concrete_path.split(".")
    for template in templates:
        template_parts = template.split(".")
        if len(template_parts) != len(concrete_parts):
            continue
        resolved: list[str] = []
        for tpl, concrete in zip(template_parts, concrete_parts, strict=True):
            if tpl == _DICT_WILDCARD:
                resolved.append("{" + concrete + "}")
            elif tpl == _LIST_WILDCARD:
                # Handled up front by _reject_list_wildcards; a list field is
                # an opaque leaf here, so this branch is defensive only.
                return None
            elif tpl != concrete:
                break
            else:
                resolved.append(concrete)
        else:
            return ".".join(resolved)
    return None


def _reject_list_wildcards(templates: set[str], where: str) -> None:
    """Raise a clear error for list-valued differentiable fields.

    ``_flatten_pytree`` only descends into dicts, so a list field arrives as
    one opaque leaf and its tensors are never registered for autograd. Left
    alone that surfaces as "element 0 of tensors does not require grad", which
    says nothing about the cause.
    """
    offenders = sorted(t for t in templates if _LIST_WILDCARD in t.split("."))
    if offenders:
        raise NotImplementedError(
            f"List-valued differentiable {where} are not supported yet: "
            f"{', '.join(offenders)}. Use a dict-valued field, or keep the "
            f"list entries as separate schema fields."
        )


def _resolve_output_names(
    flat_result: dict[str, Any], templates: list[str]
) -> list[tuple[str, str]]:
    """Pair each concrete output leaf with its wire name, in a stable order.

    Concrete keys of a dict-valued output are not knowable until ``apply``
    returns, so the differentiable output list is resolved here rather than
    derived from the schema up front.
    """
    resolved: list[tuple[str, str]] = []
    for concrete in sorted(flat_result):
        wire = _wire_name(concrete, set(templates))
        if wire is not None:
            resolved.append((concrete, wire))
    return resolved


def _unflatten_pytree(flat: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a nested dict from ``{dotted_path: value}``."""
    tree: dict[str, Any] = {}
    for path, value in flat.items():
        parts = path.split(".")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


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
        diff_input_paths: list[str],
        diff_input_wires: list[str],
        diff_output_templates: list[str],
        all_paths: set[str],
        static_inputs: dict[str, Any],
        cuda_ipc: bool,
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
        cuda_ipc_active = cuda_ipc and _supports_cuda_ipc(tesseract)
        flat_inputs = dict(static_inputs)
        for path, tensor in zip(diff_input_paths, tensors, strict=True):
            flat_inputs[path] = _tensor_to_numpy_or_cuda(
                tensor, cuda_ipc=cuda_ipc_active
            )

        with _cuda_ipc_mode(tesseract) if cuda_ipc_active else contextlib.nullcontext():
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
        (
            tesseract,
            diff_input_paths,
            diff_input_wires,
            diff_output_templates,  # noqa: RUF059
            all_paths,  # noqa: RUF059
            static_inputs,
            cuda_ipc,
            holder,
            *tensors,
        ) = inputs
        ctx.tesseract = tesseract
        ctx.diff_input_paths = diff_input_paths
        # Wire names address the endpoint; concrete paths rebuild the pytree.
        ctx.diff_input_wires = diff_input_wires
        # Resolved once here (not the raw request flag): a LocalClient can't
        # act on cuda_ipc, so backward()/jvp() must fall back to the host
        # copy for it exactly as forward() did, not retry passing GPU memory
        # to code that cannot read it.
        ctx.cuda_ipc = cuda_ipc and _supports_cuda_ipc(tesseract)

        # Each input tensor's own device, in ctx.diff_input_wires order.
        # Autograd requires the gradient backward() returns for an input to
        # live on that same input's device, regardless of what device the
        # Tesseract's VJP happens to compute/return on (e.g. always host
        # without cuda_ipc) -- backward() uses this to move each decoded
        # gradient back before returning it.
        ctx.diff_input_devices = [tensor.device for tensor in tensors]

        # This conversion is also the torch.func rejection guard: under those
        # transforms the tensors arrive storage-less and it raises a documented
        # RuntimeError. Keep it ahead of anything that reads the holder, which
        # is unpopulated on that path and would surface a bare IndexError first.
        saved_inputs: dict[str, Any] = dict(static_inputs)
        for path, tensor in zip(diff_input_paths, tensors, strict=True):
            saved_inputs[path] = _tensor_to_numpy_or_cuda(tensor, cuda_ipc=ctx.cuda_ipc)
        ctx.saved_inputs = saved_inputs

        # Output names are resolved in forward(), since a dict-valued output
        # has no concrete keys until apply() has returned.
        ctx.diff_output_wires = [wire for _, wire in holder[1]]

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        """Reverse-mode AD via the Tesseract's VJP endpoint."""
        cotangent_vector = {
            wire: _tensor_to_numpy_or_cuda(grad, cuda_ipc=ctx.cuda_ipc)
            for wire, grad in zip(ctx.diff_output_wires, grad_outputs, strict=True)
        }

        with (
            _cuda_ipc_mode(ctx.tesseract) if ctx.cuda_ipc else contextlib.nullcontext()
        ):
            vjp_result = ctx.tesseract.vector_jacobian_product(
                inputs=_unflatten_pytree(ctx.saved_inputs),
                vjp_inputs=list(ctx.diff_input_wires),
                vjp_outputs=list(ctx.diff_output_wires),
                cotangent_vector=cotangent_vector,
            )

        grad_inputs: list[torch.Tensor | None] = []
        for wire, device in zip(
            ctx.diff_input_wires, ctx.diff_input_devices, strict=True
        ):
            g = vjp_result.get(wire)
            grad_inputs.append(_to_tensor(g).to(device) if g is not None else None)

        # None for (tesseract, diff_input_paths, diff_input_wires,
        #           diff_output_templates, all_paths, static_inputs, cuda_ipc, holder)
        return (None, None, None, None, None, None, None, None, *grad_inputs)

    @staticmethod
    def jvp(
        ctx: Any,
        *tangents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Forward-mode AD via the Tesseract's JVP endpoint."""
        # tangents: (tesseract, diff_input_paths, diff_input_wires,
        #            diff_output_templates, all_paths, static_inputs, cuda_ipc,
        #            holder, *tensor_tangents)
        tensor_tangents = tangents[8:]

        tangent_vector: dict[str, Any] = {}
        jvp_inputs: list[str] = []
        for wire, t in zip(ctx.diff_input_wires, tensor_tangents, strict=True):
            if t is not None:
                tangent_vector[wire] = _tensor_to_numpy_or_cuda(
                    t, cuda_ipc=ctx.cuda_ipc
                )
                jvp_inputs.append(wire)

        if not jvp_inputs:
            return tuple(
                torch.zeros_like(_to_tensor(ctx.saved_inputs.get(path, 0.0)))
                for path in ctx.diff_input_paths[: len(ctx.diff_output_wires)]
            )

        with (
            _cuda_ipc_mode(ctx.tesseract) if ctx.cuda_ipc else contextlib.nullcontext()
        ):
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
    *,
    cuda_ipc: bool = False,
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
        cuda_ipc: If ``True``, CUDA tensors are exchanged with the Tesseract
            via CUDA IPC handles instead of a host round-trip, so array data
            never leaves the GPU. Requires a served Tesseract (``HTTPClient``)
            started with ``enable_experimental_cuda_ipc=True`` in its
            ``runtime_config`` and GPU access (e.g.
            ``Tesseract.from_image(..., gpus=["all"])``); has no effect on a
            local (in-process) client, which already shares memory. Both
            processes must share the CUDA IPC namespace (Docker's
            ``--ipc=host``). CPU tensors and plain NumPy inputs are
            unaffected either way. This is an experimental tesseract-core
            feature; see ``tesseract_core.runtime.cuda_ipc``.

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
    _reject_list_wildcards(diff_in_paths, "inputs")
    _reject_list_wildcards(diff_out_paths, "outputs")
    diff_out_templates = sorted(diff_out_paths)

    # All known dotted paths guide pytree flattening so we recurse into
    # sub-models but not into opaque dict fields.
    all_paths = diff_in_paths | diff_out_paths

    flat_inputs = _flatten_pytree(inputs, recurse_into=all_paths)

    # Resolved once (not the raw request flag): a LocalClient can't act on
    # cuda_ipc, so a CUDA tensor must still take the host round-trip for it.
    cuda_ipc_active = cuda_ipc and _supports_cuda_ipc(tesseract)

    # Partition into differentiable tensors vs static values. A declared path
    # may be a template, so match rather than compare: ``params.{}`` covers the
    # concrete leaf ``params.p`` and is addressed on the wire as ``params.{p}``.
    diff_paths: list[str] = []
    diff_wires: list[str] = []
    diff_tensors: list[torch.Tensor] = []
    static: dict[str, Any] = {}

    for path, value in flat_inputs:
        wire = (
            _wire_name(path, diff_in_paths) if isinstance(value, torch.Tensor) else None
        )
        if wire is not None:
            diff_paths.append(path)
            diff_wires.append(wire)
            diff_tensors.append(value)
        elif isinstance(value, torch.Tensor):
            static[path] = _tensor_to_numpy_or_cuda(value, cuda_ipc=cuda_ipc_active)
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
        cuda_ipc,
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
