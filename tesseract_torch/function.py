# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable PyTorch wrapper for Tesseract operations.

This module registers a Tesseract as a first-class differentiable primitive in
PyTorch's autograd graph.  The forward pass dispatches to ``tesseract.apply()``,
the backward pass to ``tesseract.vector_jacobian_product()``, and the
forward-mode JVP to ``tesseract.jacobian_vector_product()``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, get_args

import numpy as np
import torch
from tesseract_core import Tesseract

# A leaf's path, one entry per schema level. Kept as segments rather than a
# dotted string because a dict key is free to contain a dot. An int segment is
# a list position, which is what keeps it apart from a dict key that merely
# looks like one: {"0": v} carries "0" here, [v] carries 0.
type KeyType = tuple[str | int, ...]


def _supported_transports() -> frozenset[str]:
    """On-device transports, read from tesseract-core's ``gpu_transport`` enum.

    Read from the enum so the two can't drift; its ``"none"`` means host
    round-trip and is handled separately. Imported on first use because
    ``tesseract_core.runtime`` needs the ``tesseract-core[runtime]`` extra,
    which only calls that use a GPU transport need.
    """
    from tesseract_core.runtime.config import gpu_transport_type

    return frozenset(get_args(gpu_transport_type)) - {"none"}


def _target_device(tensors: Sequence[torch.Tensor]) -> torch.device:
    """The device a call's outputs go to: that of its first CUDA tensor, else the CPU.

    Outputs land there whichever transport the call uses, so that whether GPU
    arrays could stay on the device never changes the result.
    """
    for tensor in tensors:
        if tensor.is_cuda:
            return tensor.device
    return torch.device("cpu")


def _resolve_gpu_transport(tesseract: Tesseract, device: torch.device) -> str:
    """Return the transport a call uses, either ``"none"`` or a device transport.

    Only a call with CUDA tensors has GPU arrays to exchange, so only such a call
    asks the Tesseract which transport to use (see
    ``Tesseract.resolve_gpu_transport``), which may check that it works.
    """
    if device.type != "cuda":
        return "none"
    gpu_transport = tesseract.resolve_gpu_transport()
    if gpu_transport == "none":
        return gpu_transport
    supported = _supported_transports()
    if gpu_transport not in supported:
        raise ValueError(
            f"The Tesseract requests gpu_transport={gpu_transport!r}, which "
            f"tesseract-torch cannot drive (supported: "
            f"{['none', *sorted(supported)]}). Pass "
            "tesseract.with_encoding(gpu_transport='none') to apply_tesseract to "
            "copy CUDA tensors to the host instead."
        )
    return gpu_transport


_VMAP_METHODS = ("sequential", "expand_dims", "broadcast_all")


def _validate_vmap_method(vmap_method: str | None) -> None:
    """Reject a batching strategy the ``torch.vmap`` rule does not implement."""
    if vmap_method is not None and vmap_method not in _VMAP_METHODS:
        raise ValueError(
            f"Unsupported vmap_method {vmap_method!r}; "
            f"supported: {list(_VMAP_METHODS)}."
        )


def _with_gpu_transport(tesseract: Tesseract, gpu_transport: str) -> Tesseract:
    """``tesseract``, or a view of it, whose calls use ``gpu_transport``.

    A device transport sends CUDA tensors by reference and asks for outputs the
    same way; ``"none"`` keeps both on the host. An in-process Tesseract encodes
    nothing, so it is returned as it is, as is one that already requests the
    transport. Taken per call, so an autograd graph keeps working after the
    Tesseract is served again.
    """
    if tesseract.server_capabilities is None:
        return tesseract
    if (tesseract.current_encoding.gpu_transport or "none") == gpu_transport:
        return tesseract
    return tesseract.with_encoding(gpu_transport=gpu_transport)


def _device_arrays_to_numpy(value: Any) -> Any:
    """Copy the device arrays a response decoded into ``value`` to NumPy arrays.

    A non-differentiable array output comes back as a NumPy array whichever
    transport the call used, so it does not depend on whether GPU arrays could
    stay on the device. Tensors an in-process endpoint returns are left alone.
    """
    if isinstance(value, dict):
        return {key: _device_arrays_to_numpy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_device_arrays_to_numpy(item) for item in value)
    if hasattr(value, "__cuda_array_interface__") and not isinstance(
        value, torch.Tensor
    ):
        return _to_tensor(value).cpu().numpy()
    return value


def _to_tensor(arr: Any) -> torch.Tensor:
    """Convert a decoded Tesseract array to a tensor, copying if read-only.

    ``arr`` is a NumPy array (host encodings) or an ``IpcDeviceArray``
    (``cuda_ipc`` encoding, a fresh device buffer owned by this process,
    adopted zero-copy via DLPack). A ``torch.Tensor`` is passed through
    untouched, should an endpoint ever echo one back verbatim.

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


def _is_zero_or_none(t: torch.Tensor | None) -> bool:
    """True for a missing or all-zero (co)tangent. NaN counts as nonzero."""
    return t is None or not bool(t.any())


def _tensor_to_numpy_or_cuda(t: torch.Tensor, *, on_device: bool = False) -> Any:
    """Convert a torch tensor to a numpy array, or pass a CUDA tensor through.

    A CUDA tensor already exposes ``__cuda_array_interface__``, so when a device
    transport is active for this call it can be handed to the Tesseract client
    as-is and exported by IPC handle instead of copied to host. It must be
    contiguous, since the transport moves a flat byte range with no strides.
    Without a device transport, a CUDA tensor still needs the host copy below:
    the client's default encoder calls ``np.asanyarray`` on it, which cannot
    read GPU memory.

    torch.func transforms (vjp, jvp, grad) wrap tensors in a C++
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
    if on_device and t.is_cuda:
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


def _holds_array(value: Any) -> bool:
    """Whether a tensor or NumPy array sits anywhere inside *value*."""
    if isinstance(value, torch.Tensor | np.ndarray):
        return True
    return any(_holds_array(child) for _, child in _children(value) or ())


def _open_array_containers(path: KeyType, value: Any) -> list[tuple[KeyType, Any]]:
    """Split a dict or list that holds arrays into its leaves.

    A container the schema does not mark differentiable stays whole when
    flattened, which would hide its arrays from ``torch.vmap``. Tuples and
    containers without arrays are kept as they are.
    """
    if not isinstance(value, dict | list) or not _holds_array(value):
        return [(path, value)]
    return [
        item
        for key, child in _children(value)
        for item in _open_array_containers((*path, key), child)
    ]


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

# Number of non-tensor arguments forward() takes ahead of *tensors. backward()
# and jvp() see one slot per forward() argument and skip these.
_N_NON_TENSOR_ARGS = 2


@dataclass(frozen=True)
class _DispatchParams:
    """Everything a Tesseract call needs besides its input tensors.

    Bundled so the autograd function and its ``vmap`` rule take
    ``(params, slot, *tensors)`` rather than a long positional list.

    Attributes:
        tesseract: The Tesseract to call.
        tensor_paths: Input path of each tensor argument, the differentiable
            ones first.
        diff_input_wires: Wire name of each differentiable tensor argument.
        diff_output_templates: Declared paths of the differentiable outputs.
        all_paths: Every declared differentiable path, to guide flattening.
        static_inputs: The input leaves that are not tensors.
        device: Where the call's array outputs go (see :func:`_target_device`).
        gpu_transport: Resolved transport, ``"none"`` or a device transport.
        vmap_method: Batching strategy under ``torch.vmap``, or ``None``.
    """

    tesseract: Tesseract
    tensor_paths: list[KeyType]
    diff_input_wires: list[str]
    diff_output_templates: list[str]
    all_paths: set[str]
    static_inputs: dict[KeyType, Any]
    device: torch.device
    gpu_transport: str
    vmap_method: str | None


@dataclass
class _ResultSlot:
    """Where the autograd function leaves the full result for its caller.

    An object rather than a list: ``torch.func`` transforms rebuild list
    arguments on the way in, so an append to one never reaches the caller.

    ``outputs`` pairs the path of each returned tensor with its wire name, or
    ``None`` for a non-differentiable output batched under ``torch.vmap``.
    """

    flat_result: dict[KeyType, Any] = field(default_factory=dict)
    outputs: list[tuple[KeyType, str | None]] = field(default_factory=list)


class _TesseractFunction(torch.autograd.Function):
    """Low-level autograd function wrapping a Tesseract.

    This is an implementation detail.  Users should call :func:`apply_tesseract`.
    """

    @staticmethod
    def forward(
        params: _DispatchParams,
        slot: _ResultSlot,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the Tesseract forward pass, returning differentiable outputs.

        The full (flat) result dict and the resolved output names are stashed
        in *slot* so the caller can reconstruct non-differentiable outputs
        without a second ``apply()`` call.

        Output names are resolved here rather than passed in: a dict-valued
        differentiable output is declared as a template, and its concrete keys
        only exist once ``apply`` has returned.
        """
        tesseract = params.tesseract
        on_device = params.gpu_transport != "none"
        flat_inputs = dict(params.static_inputs)
        for path, tensor in zip(params.tensor_paths, tensors, strict=True):
            flat_inputs[path] = _tensor_to_numpy_or_cuda(tensor, on_device=on_device)

        result = _with_gpu_transport(tesseract, params.gpu_transport).apply(
            _unflatten_pytree(flat_inputs)
        )
        flat_result = dict(_flatten_pytree(result, recurse_into=params.all_paths))

        resolved_outputs = _resolve_output_names(
            flat_result, params.diff_output_templates
        )

        # Stash full result + resolved output names for the caller
        slot.flat_result = flat_result
        slot.outputs = resolved_outputs

        return tuple(
            _to_tensor(flat_result[c]).to(params.device) for c, _ in resolved_outputs
        )

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> None:
        """Save forward-pass metadata for use in backward / jvp."""
        # Let cotangents of unused outputs arrive as None in backward().
        ctx.set_materialize_grads(False)
        params, slot, *tensors = inputs
        ctx.tesseract = params.tesseract
        # Wire names address the endpoint; concrete paths rebuild the pytree.
        ctx.diff_input_wires = params.diff_input_wires
        # Already resolved by apply_tesseract, so backward()/jvp() talk to the
        # Tesseract exactly as forward() did.
        ctx.gpu_transport = params.gpu_transport
        ctx.on_device = params.gpu_transport != "none"
        ctx.num_tensors = len(tensors)

        # Each input tensor's own device, in ctx.diff_input_wires order.
        # Autograd requires the gradient backward() returns for an input to
        # live on that same input's device, regardless of what device the
        # Tesseract's VJP happens to compute/return on (e.g. always host on a
        # host round-trip) -- backward() uses this to move each decoded gradient
        # back before returning it.
        ctx.diff_input_devices = [
            tensor.device for tensor in tensors[: len(params.diff_input_wires)]
        ]

        saved_inputs: dict[KeyType, Any] = dict(params.static_inputs)
        for path, tensor in zip(params.tensor_paths, tensors, strict=True):
            saved_inputs[path] = _tensor_to_numpy_or_cuda(
                tensor, on_device=ctx.on_device
            )
        ctx.saved_inputs = saved_inputs

        # Output names are resolved in forward(), since a dict-valued output
        # has no concrete keys until apply() has returned.
        ctx.diff_output_wires = [wire for _, wire in slot.outputs]

        # Shape, dtype, and device of each output, in diff_output_wires order.
        # jvp() uses these to place tangents on their output's device and to
        # build zero tangents. Storing the outputs themselves would create a
        # reference cycle through their grad_fn.
        ctx.diff_output_specs = [(out.shape, out.dtype, out.device) for out in outputs]

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Reverse-mode AD via the Tesseract's VJP endpoint."""
        # Request the VJP only for outputs with a nonzero cotangent. An unused
        # output arrives as None, and any output may arrive as all zeros.
        # Requesting either would compute and transport a gradient that is
        # multiplied by zero, which is costly for a finite-difference VJP that
        # pays per output.
        cotangent_vector = {
            wire: _tensor_to_numpy_or_cuda(grad, on_device=ctx.on_device)
            for wire, grad in zip(ctx.diff_output_wires, grad_outputs, strict=True)
            if not _is_zero_or_none(grad)
        }
        active_wires = list(cotangent_vector)
        if not active_wires:
            return (None,) * (_N_NON_TENSOR_ARGS + ctx.num_tensors)

        # Request gradients only for inputs that require grad, since a
        # finite-difference VJP pays per input element. Every tensor on a
        # differentiable path reaches forward() because it may carry a
        # forward-mode tangent, so some of them need no gradient here.
        # needs_input_grad reflects requires_grad at forward time, so an input
        # left out of autograd.grad(inputs=...) is still requested.
        needed_wires = [
            wire
            for wire, needed in zip(
                ctx.diff_input_wires,
                ctx.needs_input_grad[
                    _N_NON_TENSOR_ARGS : _N_NON_TENSOR_ARGS + len(ctx.diff_input_wires)
                ],
                strict=True,
            )
            if needed
        ]

        vjp_result = _with_gpu_transport(
            ctx.tesseract, ctx.gpu_transport
        ).vector_jacobian_product(
            inputs=_unflatten_pytree(ctx.saved_inputs),
            vjp_inputs=needed_wires,
            vjp_outputs=active_wires,
            cotangent_vector=cotangent_vector,
        )

        # Inputs left out of the request get None, which autograd treats as a
        # zero gradient.
        grad_inputs: list[torch.Tensor | None] = []
        for wire, device in zip(
            ctx.diff_input_wires, ctx.diff_input_devices, strict=True
        ):
            g = vjp_result.get(wire)
            grad_inputs.append(_to_tensor(g).to(device) if g is not None else None)

        # Non-differentiable tensors follow the differentiable ones and get
        # no gradient.
        return (
            *(None,) * _N_NON_TENSOR_ARGS,
            *grad_inputs,
            *(None,) * (ctx.num_tensors - len(grad_inputs)),
        )

    @staticmethod
    def jvp(
        ctx: Any,
        *tangents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Forward-mode AD via the Tesseract's JVP endpoint."""
        tensor_tangents = tangents[
            _N_NON_TENSOR_ARGS : _N_NON_TENSOR_ARGS + len(ctx.diff_input_wires)
        ]

        # Request the JVP only for inputs with a nonzero tangent. Inputs sliced
        # from a one-hot-seeded dual tensor, as when building a Jacobian column
        # by column, arrive with all-zero tangents rather than None.
        tangent_vector = {
            wire: _tensor_to_numpy_or_cuda(t, on_device=ctx.on_device)
            for wire, t in zip(ctx.diff_input_wires, tensor_tangents, strict=True)
            if not _is_zero_or_none(t)
        }
        jvp_inputs = list(tangent_vector)

        # All input tangents are zero, so all output tangents are too. Torch
        # rejects None from jvp(), so return explicit zeros without calling
        # the Tesseract.
        if not jvp_inputs:
            return tuple(
                torch.zeros(shape, dtype=dtype, device=device)
                for shape, dtype, device in ctx.diff_output_specs
            )

        jvp_result = _with_gpu_transport(
            ctx.tesseract, ctx.gpu_transport
        ).jacobian_vector_product(
            inputs=_unflatten_pytree(ctx.saved_inputs),
            jvp_inputs=jvp_inputs,
            jvp_outputs=list(ctx.diff_output_wires),
            tangent_vector=tangent_vector,
        )

        return tuple(
            _to_tensor(jvp_result[wire]).to(device)
            for wire, (_, _, device) in zip(
                ctx.diff_output_wires, ctx.diff_output_specs, strict=True
            )
        )

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple[int | None, ...],
        params: _DispatchParams,
        slot: _ResultSlot,
        *tensors: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        """Batching rule for ``torch.vmap``, following ``params.vmap_method``.

        ``"sequential"`` calls the Tesseract once per batch element.
        ``"expand_dims"`` and ``"broadcast_all"`` call it once, after giving
        every unbatched array input a leading dimension of size 1 or of the
        batch size respectively.

        Every array output comes back batched along dim 0, the
        non-differentiable ones as tensors too, since a NumPy array cannot
        carry the batch dimension. Other outputs are returned once.
        """
        method = params.vmap_method
        if method is None:
            raise NotImplementedError(
                "torch.vmap over apply_tesseract needs a batching strategy. Pass "
                f"vmap_method (one of {list(_VMAP_METHODS)}) to apply_tesseract."
            )
        batch_size = info.batch_size
        dims = in_dims[_N_NON_TENSOR_ARGS:]
        if method == "sequential" and batch_size == 0:
            raise ValueError(
                'vmap_method="sequential" cannot batch over an empty batch, since '
                "it takes the outputs from the per-element calls."
            )
        if method == "sequential":
            calls = []
            for i in range(batch_size):
                sliced = [
                    t if d is None else t.select(d, i)
                    for t, d in zip(tensors, dims, strict=True)
                ]
                calls.append(_call_leaves(params, sliced))
            batched = {
                path: _stack_leaf(path, [call[path] for call in calls])
                for path in calls[0]
            }
        else:
            moved = [
                _with_batch_dim(t, method, batch_size) if d is None else t.movedim(d, 0)
                for t, d in zip(tensors, dims, strict=True)
            ]
            static = {
                path: _with_batch_dim(value, method, batch_size)
                for path, value in params.static_inputs.items()
            }
            leaves = _call_leaves(replace(params, static_inputs=static), moved)
            batched = {
                path: _expand_leaf(path, value, batch_size, method)
                for path, value in leaves.items()
            }

        slot.flat_result = batched
        slot.outputs = [
            (path, None)
            for path, value in batched.items()
            if isinstance(value, torch.Tensor)
        ]
        output_tensors = tuple(batched[path] for path, _ in slot.outputs)
        return output_tensors, (0,) * len(output_tensors)


def _call_leaves(
    params: _DispatchParams, tensors: list[torch.Tensor]
) -> dict[KeyType, Any]:
    """Call the Tesseract once and return every output leaf by path.

    Flattens all the way down, so arrays inside a container the schema does not
    mark differentiable are found too. Arrays come back as tensors, the
    differentiable ones still attached to autograd.
    """
    slot = _ResultSlot()
    output_tensors = _TesseractFunction.apply(params, slot, *tensors)
    returned = {
        path: tensor
        for (path, _), tensor in zip(slot.outputs, output_tensors, strict=True)
    }
    leaves: dict[KeyType, Any] = {}
    for path, value in slot.flat_result.items():
        if path in returned:
            leaves[path] = returned[path]
            continue
        for leaf_path, leaf in _flatten_pytree(value, path) or [(path, value)]:
            is_array = isinstance(leaf, np.ndarray | np.generic) or hasattr(
                leaf, "__cuda_array_interface__"
            )
            leaves[leaf_path] = _to_tensor(leaf).to(params.device) if is_array else leaf
    return leaves


def _with_batch_dim(value: Any, vmap_method: str, batch_size: int) -> Any:
    """Give an unbatched tensor or NumPy array a leading batch dimension.

    Of size 1 for ``"expand_dims"``, and of the batch size for
    ``"broadcast_all"``. Any other value is returned as it is.
    """
    if not isinstance(value, torch.Tensor | np.ndarray):
        return value
    value = value[None]
    if vmap_method == "broadcast_all":
        is_tensor = isinstance(value, torch.Tensor)
        broadcast = torch.broadcast_to if is_tensor else np.broadcast_to
        value = broadcast(value, (batch_size, *value.shape[1:]))
    return value


def _stack_leaf(path: KeyType, values: list[Any]) -> Any:
    """Stack one output leaf of the per-element calls along a new dim 0.

    A leaf that is not an array cannot carry a batch dimension, so the value
    from the first element is returned, with a warning if another differs.
    """
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    for index, value in enumerate(values[1:], start=1):
        if _leaves_differ(value, first):
            warnings.warn(
                f"Tesseract returned the non-array output "
                f"{'.'.join(str(part) for part in path)!r} as {value!r} for batch "
                f"element {index}, but {first!r} for element 0. Non-array outputs "
                "cannot carry a batch dimension, so the value from element 0 is "
                "the one apply_tesseract returns and the others are ignored.",
                UserWarning,
                stacklevel=2,
            )
            break
    return first


def _leaves_differ(value: Any, first: Any) -> bool:
    """Whether two non-array output leaves disagree.

    ``!=`` settles it for decoded response data. A value whose comparison does
    not reduce to a bool falls back to identity.
    """
    try:
        return bool(value != first)
    except (TypeError, ValueError):
        return value is not first


def _expand_leaf(path: KeyType, value: Any, batch_size: int, vmap_method: str) -> Any:
    """Check an output leaf of a batched call and broadcast it to the batch.

    Under ``"expand_dims"`` an output that depends on no batched input comes
    back with a leading dimension of size 1, which is expanded to the batch
    size. Under ``"broadcast_all"`` every array input already has the full
    batch dimension, so only that size is accepted.
    """
    if not isinstance(value, torch.Tensor):
        return value
    allowed = (batch_size,) if vmap_method == "broadcast_all" else (batch_size, 1)
    if value.ndim == 0 or value.shape[0] not in allowed:
        sizes = " or ".join(str(size) for size in dict.fromkeys(allowed))
        raise ValueError(
            f"vmap_method={vmap_method!r} needs every array output to have a "
            f"leading batch dimension of size {sizes}, but "
            f"{'.'.join(str(part) for part in path)!r} has shape "
            f"{tuple(value.shape)}."
        )
    return value.expand(batch_size, *value.shape[1:])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_tesseract(
    tesseract: Tesseract,
    inputs: dict[str, Any],
    *,
    vmap_method: str | None = None,
) -> dict[str, Any]:
    """Call a Tesseract as a differentiable PyTorch operation.

    Infers which inputs/outputs are differentiable from the Tesseract's schema.
    Torch tensors provided for differentiable fields participate in autograd;
    all other values are passed through as static inputs.

    Supports both reverse-mode (``.backward()``) and forward-mode
    (``torch.autograd.forward_ad``) differentiation.

    Tensor outputs go to the device of the first CUDA tensor among the inputs,
    or to the CPU if there is none. CUDA tensors stay on the device whenever the
    Tesseract offers a GPU transport that works from this process, and are copied
    to the host otherwise (see ``Tesseract.resolve_gpu_transport``); either way
    the result is the same. Pass ``tesseract.with_encoding(gpu_transport="none")``
    to always copy them to the host. An in-process Tesseract receives CUDA
    tensors as they are only if it was created with
    ``Tesseract.from_tesseract_api(..., gpu_transport="cuda_ipc")``, and its
    endpoints must then handle device arrays (e.g. compute with torch). GPU
    transports are an experimental tesseract-core feature.

    Args:
        tesseract: A Tesseract instance.
        inputs: Nested dict matching the Tesseract's input schema.  Provide
            ``torch.Tensor`` for array fields you want gradients through,
            and plain Python / NumPy values for everything else.
        vmap_method: How the call is batched under ``torch.vmap``. ``None``
            (default) raises if a batched tensor reaches the call.
            ``"sequential"`` calls the Tesseract once per batch element and
            works with any schema.
            ``"expand_dims"`` gives every unbatched tensor or NumPy array input
            a leading dimension of size 1 and calls the Tesseract once; its
            schema must accept the extra leading dimension (e.g.
            ``Array[..., Float64]``) and it must broadcast size 1 against the
            batch. ``"broadcast_all"`` does the same, but broadcasts the
            unbatched inputs to the full batch size, for Tesseracts that need
            matching shapes. Under ``torch.vmap`` non-differentiable array
            outputs are returned as tensors, since a NumPy array cannot carry
            the batch dimension. A non-array output is returned once: the
            first element's value under ``"sequential"``, with a warning if
            another element differs, and the batched call's value, which
            describes the whole batch, under the other two methods.
            See :doc:`/content/vmap-methods`.

    Returns:
        Nested dict matching the Tesseract's output schema, with
        differentiable array outputs as ``torch.Tensor`` (with ``grad_fn``
        when inputs require grad) and non-differentiable outputs as-is
        (NumPy arrays or scalars; array outputs become tensors under
        ``torch.vmap``, on the same device as the differentiable ones).

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
    _validate_vmap_method(vmap_method)

    openapi = tesseract.openapi_schema
    diff_in_paths = _get_differentiable_arrays(openapi, "ApplyInputSchema")
    diff_out_paths = _get_differentiable_arrays(openapi, "ApplyOutputSchema")
    diff_out_templates = sorted(diff_out_paths)

    # All known dotted paths guide pytree flattening so we recurse into
    # sub-models but not into opaque dict fields.
    all_paths = diff_in_paths | diff_out_paths

    flat_inputs = [
        item
        for path, value in _flatten_pytree(inputs, recurse_into=all_paths)
        for item in _open_array_containers(path, value)
    ]

    # Partition into differentiable tensors vs static values. A declared path
    # may be a template, so match rather than compare: ``params.{}`` covers the
    # concrete leaf ``params.p`` and is addressed on the wire as ``params.{p}``.
    diff_paths: list[KeyType] = []
    diff_wires: list[str] = []
    diff_tensors: list[torch.Tensor] = []
    nondiff_paths: list[KeyType] = []
    nondiff_tensors: list[torch.Tensor] = []
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
            nondiff_paths.append(path)
            nondiff_tensors.append(value.detach())
        else:
            static[path] = value

    device = _target_device(diff_tensors + nondiff_tensors)
    params = _DispatchParams(
        tesseract=tesseract,
        tensor_paths=diff_paths + nondiff_paths,
        diff_input_wires=diff_wires,
        diff_output_templates=diff_out_templates,
        all_paths=all_paths,
        static_inputs=static,
        device=device,
        gpu_transport=_resolve_gpu_transport(tesseract, device),
        vmap_method=vmap_method,
    )
    slot = _ResultSlot()
    output_tensors = _TesseractFunction.apply(
        params, slot, *diff_tensors, *nondiff_tensors
    )

    # Reconstruct full output pytree. Names were resolved inside forward(),
    # since a dict-valued output has no concrete keys until apply() returns.
    returned = {
        concrete: tensor
        for (concrete, _wire), tensor in zip(slot.outputs, output_tensors, strict=True)
    }
    # forward() already adopted the arrays behind the returned tensors.
    flat_result = {
        path: value if path in returned else _device_arrays_to_numpy(value)
        for path, value in slot.flat_result.items()
    }
    flat_result.update(returned)

    return _unflatten_pytree(flat_result)
