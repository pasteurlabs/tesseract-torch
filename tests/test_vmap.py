# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``torch.vmap`` over ``apply_tesseract``, for each ``vmap_method``.

Every batched result is checked against a Python loop of unbatched calls,
which is what the batching rule has to reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD
from pydantic import ValidationError

from tesseract_torch import apply_tesseract

METHODS = ["sequential", "expand_dims", "broadcast_all"]
B, N = 4, 3


def _data():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(B, N, dtype=torch.float64, generator=gen)
    b = torch.randn(N, dtype=torch.float64, generator=gen)
    c = np.linspace(0.0, 1.0, N)
    return a, b, c


def _call(tess, a, b, c, vmap_method=None):
    return apply_tesseract(tess, {"a": a, "b": b, "c": c}, vmap_method=vmap_method)


def _call_with(tess, a, b, c, method, **extra):
    inputs = {"a": a, "b": b, "c": c, **extra}
    return apply_tesseract(tess, inputs, vmap_method=method)["extras"]["z"]


def _spy(monkeypatch, tess):
    """Count the calls reaching the Tesseract's apply and VJP endpoints."""
    counts = {"apply": 0, "vector_jacobian_product": 0}
    for name in counts:
        endpoint = getattr(tess, name)

        def spy(*args, _name=name, _endpoint=endpoint, **kwargs):
            counts[_name] += 1
            return _endpoint(*args, **kwargs)

        monkeypatch.setattr(tess, name, spy)
    return counts


@pytest.mark.parametrize("method", METHODS)
def test_outputs_and_grads_match_a_loop(batched_tess, method):
    """Batched ``a``, shared ``b``: both outputs and both gradients."""
    a, b, c = _data()

    a_loop, b_loop = a.clone().requires_grad_(), b.clone().requires_grad_()
    loop = [_call(batched_tess, a_loop[i], b_loop, c) for i in range(B)]
    y_loop = torch.stack([out["y"] for out in loop])
    y_loop.sum().backward()

    a_vmap, b_vmap = a.clone().requires_grad_(), b.clone().requires_grad_()

    def f(x):
        out = _call(batched_tess, x, b_vmap, c, method)
        return out["y"], out["extras"]["z"]

    y, z = torch.vmap(f)(a_vmap)
    y.sum().backward()

    torch.testing.assert_close(y, y_loop)
    np.testing.assert_allclose(
        z.numpy(), np.stack([out["extras"]["z"] for out in loop])
    )
    torch.testing.assert_close(a_vmap.grad, a_loop.grad)
    torch.testing.assert_close(b_vmap.grad, b_loop.grad)


@pytest.mark.parametrize(
    ("method", "calls"), [("sequential", B), ("expand_dims", 1), ("broadcast_all", 1)]
)
def test_endpoint_call_counts(batched_tess, monkeypatch, method, calls):
    """``sequential`` calls each endpoint once per element, the others once."""
    a, b, c = _data()
    a.requires_grad_()
    b.requires_grad_()
    counts = _spy(monkeypatch, batched_tess)

    y = torch.vmap(lambda x: _call(batched_tess, x, b, c, method)["y"])(a)
    y.sum().backward()

    assert counts == {"apply": calls, "vector_jacobian_product": calls}


@pytest.mark.parametrize(
    ("method", "shape"), [("expand_dims", (1, N)), ("broadcast_all", (B, N))]
)
def test_unbatched_inputs_get_a_batch_dim(batched_tess, monkeypatch, method, shape):
    """Unbatched tensors and NumPy arrays both gain the leading dimension."""
    a, b, c = _data()
    seen = []
    apply = batched_tess.apply
    monkeypatch.setattr(
        batched_tess, "apply", lambda inputs: seen.append(inputs) or apply(inputs)
    )

    torch.vmap(lambda x: _call(batched_tess, x, b, c, method)["y"])(a)

    (inputs,) = seen
    assert inputs["a"].shape == (B, N)
    assert inputs["b"].shape == shape
    assert inputs["c"].shape == shape


@pytest.mark.parametrize("method", METHODS)
def test_forward_mode_matches_a_loop(batched_tess, method):
    """Tangents on the batched ``a`` and on the shared ``b`` both flow through."""
    a, b, c = _data()
    tangent_a, tangent_b = torch.ones_like(a), torch.full_like(b, 0.5)

    with fwAD.dual_level():
        b_dual = fwAD.make_dual(b, tangent_b)
        loop = [
            _call(batched_tess, fwAD.make_dual(a[i], tangent_a[i]), b_dual, c)["y"]
            for i in range(B)
        ]
        loop_tangent = torch.stack([fwAD.unpack_dual(y).tangent for y in loop])

        y = torch.vmap(lambda x: _call(batched_tess, x, b_dual, c, method)["y"])(
            fwAD.make_dual(a, tangent_a)
        )
        tangent = fwAD.unpack_dual(y).tangent

    torch.testing.assert_close(tangent, loop_tangent)


@pytest.mark.parametrize("method", METHODS)
def test_nested_vmap(batched_tess, method):
    a, b, c = _data()

    def f(x):
        return _call(batched_tess, x, b, c, method)["y"]

    y = torch.vmap(torch.vmap(f))(a.reshape(2, 2, N))

    torch.testing.assert_close(y.reshape(B, N), torch.stack([f(x) for x in a]))


@pytest.mark.parametrize("method", METHODS)
def test_batch_dim_not_at_zero(batched_tess, method):
    """``a`` batched along its last dimension, ``b`` along its first."""
    a, b, c = _data()
    bs = b + torch.arange(B, dtype=torch.float64)[:, None]

    y = torch.vmap(
        lambda x, bb: _call(batched_tess, x, bb, c, method)["y"], in_dims=(1, 0)
    )(a.T, bs)

    loop = torch.stack([_call(batched_tess, a[i], bs[i], c)["y"] for i in range(B)])
    torch.testing.assert_close(y, loop)


@pytest.mark.parametrize(
    ("method", "calls"), [("sequential", B), ("expand_dims", 2), ("broadcast_all", 2)]
)
def test_chunk_size(batched_tess, monkeypatch, method, calls):
    """``chunk_size=3`` over 4 elements makes batched calls of 3 and 1."""
    a, b, c = _data()
    counts = _spy(monkeypatch, batched_tess)

    y = torch.vmap(lambda x: _call(batched_tess, x, b, c, method)["y"], chunk_size=3)(a)

    torch.testing.assert_close(y, a**3 + b * a)
    assert counts["apply"] == calls


@pytest.mark.parametrize("method", METHODS)
def test_batched_nondiff_input(batched_tess, method):
    """A tensor on a non-differentiable field is batched like any other.

    ``y`` does not depend on it, so it comes back the same for every element.
    """
    a, b, _ = _data()
    cs = torch.linspace(-1.0, 1.0, B * N, dtype=torch.float64).reshape(B, N)

    def f(cc):
        out = _call(batched_tess, a[0], b, cc, method)
        return out["y"], out["extras"]["z"]

    y, z = torch.vmap(f)(cs)

    torch.testing.assert_close(z, a[0] + cs)
    torch.testing.assert_close(y, (a[0] ** 3 + b * a[0]).expand(B, N))


def test_non_array_output_drift_warns_under_sequential(batched_tess):
    """Under ``sequential`` a non-array output comes from the first element."""
    a, b, c = _data()
    seen = []

    def f(x):
        out = _call(batched_tess, x, b, c, "sequential")
        seen.append(out["extras"]["positive"])
        return out["y"]

    torch.vmap(f)(a.abs() + 1.0)
    assert seen == [True]

    mixed = torch.stack([a[0].abs() + 1.0, -a[1].abs() - 1.0])
    with pytest.warns(UserWarning, match=r"'extras\.positive' as False for batch"):
        y = torch.vmap(f)(mixed)
    assert seen[-1] is True
    torch.testing.assert_close(y, mixed**3 + b * mixed)


@pytest.mark.parametrize("method", ["expand_dims", "broadcast_all"])
def test_non_array_output_describes_the_batched_call(batched_tess, method):
    """One call covers the whole batch, so its non-array output does too."""
    a, b, c = _data()
    mixed = torch.stack([a[0].abs() + 1.0, -a[1].abs() - 1.0])
    seen = []

    def f(x):
        out = _call(batched_tess, x, b, c, method)
        seen.append(out["extras"]["positive"])
        return out["y"]

    torch.vmap(f)(mixed)
    assert seen == [False]


@pytest.mark.parametrize(
    ("method", "raises"), [("expand_dims", False), ("broadcast_all", True)]
)
def test_leading_dim_of_one(batched_tess, monkeypatch, method, raises):
    """Only ``expand_dims`` reads a leading dimension of 1 as unbatched."""
    a, b, c = _data()
    endpoint = batched_tess.apply

    def apply(inputs):
        out = endpoint(inputs)
        out["extras"]["z"] = out["extras"]["z"][:1]
        return out

    monkeypatch.setattr(batched_tess, "apply", apply)

    def f(x):
        return _call(batched_tess, x, b, c, method)["extras"]["z"]

    if raises:
        with pytest.raises(ValueError, match=r"size 4, but 'extras\.z' has shape"):
            torch.vmap(f)(a)
    else:
        assert torch.vmap(f)(a).shape == (B, N)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("as_tensor", [True, False])
def test_array_inside_a_nondiff_submodel(batched_tess, method, as_tensor):
    """An array in a model with no differentiable field is batched too."""
    a, b, c = _data()
    shifts = torch.linspace(-1.0, 1.0, B * N, dtype=torch.float64).reshape(B, N)
    shift = shifts[0] if as_tensor else shifts[0].numpy()

    z = torch.vmap(
        lambda s, x: _call_with(batched_tess, x, b, c, method, opts={"shift": s})
    )(shifts, a)
    z_shared = torch.vmap(
        lambda x: _call_with(batched_tess, x, b, c, method, opts={"shift": shift})
    )(a)

    torch.testing.assert_close(z, a + torch.from_numpy(c) + shifts)
    torch.testing.assert_close(z_shared, a + torch.from_numpy(c) + shifts[0])


@pytest.mark.parametrize("method", METHODS)
def test_empty_container_output_is_kept(batched_tess, method):
    a, b, c = _data()
    seen = []

    def f(x):
        out = _call(batched_tess, x, b, c, method)
        seen.append(out["tags"])
        return out["y"]

    torch.vmap(f)(a)
    assert seen == [[]]


def test_sequential_empty_batch_raises(batched_tess):
    _, b, c = _data()

    with pytest.raises(ValueError, match="empty batch"):
        torch.vmap(lambda x: _call(batched_tess, x, b, c, "sequential")["y"])(
            torch.zeros(0, N, dtype=torch.float64)
        )


@pytest.mark.parametrize("method", METHODS)
def test_numpy_scalar_on_a_non_array_field_is_not_batched(batched_tess, method):
    a, b, c = _data()

    z = torch.vmap(lambda x: _call_with(batched_tess, x, b, c, method, n=np.int64(2)))(
        a
    )

    torch.testing.assert_close(z, 2.0 * (a + torch.from_numpy(c)))


def test_output_without_a_batch_dim_raises(batched_tess):
    """With ``a`` and ``b`` passed as Python floats, ``y`` comes back 0-d."""
    cs = torch.zeros(B, N, dtype=torch.float64)

    with pytest.raises(ValueError, match="'y' has shape \\(\\)"):
        torch.vmap(
            lambda cc: apply_tesseract(
                batched_tess, {"a": 1.0, "b": 2.0, "c": cc}, vmap_method="expand_dims"
            )["extras"]["z"]
        )(cs)


def test_vmap_method_does_not_change_an_unbatched_call(batched_tess):
    a, b, c = _data()

    out = _call(batched_tess, a[0], b, c, "expand_dims")

    torch.testing.assert_close(out["y"], _call(batched_tess, a[0], b, c)["y"])
    assert isinstance(out["extras"]["z"], np.ndarray)


def test_vmap_without_a_method_raises(batched_tess):
    a, b, c = _data()

    with pytest.raises(NotImplementedError, match="Pass vmap_method"):
        torch.vmap(lambda x: _call(batched_tess, x, b, c)["y"])(a)


def test_unknown_method_raises(batched_tess):
    a, b, c = _data()

    with pytest.raises(ValueError, match="Unsupported vmap_method 'loop'"):
        _call(batched_tess, a[0], b, c, "loop")


def test_gpu_transport_with_vmap_method_raises(batched_tess):
    a, b, c = _data()

    with pytest.raises(ValueError, match="gpu_transport"):
        apply_tesseract(
            batched_tess,
            {"a": a[0], "b": b, "c": c},
            gpu_transport="cuda_ipc",
            vmap_method="sequential",
        )


def test_nested_schema_sequential(nested_tess):
    """Fixed shapes, nested models and 0-d outputs, one call per element."""
    vs = torch.arange(B * 3, dtype=torch.float32).reshape(B, 3)
    w = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    def f(v):
        out = apply_tesseract(
            nested_tess,
            {
                "scalars": {"a": 5.0, "b": 2.0},
                "vectors": {"v": v, "w": w},
                "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
            },
            vmap_method="sequential",
        )
        return out["vectors"]["v"], out["vectors"]["w"], out["scalars"]["b"]

    v_out, w_out, b_out = torch.vmap(f)(vs)

    torch.testing.assert_close(v_out, 10.0 * vs + torch.from_numpy(w))
    torch.testing.assert_close(w_out, torch.from_numpy(w).expand(B, 3))
    torch.testing.assert_close(b_out, torch.full((B,), 2.0))


@pytest.mark.parametrize("method", ["expand_dims", "broadcast_all"])
def test_fixed_rank_schema_rejects_the_batch_dim(nonlinear_tess, method):
    """``Array[(None,), Float64]`` cannot take the extra leading dimension."""
    a, b, _ = _data()

    with pytest.raises(ValidationError, match="wrong number of dimensions"):
        torch.vmap(
            lambda x: apply_tesseract(
                nonlinear_tess, {"a": x, "b": b}, vmap_method=method
            )["y"]
        )(a)


def test_fixed_rank_schema_works_sequentially(nonlinear_tess):
    a, b, _ = _data()

    y = torch.vmap(
        lambda x: apply_tesseract(
            nonlinear_tess, {"a": x, "b": b}, vmap_method="sequential"
        )["y"]
    )(a)

    torch.testing.assert_close(y, a**3 + b * a)
