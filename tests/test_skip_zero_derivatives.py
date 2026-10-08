# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""backward() and jvp() must skip zero (co)tangents and inputs without grad.

The ``nested_tesseract`` fixture scales ``scalars.a`` and ``vectors.v`` by 10
independently and records the wires in each VJP and JVP request it receives.
"""

import json
import types

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from tesseract_torch import apply_tesseract
from tesseract_torch.function import _N_NON_TENSOR_ARGS, _TesseractFunction


def _records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _inputs(a, v):
    return {
        "scalars": {"a": a, "b": 2.0},
        "vectors": {"v": v, "w": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
        "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
    }


# ---------------------------------------------------------------------------
# Reverse mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loss_fn", "expected_request", "expected_a_grad", "expected_v_grad"),
    [
        (lambda out: out["scalars"]["a"], ["scalars.a"], 10.0, [0.0, 0.0, 0.0]),
        (
            lambda out: out["vectors"]["v"].sum(),
            ["vectors.v"],
            0.0,
            [10.0, 10.0, 10.0],
        ),
        (
            lambda out: out["scalars"]["a"] + out["vectors"]["v"].sum(),
            ["scalars.a", "vectors.v"],
            10.0,
            [10.0, 10.0, 10.0],
        ),
    ],
    ids=["scalar", "vector", "both"],
)
def test_backward_requests_only_used_outputs(
    nested_tess,
    tmp_path,
    monkeypatch,
    loss_fn,
    expected_request,
    expected_a_grad,
    expected_v_grad,
):
    """The VJP request lists exactly the outputs the loss used.

    An input reachable only from an unused output gets the fixture's shaped
    zero, so its grad is zero rather than None.
    """
    record = tmp_path / "vjp_record.jsonl"
    monkeypatch.setenv("NESTED_VJP_RECORD", str(record))

    a = torch.tensor(5.0, requires_grad=True)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    loss_fn(out).backward()

    assert _records(record) == [expected_request]
    assert torch.allclose(a.grad, torch.tensor(expected_a_grad), atol=1e-4)
    assert torch.allclose(v.grad, torch.tensor(expected_v_grad), atol=1e-4)


def test_backward_drops_zero_cotangent(nested_tess, tmp_path, monkeypatch):
    """An output used with an all-zero cotangent is dropped like an unused one."""
    record = tmp_path / "vjp_record.jsonl"
    monkeypatch.setenv("NESTED_VJP_RECORD", str(record))

    a = torch.tensor(5.0, requires_grad=True)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    # vectors.v is on the backward path, but its cotangent is all zeros
    (out["scalars"]["a"] + 0.0 * out["vectors"]["v"].sum()).backward()

    reqs = _records(record)
    assert reqs == [["scalars.a"]], (
        f"expected a single VJP request for ['scalars.a'], got {reqs}"
    )
    assert torch.allclose(a.grad, torch.tensor(10.0), atol=1e-4)
    assert torch.allclose(v.grad, torch.zeros(3), atol=1e-4)


def test_backward_with_only_zero_cotangents_skips_vjp(nested_tess, monkeypatch):
    """Every cotangent zero -> the VJP is skipped and every grad is zero.

    The gradients must be zeros rather than None, which autograd reads as an
    unused input: torch.autograd.grad would fail and .grad would stay unset.
    """

    def _no_vjp(**kwargs):
        raise AssertionError("VJP called despite all-zero cotangents")

    monkeypatch.setattr(nested_tess, "vector_jacobian_product", _no_vjp)

    a = torch.tensor(5.0, requires_grad=True)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    loss = 0.0 * out["scalars"]["a"] + 0.0 * out["vectors"]["v"].sum()

    grad_a, grad_v = torch.autograd.grad(loss, (a, v), retain_graph=True)
    assert torch.equal(grad_a, torch.zeros(()))
    assert torch.equal(grad_v, torch.zeros(3))

    loss.backward()
    assert torch.equal(a.grad, torch.zeros(()))
    assert torch.equal(v.grad, torch.zeros(3))


@pytest.mark.parametrize(
    ("a_requires_grad", "v_requires_grad", "expected_request"),
    [
        (True, False, ["scalars.a"]),
        (False, True, ["vectors.v"]),
        (True, True, ["scalars.a", "vectors.v"]),
    ],
)
def test_backward_requests_only_inputs_requiring_grad(
    nested_tess,
    tmp_path,
    monkeypatch,
    a_requires_grad,
    v_requires_grad,
    expected_request,
):
    """A differentiable input that does not require grad is left out of vjp_inputs."""
    record = tmp_path / "vjp_inputs_record.jsonl"
    monkeypatch.setenv("NESTED_VJP_INPUTS_RECORD", str(record))

    a = torch.tensor(5.0, requires_grad=a_requires_grad)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=v_requires_grad)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    (out["scalars"]["a"] + out["vectors"]["v"].sum()).backward()

    assert _records(record) == [expected_request]
    if a_requires_grad:
        assert torch.allclose(a.grad, torch.tensor(10.0), atol=1e-4)
    else:
        assert a.grad is None
    if v_requires_grad:
        assert torch.allclose(v.grad, 10.0 * torch.ones(3), atol=1e-4)
    else:
        assert v.grad is None


# ---------------------------------------------------------------------------
# Forward mode
# ---------------------------------------------------------------------------


def _jvp_from_seed(tess, seed):
    """Output tangents when both inputs are slices of one dual seeded with ``seed``."""
    params = torch.tensor([5.0, 1.0, 2.0, 3.0])
    with fwAD.dual_level():
        p = fwAD.make_dual(params, seed)
        out = apply_tesseract(tess, _inputs(p[0], p[1:]))
        return (
            fwAD.unpack_dual(out["scalars"]["a"]).tangent,
            fwAD.unpack_dual(out["vectors"]["v"]).tangent,
        )


@pytest.mark.parametrize(
    ("seed", "expected_request", "expected_a", "expected_v"),
    [
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), ["scalars.a"], 10.0, [0.0, 0.0, 0.0]),
        (torch.tensor([0.0, 0.0, 1.0, 0.0]), ["vectors.v"], 0.0, [0.0, 10.0, 0.0]),
        (
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            ["scalars.a", "vectors.v"],
            10.0,
            [0.0, 10.0, 0.0],
        ),
    ],
)
def test_jvp_requests_only_nonzero_tangents(
    nested_tess, tmp_path, monkeypatch, seed, expected_request, expected_a, expected_v
):
    record = tmp_path / "jvp_record.jsonl"
    monkeypatch.setenv("NESTED_JVP_RECORD", str(record))

    tangent_a, tangent_v = _jvp_from_seed(nested_tess, seed)

    assert _records(record) == [expected_request]
    assert torch.allclose(tangent_a, torch.tensor(expected_a), atol=1e-4)
    assert torch.allclose(tangent_v, torch.tensor(expected_v), atol=1e-4)


def test_jvp_all_zero_tangents_skips_call(nested_tess, tmp_path, monkeypatch):
    """Every tangent zero -> no JVP call, and zero tangents shaped like outputs."""
    record = tmp_path / "jvp_record.jsonl"
    monkeypatch.setenv("NESTED_JVP_RECORD", str(record))

    tangent_a, tangent_v = _jvp_from_seed(nested_tess, torch.zeros(4))

    assert _records(record) == []
    assert tangent_a.shape == torch.Size([])
    assert tangent_v.shape == torch.Size([3])
    assert not tangent_a.any()
    assert not tangent_v.any()


def test_jvp_moves_tangents_to_output_device(nested_tess, monkeypatch):
    """Each tangent lands on its output's device, wherever the JVP computed it.

    The ``meta`` device stands in for CUDA so this runs without a GPU.
    """
    monkeypatch.setattr(
        nested_tess,
        "jacobian_vector_product",
        lambda **kwargs: {"y": np.ones(3, dtype=np.float32)},
    )

    ctx = types.SimpleNamespace(
        tesseract=nested_tess,
        diff_input_wires=["x"],
        diff_output_wires=["y"],
        diff_output_specs=[(torch.Size([3]), torch.float32, torch.device("meta"))],
        gpu_transport="none",
        on_device=False,
        static_inputs={},
        tensor_paths=[("x",)],
        tensor_dtypes=[None],
        saved_tensors=(torch.ones(3),),
    )

    (tangent,) = _TesseractFunction.jvp(
        ctx, *(None,) * _N_NON_TENSOR_ARGS, torch.ones(3)
    )

    assert tangent.device.type == "meta"
