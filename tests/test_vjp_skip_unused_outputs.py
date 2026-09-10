# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""backward() must request a VJP only for the outputs a loss actually used.

The ``nested_tesseract`` fixture has two independent differentiable outputs
(``scalars.a`` from ``scalars.a`` and ``vectors.v`` from ``vectors.v``, a
diagonal map) and records the ``vjp_outputs`` its VJP receives, so we can
assert on the exact request; correctness of the surviving gradient is checked
alongside.
"""

import json
import types

import numpy as np
import torch

from tesseract_torch import apply_tesseract
from tesseract_torch.function import _TesseractFunction


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _inputs(a, v):
    return {
        "scalars": {"a": a, "b": 2.0},
        "vectors": {"v": v, "w": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
        "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
    }


def test_backward_requests_only_scalar_output(nested_tess, tmp_path, monkeypatch):
    """Loss uses scalars.a only -> the VJP request lists exactly that; grad is 10."""
    record = tmp_path / "vjp_record.jsonl"
    monkeypatch.setenv("NESTED_VJP_RECORD", str(record))

    a = torch.tensor(5.0, requires_grad=True)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    # loss depends on scalars.a alone; vectors.v is never used
    out["scalars"]["a"].backward()

    reqs = _records(record)
    assert reqs == [["scalars.a"]], (
        f"expected a single VJP request for ['scalars.a'], got {reqs}"
    )
    # d(new_a)/da = 10. vectors.v is dropped from vjp_outputs, so its input
    # wire gets the shaped zero the runtime still requires (grad 0, not None).
    assert torch.allclose(a.grad, torch.tensor(10.0), atol=1e-4)


def test_backward_requests_only_vector_output(nested_tess, tmp_path, monkeypatch):
    """Loss uses vectors.v only -> the VJP request lists exactly that; grad is 10."""
    record = tmp_path / "vjp_record.jsonl"
    monkeypatch.setenv("NESTED_VJP_RECORD", str(record))

    a = torch.tensor(5.0, requires_grad=True)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    out["vectors"]["v"].sum().backward()

    reqs = _records(record)
    assert reqs == [["vectors.v"]], (
        f"expected a single VJP request for ['vectors.v'], got {reqs}"
    )
    # d(new_v)/dv = 10. scalars.a is dropped from vjp_outputs, so its input
    # wire gets the shaped zero the runtime still requires (grad 0, not None).
    assert torch.allclose(v.grad, 10.0 * torch.ones(3), atol=1e-4)


def test_backward_requests_both_when_both_used(nested_tess, tmp_path, monkeypatch):
    """Loss uses both outputs -> the VJP request lists both; grads are 10 each."""
    record = tmp_path / "vjp_record.jsonl"
    monkeypatch.setenv("NESTED_VJP_RECORD", str(record))

    a = torch.tensor(5.0, requires_grad=True)
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = apply_tesseract(nested_tess, _inputs(a, v))
    (out["scalars"]["a"].sum() + out["vectors"]["v"].sum()).backward()

    reqs = _records(record)
    assert reqs == [["scalars.a", "vectors.v"]], (
        f"expected one VJP request for both outputs, got {reqs}"
    )
    assert torch.allclose(a.grad, torch.tensor(10.0), atol=1e-4)
    assert torch.allclose(v.grad, 10.0 * torch.ones(3), atol=1e-4)


def test_backward_with_no_cotangents_skips_vjp():
    """No output carries a cotangent -> the VJP is skipped and every grad is None.

    Autograd prunes a node whose outputs are all off the backward path before
    calling backward(), so this guard is not reachable through a normal
    .backward(); we drive it directly with all-None grad_outputs.
    """

    class _NoVJP:
        def vector_jacobian_product(self, **kwargs):
            raise AssertionError("VJP called despite no incoming cotangent")

    ctx = types.SimpleNamespace(
        tesseract=_NoVJP(),
        diff_output_wires=["scalars.a", "vectors.v"],
        diff_input_wires=["scalars.a", "vectors.v"],
        saved_inputs={},
    )

    grad_inputs = _TesseractFunction.backward(ctx, None, None)

    # Eight None for the non-tensor arguments, then one None per input wire.
    assert grad_inputs == (None,) * (8 + len(ctx.diff_input_wires))
