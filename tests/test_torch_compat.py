# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from tesseract_torch import apply_tesseract

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(requires_grad: bool = False):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=requires_grad)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    return {"a": a, "b": b}, a


def _make_nested_inputs(requires_grad: bool = False):
    v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=requires_grad)
    return {
        "scalars": {"a": torch.tensor(5.0, requires_grad=requires_grad), "b": 2.0},
        "vectors": {"v": v, "w": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
        "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
    }, v


def test_forward_pass(vectoradd_tess):
    inputs, _ = _make_inputs()
    result = apply_tesseract(vectoradd_tess, inputs)

    assert "c" in result
    assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))


def test_output_types(vectoradd_tess):
    inputs, _ = _make_inputs()
    result = apply_tesseract(vectoradd_tess, inputs)

    assert isinstance(result["c"], torch.Tensor)


def test_backward_pass(vectoradd_tess):
    inputs, a = _make_inputs(requires_grad=True)
    result = apply_tesseract(vectoradd_tess, inputs)

    # c = a + b, so dc/da = 1 everywhere
    result["c"].sum().backward()

    expected_grad = torch.ones_like(a)
    assert torch.allclose(a.grad, expected_grad, atol=1e-4)


def test_autograd_grad(vectoradd_tess):
    inputs, a = _make_inputs(requires_grad=True)
    result = apply_tesseract(vectoradd_tess, inputs)

    (grad_a,) = torch.autograd.grad(result["c"].sum(), a)

    expected_grad = torch.ones_like(a)
    assert torch.allclose(grad_a, expected_grad, atol=1e-4)


def test_forward_mode_jvp(vectoradd_tess):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    # Tangent: perturb only the first element of a
    tangent_a = torch.zeros_like(a)
    tangent_a[0] = 1.0

    with fwAD.dual_level():
        a_dual = fwAD.make_dual(a, tangent_a)
        result = apply_tesseract(vectoradd_tess, {"a": a_dual, "b": b})
        _, tangent_c = fwAD.unpack_dual(result["c"])

    assert tangent_c is not None
    # c = a + b, so dc/da = identity -> tangent_c = tangent_a = [1, 0, 0]
    assert torch.allclose(tangent_c, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# Nested schema tests
# ---------------------------------------------------------------------------


def test_nested_forward_pass(nested_tess):
    inputs, _ = _make_nested_inputs()
    result = apply_tesseract(nested_tess, inputs)

    # new_a = a * 10 + b = 5 * 10 + 2 = 52
    assert "scalars" in result
    assert torch.allclose(result["scalars"]["a"], torch.tensor(52.0))

    # new_v = v * 10 + w = [10, 20, 30] + [0, 1, 2] = [10, 21, 32]
    assert "vectors" in result
    assert torch.allclose(result["vectors"]["v"], torch.tensor([10.0, 21.0, 32.0]))


def test_nested_nondiff_output_types(nested_tess):
    inputs, _ = _make_nested_inputs()
    result = apply_tesseract(nested_tess, inputs)

    # Differentiable outputs should be torch.Tensor
    assert isinstance(result["scalars"]["a"], torch.Tensor)
    assert isinstance(result["vectors"]["v"], torch.Tensor)

    # Non-differentiable outputs should NOT be torch.Tensor
    assert not isinstance(result["scalars"]["b"], torch.Tensor)
    assert not isinstance(result["vectors"]["w"], torch.Tensor)


def test_nondiff_input_as_tensor(nested_tess):
    """Non-differentiable inputs provided as torch.Tensor are detached and passed through."""
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    inputs = {
        "scalars": {"a": torch.tensor(5.0, requires_grad=True), "b": torch.tensor(2.0)},
        "vectors": {"v": v, "w": torch.tensor([0.0, 1.0, 2.0])},
        "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
    }
    result = apply_tesseract(nested_tess, inputs)

    # Forward pass should still work correctly
    assert torch.allclose(result["vectors"]["v"], torch.tensor([10.0, 21.0, 32.0]))

    # Backward through differentiable output should work
    result["vectors"]["v"].sum().backward()
    assert torch.allclose(v.grad, torch.full_like(v, 10.0), atol=1e-4)


def test_nested_backward_pass(nested_tess):
    inputs, v = _make_nested_inputs(requires_grad=True)
    result = apply_tesseract(nested_tess, inputs)

    # new_v = v * 10 + w, so d(sum(new_v))/dv = 10
    result["vectors"]["v"].sum().backward()
    assert torch.allclose(v.grad, torch.full_like(v, 10.0), atol=1e-4)


def test_nested_forward_mode_jvp(nested_tess):
    v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    tangent_v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    with fwAD.dual_level():
        v_dual = fwAD.make_dual(v, tangent_v)
        result = apply_tesseract(
            nested_tess,
            {
                "scalars": {"a": 5.0, "b": 2.0},
                "vectors": {
                    "v": v_dual,
                    "w": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                },
                "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
            },
        )
        _, tangent_out = fwAD.unpack_dual(result["vectors"]["v"])

    assert tangent_out is not None
    # new_v = v * 10 + w, so d(new_v)/dv = 10*I -> tangent = 10 * [1, 0, 0]
    assert torch.allclose(tangent_out, torch.tensor([10.0, 0.0, 0.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# Missing endpoint error handling
# ---------------------------------------------------------------------------


def test_forward_pass_without_ad_endpoints(forwardonly_tess):
    """Forward pass works even when JVP/VJP endpoints are missing."""
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = apply_tesseract(forwardonly_tess, {"a": a})

    assert torch.allclose(result["b"], torch.tensor([2.0, 4.0, 6.0]))


def test_backward_raises_without_vjp_endpoint(forwardonly_tess):
    """Reverse-mode AD raises NotImplementedError when VJP endpoint is missing."""
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
    result = apply_tesseract(forwardonly_tess, {"a": a})

    with pytest.raises(NotImplementedError, match="VJP"):
        result["b"].sum().backward()


def test_forward_ad_raises_without_jvp_endpoint(forwardonly_tess):
    """Forward-mode AD raises NotImplementedError when JVP endpoint is missing."""
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    tangent = torch.ones_like(a)

    with pytest.raises(NotImplementedError, match="JVP"):
        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, tangent)
            apply_tesseract(forwardonly_tess, {"a": a_dual})
