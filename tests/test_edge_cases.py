# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge-case and regression tests for tesseract-torch.

Covers reported issues and scenarios that early testers are likely to encounter.
"""

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from tesseract_torch import apply_tesseract

# ---------------------------------------------------------------------------
# Issue 1: Tesseract.apply (from tesseract_core) with torch tensors
# ---------------------------------------------------------------------------


class TestTesseractCoreApplyWithTorchTensors:
    """Tesseract.apply (the raw Core API) doesn't handle torch tensors well.

    Known tesseract_core bugs:
    - HTTP client: ``_encode_array`` does ``arr.dtype.name`` which raises
      ``AttributeError: 'torch.dtype' object has no attribute 'name'`` for
      *any* torch tensor (torch.dtype != numpy.dtype).
    - Local client: ``np.asarray()`` during Pydantic validation raises
      ``RuntimeError: Can't call numpy() on Tensor that requires grad`` when
      the tensor tracks gradients.

    These tests document the local-client failure mode (HTTP variant needs a
    running container so is not tested here) and verify that
    ``apply_tesseract`` works around both issues.
    """

    def test_torch_dtype_has_no_name_attribute(self):
        """torch.dtype lacks the .name attr that numpy.dtype has (root cause of HTTP crash)."""
        assert not hasattr(torch.tensor(1.0).dtype, "name")
        assert hasattr(np.dtype(np.float32), "name")

    def test_apply_with_plain_torch_tensors_local(self, vectoradd_tess):
        """Raw Tesseract.apply with local client happens to work for plain tensors."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = vectoradd_tess.apply({"a": a, "b": b})
        np.testing.assert_allclose(result["c"], [5.0, 7.0, 9.0])

    def test_apply_with_requires_grad_tensors_fails(self, vectoradd_tess):
        """Raw Tesseract.apply crashes with requires_grad tensors (known Core bug)."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0])
        with pytest.raises(RuntimeError, match="requires grad"):
            vectoradd_tess.apply({"a": a, "b": b})

    def test_apply_tesseract_handles_all_torch_tensors(self, vectoradd_tess):
        """apply_tesseract (the wrapper) converts tensors before they reach Core."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0])
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))


# ---------------------------------------------------------------------------
# Issue 2: torch.func.vjp crashes with "Tensor that doesn't have storage"
# ---------------------------------------------------------------------------


class TestTorchFuncTransformsRejected:
    """torch.func transforms are not supported and should raise clearly."""

    def test_torch_func_vjp_raises(self, vectoradd_tess):
        params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match=r"does not support torch\.func"):
            torch.func.vjp(
                lambda p: apply_tesseract(vectoradd_tess, {"a": p, "b": b})["c"],
                params,
            )

    def test_torch_func_jvp_raises(self, vectoradd_tess):
        params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tangent = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match=r"does not support torch\.func"):
            torch.func.jvp(
                lambda p: apply_tesseract(vectoradd_tess, {"a": p, "b": b})["c"],
                (params,),
                (tangent,),
            )

    def test_torch_func_grad_raises(self, vectoradd_tess):
        params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match=r"does not support torch\.func"):
            torch.func.grad(
                lambda p: apply_tesseract(vectoradd_tess, {"a": p, "b": b})["c"].sum()
            )(params)

    def test_error_message_suggests_alternatives(self, vectoradd_tess):
        params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        with pytest.raises(RuntimeError, match=r"torch\.autograd\.grad"):
            torch.func.vjp(
                lambda p: apply_tesseract(vectoradd_tess, {"a": p, "b": b})["c"],
                params,
            )


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------


class TestDtypes:
    """Verify that different tensor dtypes are handled correctly."""

    def test_float64_input(self, vectoradd_tess):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        result["c"].sum().backward()
        assert a.grad is not None

    def test_float64_gradient_correctness(self, vectoradd_tess):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        result["c"].sum().backward()
        assert torch.allclose(a.grad, torch.ones(3, dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient accumulation and multiple backward passes
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    def test_gradient_accumulates(self, vectoradd_tess):
        """Calling backward twice accumulates gradients."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        for _ in range(2):
            result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
            result["c"].sum().backward()

        # Gradient accumulated twice: 1 + 1 = 2
        assert torch.allclose(a.grad, torch.full((3,), 2.0), atol=1e-4)

    def test_gradient_after_zero_grad(self, vectoradd_tess):
        """Zeroing grad between backward calls gives fresh gradients."""
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        result["c"].sum().backward()
        a.grad.zero_()

        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        result["c"].sum().backward()
        assert torch.allclose(a.grad, torch.ones(3), atol=1e-4)


# ---------------------------------------------------------------------------
# torch.no_grad and detached tensors
# ---------------------------------------------------------------------------


class TestNoGradAndDetach:
    def test_no_grad_context(self, vectoradd_tess):
        """apply_tesseract works inside torch.no_grad()."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        with torch.no_grad():
            result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})

        assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))
        assert not result["c"].requires_grad

    def test_detached_input(self, vectoradd_tess):
        """Detached tensors work but produce no gradient."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        result = apply_tesseract(vectoradd_tess, {"a": a.detach(), "b": b})
        assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))
        assert not result["c"].requires_grad


# ---------------------------------------------------------------------------
# Multiple differentiable inputs
# ---------------------------------------------------------------------------


class TestMultipleDiffInputs:
    def test_backward_multiple_diff_inputs(self, nested_tess):
        """Backward pass with multiple differentiable inputs produces correct grads."""
        a = torch.tensor(5.0, requires_grad=True)
        v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        result = apply_tesseract(
            nested_tess,
            {
                "scalars": {"a": a, "b": 2.0},
                "vectors": {
                    "v": v,
                    "w": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                },
                "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
            },
        )

        # new_a = a * 10 + b => d(new_a)/da = 10
        result["scalars"]["a"].backward()
        assert torch.allclose(a.grad, torch.tensor(10.0), atol=1e-4)

    def test_jvp_multiple_diff_inputs(self, nested_tess):
        """Forward-mode JVP with multiple differentiable inputs."""
        a = torch.tensor(5.0)
        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tangent_a = torch.tensor(1.0)
        tangent_v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, tangent_a)
            v_dual = fwAD.make_dual(v, tangent_v)

            result = apply_tesseract(
                nested_tess,
                {
                    "scalars": {"a": a_dual, "b": 2.0},
                    "vectors": {
                        "v": v_dual,
                        "w": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                    },
                    "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
                },
            )

            _, tangent_out_a = fwAD.unpack_dual(result["scalars"]["a"])
            _, tangent_out_v = fwAD.unpack_dual(result["vectors"]["v"])

        # new_a = a * 10 + b => d(new_a)/da = 10
        assert torch.allclose(tangent_out_a, torch.tensor(10.0), atol=1e-4)
        # new_v = v * 10 + w => d(new_v)/dv tangent = 10 * [1, 0, 0]
        assert torch.allclose(tangent_out_v, torch.tensor([10.0, 0.0, 0.0]), atol=1e-4)


# ---------------------------------------------------------------------------
# Scalar (0-d) tensors
# ---------------------------------------------------------------------------


class TestScalarTensors:
    def test_scalar_input_forward(self, nested_tess):
        """Scalar (0-d) tensor inputs work in forward pass."""
        a = torch.tensor(5.0)
        result = apply_tesseract(
            nested_tess,
            {
                "scalars": {"a": a, "b": 2.0},
                "vectors": {
                    "v": torch.tensor([1.0, 2.0, 3.0]),
                    "w": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                },
                "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
            },
        )
        assert torch.allclose(result["scalars"]["a"], torch.tensor(52.0))

    def test_scalar_input_backward(self, nested_tess):
        """Scalar (0-d) tensor inputs support backward."""
        a = torch.tensor(5.0, requires_grad=True)
        result = apply_tesseract(
            nested_tess,
            {
                "scalars": {"a": a, "b": 2.0},
                "vectors": {
                    "v": torch.tensor([1.0, 2.0, 3.0]),
                    "w": np.array([0.0, 1.0, 2.0], dtype=np.float32),
                },
                "other_stuff": {"s": "hello", "i": 42, "f": 3.14},
            },
        )
        result["scalars"]["a"].backward()
        assert torch.allclose(a.grad, torch.tensor(10.0), atol=1e-4)


# ---------------------------------------------------------------------------
# Input type variations
# ---------------------------------------------------------------------------


class TestInputTypeVariations:
    def test_all_numpy_inputs(self, vectoradd_tess):
        """All-numpy inputs (no torch tensors) work correctly."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        # Differentiable outputs are always torch tensors
        assert isinstance(result["c"], torch.Tensor)
        assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))

    def test_all_torch_inputs(self, vectoradd_tess):
        """All-torch inputs (including non-diff) work correctly."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = torch.tensor([4.0, 5.0, 6.0])
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))
        result["c"].sum().backward()
        assert torch.allclose(a.grad, torch.ones(3), atol=1e-4)

    def test_python_list_for_non_diff(self, vectoradd_tess):
        """Python lists for non-differentiable inputs work."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = [4.0, 5.0, 6.0]
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert torch.allclose(result["c"], torch.tensor([5.0, 7.0, 9.0]))


# ---------------------------------------------------------------------------
# Autograd graph integrity
# ---------------------------------------------------------------------------


class TestAutogradGraphIntegrity:
    def test_output_has_grad_fn(self, vectoradd_tess):
        """Output tensor has grad_fn when input requires grad."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert result["c"].grad_fn is not None

    def test_output_no_grad_fn_when_no_grad(self, vectoradd_tess):
        """Output tensor has no grad_fn when no input requires grad."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert result["c"].grad_fn is None

    def test_retain_graph(self, vectoradd_tess):
        """retain_graph=True allows multiple backward calls on the same graph."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        result["c"].sum().backward(retain_graph=True)
        grad1 = a.grad.clone()
        a.grad.zero_()
        result["c"].sum().backward()
        assert torch.allclose(grad1, a.grad)

    def test_higher_order_grad_not_supported(self, vectoradd_tess):
        """Higher-order gradients via create_graph are not supported.

        The VJP/JVP calls are opaque to autograd, so the backward pass itself
        is not differentiable. This is a known limitation.
        """
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        (grad_a,) = torch.autograd.grad(result["c"].sum(), a, create_graph=True)
        with pytest.raises(RuntimeError, match="does not require grad"):
            torch.autograd.grad(grad_a.sum(), a)

    def test_chained_tesseracts(self, vectoradd_tess):
        """Two chained apply_tesseract calls propagate gradients correctly."""
        a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        # First call: c = a + b
        r1 = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        # Second call: d = c + b = a + 2b
        r2 = apply_tesseract(vectoradd_tess, {"a": r1["c"], "b": b})

        r2["c"].sum().backward()
        # d = a + 2b => dd/da = 1
        assert torch.allclose(a.grad, torch.ones(3), atol=1e-4)


# ---------------------------------------------------------------------------
# GPU support (if available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    def test_cuda_tensor_forward(self, vectoradd_tess):
        """CUDA tensors are handled (converted to CPU internally)."""
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        assert torch.allclose(result["c"].cpu(), torch.tensor([5.0, 7.0, 9.0]))

    def test_cuda_tensor_backward(self, vectoradd_tess):
        """CUDA tensors support backward pass."""
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        result = apply_tesseract(vectoradd_tess, {"a": a, "b": b})
        result["c"].sum().backward()
        assert a.grad is not None


# ---------------------------------------------------------------------------
# Edge paths in function.py
# ---------------------------------------------------------------------------


class TestFlattenEdgePaths:
    """Shapes the pytree walk has to keep rather than descend into."""

    def test_empty_sub_dict_stays_a_leaf(self):
        """An empty dict is a value, not a branch to walk.

        Recursing into it would yield nothing, so the key would drop out of
        the flattened inputs entirely and never reach the Tesseract.
        """
        from tesseract_torch.function import _flatten_pytree

        tree = {"a": 1, "empty": {}, "nested": {"b": 2}}
        assert dict(_flatten_pytree(tree)) == {"a": 1, "empty": {}, "nested.b": 2}
        assert dict(_flatten_pytree(tree, recurse_into={"nested.b"})) == {
            "a": 1,
            "empty": {},
            "nested.b": 2,
        }


class TestPartialForwardTangents:
    """Forward mode when only some differentiable inputs carry a tangent."""

    @pytest.mark.parametrize("seeded", ["a", "b"])
    def test_only_the_seeded_input_is_sent(self, nonlinear_tess, seeded):
        """Seeding one input gives its own term and not the other's.

        The fixture is ``y = a**3 + b*a``, so the two inputs have different
        derivatives and sending the wrong wire would be visible in the value.
        """
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
        tangent_in = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)

        expected = (3 * a**2 + b) if seeded == "a" else a

        with fwAD.dual_level():
            inputs = {"a": a, "b": b}
            inputs[seeded] = fwAD.make_dual(inputs[seeded], tangent_in)
            out = apply_tesseract(nonlinear_tess, inputs)["y"]
            tangent = fwAD.unpack_dual(out).tangent

        torch.testing.assert_close(tangent, expected * tangent_in)
