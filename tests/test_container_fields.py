# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-valued differentiable fields.

tesseract-core declares ``dict[str, Differentiable[...]]`` as a template,
``params.{}``, and accepts the concrete key in braces on the wire,
``params.{p}``. Matching those paths literally used to detach the tensors on
the input side and raise ``KeyError`` on the output side.

The input case is the dangerous one: with a mixed schema the forward pass is
right, the plain field's gradient is right, nothing raises, and only the dict
field comes back ``None``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from tesseract_torch import apply_tesseract


def COEFF(index: int) -> float:
    """Mirrors ``coefficient`` in the list fixture: distinct weight per entry."""
    return 2.0 + index


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.ones(3, dtype=torch.float32, requires_grad=True)
    p = torch.ones(3, dtype=torch.float32, requires_grad=True)
    return x, p


def test_dict_input_field_receives_its_gradient(dict_tess):
    """A dict-valued input must not be silently detached."""
    x, p = _inputs()
    out = apply_tesseract(dict_tess, {"x": x, "params": {"p": p}})

    np.testing.assert_allclose(out["y"].detach().numpy(), np.full(3, 12.0), rtol=1e-6)
    out["y"].sum().backward()

    assert p.grad is not None, "dict-valued input silently lost its gradient"
    np.testing.assert_allclose(p.grad.numpy(), np.full(3, 10.0), rtol=1e-6)
    np.testing.assert_allclose(x.grad.numpy(), np.full(3, 2.0), rtol=1e-6)


def test_dict_output_field_is_differentiable(dict_tess):
    """A dict-valued output must resolve to tensors and carry gradients back."""
    x, p = _inputs()
    out = apply_tesseract(dict_tess, {"x": x, "params": {"p": p}})

    assert sorted(out["outs"]) == ["a", "b"]
    np.testing.assert_allclose(out["outs"]["a"].detach().numpy(), np.full(3, 3.0))
    np.testing.assert_allclose(out["outs"]["b"].detach().numpy(), np.full(3, 7.0))

    (out["outs"]["a"].sum() + out["outs"]["b"].sum()).backward()
    # d/dx of 3x + 7x
    np.testing.assert_allclose(x.grad.numpy(), np.full(3, 10.0), rtol=1e-6)


def test_gradients_match_an_explicit_reference(dict_tess):
    """Every field's gradient, checked against the closed form together."""
    x, p = _inputs()
    out = apply_tesseract(dict_tess, {"x": x, "params": {"p": p}})
    (out["y"].sum() + out["outs"]["a"].sum() + out["outs"]["b"].sum()).backward()

    # y = 2x + 10p, outs.a = 3x, outs.b = 7x
    np.testing.assert_allclose(x.grad.numpy(), np.full(3, 12.0), rtol=1e-6)
    np.testing.assert_allclose(p.grad.numpy(), np.full(3, 10.0), rtol=1e-6)


class TestListValuedFields:
    """A list-valued differentiable field is differentiated per position.

    The fixture weighs entry i by ``coefficient(i)``, distinct per position, so
    a gradient delivered to the wrong one is visible in the value and not only
    in the path.
    """

    def test_forward_sums_the_weighted_entries(self, list_tess):
        out = apply_tesseract(list_tess, {"xs": [torch.ones(3), torch.ones(3)]})
        expected = COEFF(0) + COEFF(1)
        np.testing.assert_allclose(out["total"].detach().numpy(), np.full(3, expected))

    def test_each_position_gets_its_own_gradient(self, list_tess):
        xs = [torch.ones(3, requires_grad=True) for _ in range(2)]
        apply_tesseract(list_tess, {"xs": xs})["total"].sum().backward()
        for i, x in enumerate(xs):
            np.testing.assert_allclose(x.grad.numpy(), np.full(3, COEFF(i)))

    def test_a_static_entry_stays_static(self, list_tess):
        grad_entry = torch.ones(3, requires_grad=True)
        static = torch.ones(3)
        apply_tesseract(list_tess, {"xs": [grad_entry, static]})[
            "total"
        ].sum().backward()
        np.testing.assert_allclose(grad_entry.grad.numpy(), np.full(3, COEFF(0)))
        assert static.grad is None

    @pytest.mark.parametrize("grad_at", [(1,), (3,), (1, 3), (0, 2, 3)])
    def test_gradients_land_on_the_right_positions(self, list_tess, grad_at):
        """Differentiated entries need not be first, or contiguous, or alone."""
        xs = [torch.full((3,), float(i), requires_grad=i in grad_at) for i in range(4)]
        apply_tesseract(list_tess, {"xs": xs})["total"].sum().backward()
        for i, x in enumerate(xs):
            if i in grad_at:
                np.testing.assert_allclose(x.grad.numpy(), np.full(3, COEFF(i)))
            else:
                assert x.grad is None, f"entry {i} is static and must stay so"

    def test_forward_mode_through_a_later_position(self, list_tess):
        with fwAD.dual_level():
            xs = [torch.ones(3) for _ in range(4)]
            xs[2] = fwAD.make_dual(xs[2], torch.ones(3))
            out = apply_tesseract(list_tess, {"xs": xs})
            _, tangent = fwAD.unpack_dual(out["total"])
        np.testing.assert_allclose(tangent.numpy(), np.full(3, COEFF(2)))


def test_a_numeric_dict_key_is_not_a_list_position():
    """``{"0": v}`` and ``[v]`` must not collapse into each other.

    Positions arrive as ints and dict keys as strings, which is what lets the
    rebuild tell a list from a dict whose keys happen to look like numbers.
    """
    from tesseract_torch.function import _flatten_pytree, _unflatten_pytree

    as_dict = {"params": {"0": torch.ones(2), "1": torch.zeros(2)}}
    flat = dict(_flatten_pytree(as_dict, recurse_into={"params.{}"}))
    assert set(flat) == {("params", "0"), ("params", "1")}
    assert isinstance(_unflatten_pytree(flat)["params"], dict)

    as_list = {"xs": [torch.ones(2), torch.zeros(2)]}
    flat = dict(_flatten_pytree(as_list, recurse_into={"xs.[]"}))
    assert set(flat) == {("xs", 0), ("xs", 1)}
    assert isinstance(_unflatten_pytree(flat)["xs"], list)
