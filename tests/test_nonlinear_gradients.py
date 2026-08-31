# Copyright 2026 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient correctness where the evaluation point matters.

The other fixtures here are affine, so their Jacobians are constants. That
makes a whole class of bug invisible: a gradient taken at the wrong
linearization point gives the same answer as one taken at the right place.
Replacing the saved inputs with zeros in ``backward`` leaves the rest of
this suite green.

``nonlinear_tesseract`` is cubic with a cross term, ``y = a**3 + b*a``, so
the point shows up in the answer and that mutation fails here.
"""

from __future__ import annotations

import numpy as np
import torch

from tesseract_torch import apply_tesseract


def _inputs(scale: float = 1.0):
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True) * scale
    b = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
    return a.detach().requires_grad_(), b.detach().requires_grad_()


def test_backward_matches_the_closed_form(nonlinear_tess):
    """dy/da = 3a**2 + b and dy/db = a, both evaluated at the actual inputs."""
    a, b = _inputs()
    out = apply_tesseract(nonlinear_tess, {"a": a, "b": b})
    out["y"].sum().backward()

    a_np, b_np = a.detach().numpy(), b.detach().numpy()
    np.testing.assert_allclose(a.grad.numpy(), 3.0 * a_np**2 + b_np, rtol=1e-10)
    np.testing.assert_allclose(b.grad.numpy(), a_np, rtol=1e-10)


def test_gradient_changes_with_the_evaluation_point(nonlinear_tess):
    """The same tensor shape at a different point must give a different gradient.

    This is what an affine fixture cannot express, and it is what catches a
    backward pass that linearizes somewhere other than its own inputs.
    """
    grads = []
    for scale in (1.0, 2.0):
        a, b = _inputs(scale)
        out = apply_tesseract(nonlinear_tess, {"a": a, "b": b})
        out["y"].sum().backward()
        grads.append(a.grad.numpy().copy())

    assert not np.allclose(grads[0], grads[1])


def test_forward_mode_agrees_with_reverse_mode(nonlinear_tess):
    """Forward and reverse mode must describe the same Jacobian at the same point."""
    a, b = _inputs()
    tangent = torch.ones_like(a)

    with torch.autograd.forward_ad.dual_level():
        dual_a = torch.autograd.forward_ad.make_dual(a.detach(), tangent)
        out = apply_tesseract(nonlinear_tess, {"a": dual_a, "b": b.detach()})
        _, jvp = torch.autograd.forward_ad.unpack_dual(out["y"])

    a2, b2 = _inputs()
    out2 = apply_tesseract(nonlinear_tess, {"a": a2, "b": b2})
    out2["y"].sum().backward()

    # y is elementwise, so J is diagonal and J @ 1 equals the vjp of ones.
    np.testing.assert_allclose(jvp.numpy(), a2.grad.numpy(), rtol=1e-10)


def test_gradcheck(nonlinear_tess):
    """Numerical gradcheck through the Tesseract boundary."""
    a, b = _inputs()

    def f(a_, b_):
        return apply_tesseract(nonlinear_tess, {"a": a_, "b": b_})["y"]

    assert torch.autograd.gradcheck(f, (a, b), eps=1e-6, atol=1e-8)
