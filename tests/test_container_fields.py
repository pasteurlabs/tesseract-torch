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

from tesseract_torch import apply_tesseract


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


def test_list_valued_differentiable_field_is_rejected_loudly(tmp_path):
    """A list field cannot be supported yet, so say so rather than detach it.

    ``_flatten_pytree`` only descends into dicts, so a list field arrives as a
    single opaque leaf and its tensors never register for autograd. Without
    this guard that surfaces as "element 0 of tensors does not require grad".
    """
    from tesseract_core import Tesseract

    api = tmp_path / "tesseract_api.py"
    api.write_text(
        "from typing import Any\n"
        "import numpy as np\n"
        "from pydantic import BaseModel\n"
        "from tesseract_core.runtime import Array, Differentiable, Float32\n"
        "class InputSchema(BaseModel):\n"
        "    items: list[Differentiable[Array[(3,), Float32]]]\n"
        "class OutputSchema(BaseModel):\n"
        "    y: Differentiable[Array[(3,), Float32]]\n"
        "def apply(inputs: InputSchema) -> OutputSchema:\n"
        "    d = inputs.model_dump()\n"
        "    return {'y': np.asarray(d['items'][0]) + np.asarray(d['items'][1])}\n"
        "def abstract_eval(abstract_inputs: Any) -> dict:\n"
        "    return {'y': {'shape': (3,), 'dtype': 'float32'}}\n"
    )
    (tmp_path / "tesseract_config.yaml").write_text(
        'name: listwild\nversion: "0.1.0"\n'
    )

    tess = Tesseract.from_tesseract_api(api)
    items = [torch.ones(3, requires_grad=True) for _ in range(2)]
    with pytest.raises(NotImplementedError, match="List-valued differentiable"):
        apply_tesseract(tess, {"items": items})
