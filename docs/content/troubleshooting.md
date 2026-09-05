# Troubleshooting

The errors below are raised on purpose, at the point where Tesseract-Torch can still say what went wrong. Each entry lists the message you will see, why it happens, and what to do about it.

## `apply_tesseract` inside a `torch.func` transform

```
RuntimeError: apply_tesseract does not support torch.func transforms (torch.func.vjp, torch.func.jvp, torch.func.grad, etc.). Use the standard autograd API instead:
  - Reverse mode: result['y'].backward() or torch.autograd.grad()
  - Forward mode: torch.autograd.forward_ad (dual tensors)
```

**Cause.** `torch.func` transforms (`torch.func.vjp`, `torch.func.jvp`, `torch.func.grad`, `torch.func.vmap`) trace your function with functionalized tensors that have no backing storage. A Tesseract endpoint receives NumPy arrays, and such a tensor cannot be converted into one.

**Fix.** Use PyTorch's standard autograd API, which `apply_tesseract` supports in both modes:

```python
# Reverse mode
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
result = apply_tesseract(tess, inputs={"x": x})
(grad_x,) = torch.autograd.grad(result["y"].sum(), x)

# Forward mode
import torch.autograd.forward_ad as fwAD

with fwAD.dual_level():
    x_dual = fwAD.make_dual(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3))
    result = apply_tesseract(tess, inputs={"x": x_dual})
    _, jvp = fwAD.unpack_dual(result["y"])
```

## Reverse-mode AD against a Tesseract without `vector_jacobian_product`

```
NotImplementedError: Vector Jacobian Product (VJP) not implemented for this Tesseract.
```

**Cause.** `.backward()` and `torch.autograd.grad` are dispatched to the Tesseract's [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) endpoint. The Tesseract you called only defines `apply`, so there is nothing to dispatch to. The forward pass itself succeeds; the error appears when the backward pass runs.

**Fix.** Either implement `vector_jacobian_product` in the Tesseract, or do not ask for gradients through it: pass inputs without `requires_grad=True` and it behaves as a plain, non-differentiable operation.

## Forward-mode AD against a Tesseract without `jacobian_vector_product`

```
NotImplementedError: Jacobian Vector Product (JVP) not implemented for this Tesseract.
```

**Cause.** The forward-mode counterpart of the previous entry: a call inside `torch.autograd.forward_ad.dual_level()` with dual tensors as inputs is dispatched to the Tesseract's [`jacobian_vector_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#jacobian-vector-product) endpoint, which this Tesseract does not define.

**Fix.** Implement `jacobian_vector_product` in the Tesseract, or pass plain (non-dual) tensors.

## List-valued differentiable fields

```
NotImplementedError: List-valued differentiable inputs are not supported yet: xs.[]. Use a dict-valued field, or keep the list entries as separate schema fields.
```

**Cause.** Tesseract-Torch walks nested dictionaries and sub-models to find differentiable leaves, but a `list[Differentiable[...]]` field arrives as a single opaque value, so its tensors cannot be registered with autograd. The same applies to list-valued differentiable outputs.

**Fix.** Change the schema to a dict-valued field, or to one field per entry:

```python
class InputSchema(BaseModel):
    xs: dict[str, Differentiable[Array[(3,), Float32]]]
```

and call it with `{"xs": {"first": x0, "second": x1}}`.

## Forward-mode AD with an in-process PyTorch Tesseract

```
RuntimeError: Error running Tesseract API jacobian_vector_product: Nested forward mode AD is not supported at the moment
```

**Cause.** `Tesseract.from_tesseract_api(...)` runs the Tesseract in your own Python process. If its `jacobian_vector_product` is itself implemented with `torch.func.jvp` (as the Tesseracts in `examples/` are), that call opens a second forward-AD level inside your `dual_level()` block, which PyTorch refuses.

**Fix.** Serve the Tesseract in a container (`Tesseract.from_image(...)` followed by `.serve()`, or a `with` block), which is how the examples run it. Reverse-mode AD is unaffected either way.
