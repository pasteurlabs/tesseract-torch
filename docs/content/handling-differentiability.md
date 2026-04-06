# Handling Differentiability

Every Tesseract defines its interface through Pydantic `BaseModel` classes (`InputSchema` and `OutputSchema`). These schemas describe the structure, shapes, and dtypes of all array fields, and crucially, which fields are **differentiable**.

## The `Differentiable[...]` annotation

Fields wrapped with `Differentiable[...]` participate in automatic differentiation. Fields without it are treated as constants by PyTorch's autograd, even if their values change between calls.

```python
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32

class InputSchema(BaseModel):
    x: Differentiable[Array[(3,), Float32]]   # differentiable
    label: Array[(1,), Float32]               # non-differentiable

class OutputSchema(BaseModel):
    loss: Differentiable[Array[(), Float32]]  # differentiable
    metadata: Array[(4,), Float32]            # non-differentiable
```

Fields can be arbitrarily nested (dicts, lists, and nested models). `Differentiable[...]` applies per-leaf:

```python
class InputSchema(BaseModel):
    params: dict[str, Differentiable[Array[(None,), Float32]]]  # all leaves differentiable
    config: dict[str, Array[(None,), Float32]]                  # all leaves non-differentiable
```

When `apply_tesseract` is called, Tesseract-Torch inspects these annotations to determine which inputs participate in the autograd graph and which outputs are returned as `torch.Tensor` with `grad_fn`.

## Non-differentiable inputs

Non-differentiable inputs are passed through to the Tesseract as static values. They do not participate in gradient computation. If you provide a `torch.Tensor` for a non-differentiable field, it will be detached and converted to NumPy before being sent to the Tesseract.

To differentiate only with respect to specific inputs, simply provide `torch.Tensor` with `requires_grad=True` only for the fields you want gradients through:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)   # will get gradients
label = torch.tensor([0.0])                                # no gradients

result = apply_tesseract(tess, {"x": x, "label": label})
result["loss"].backward()
print(x.grad)  # has gradients
```

## Non-differentiable outputs

Output fields not marked as `Differentiable[...]` are returned as plain NumPy arrays or Python scalars. Only differentiable output fields are returned as `torch.Tensor` instances that participate in autograd.

If you need to use a non-differentiable output in downstream PyTorch computation, convert it explicitly:

```python
result = apply_tesseract(tess, inputs)
loss = result["loss"]           # torch.Tensor with grad_fn
metadata = result["metadata"]   # NumPy array, no grad_fn

# Convert if needed for downstream use
metadata_tensor = torch.as_tensor(metadata)
```
