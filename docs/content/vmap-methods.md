# Batching with `torch.vmap`

When you wrap an `apply_tesseract` call in `torch.vmap`, the `vmap_method` argument controls how the batch dimension reaches the Tesseract. The options below go from safe and slow to faster but more demanding of the Tesseract.

- **Start with `"sequential"`** if you are unsure. It works with any schema, including fixed-rank arrays.
- **Use `"expand_dims"`** when the Tesseract accepts a leading batch dimension on every array input and broadcasts it, as NumPy does.
- **Use `"broadcast_all"`** when the Tesseract accepts a leading batch dimension but needs every array input to have the same one.

The `"auto_experimental"` method from tesseract-jax is not available yet; use `"expand_dims"` or `"sequential"` instead.

## Quick reference

| Method            | Unbatched array inputs | Tesseract calls per `torch.vmap` | Tesseract requirement                      |
| ----------------- | ---------------------- | -------------------------------- | ------------------------------------------ |
| `None` (default)  |                        | Raises                           |                                            |
| `"sequential"`    | Unchanged              | One per batch element            | Any schema                                 |
| `"expand_dims"`   | `(1, ...)`             | One                              | Accepts and broadcasts a leading batch dim |
| `"broadcast_all"` | `(batch_size, ...)`    | One                              | Accepts a leading batch dim                |

## Methods in detail

### `None` (default)

```python
apply_tesseract(tess, inputs)
```

Calls outside `torch.vmap` work as usual. Inside `torch.vmap` the call raises once a batched tensor reaches `apply_tesseract`:

```
NotImplementedError: torch.vmap over apply_tesseract needs a batching strategy. Pass vmap_method (one of ['sequential', 'expand_dims', 'broadcast_all']) to apply_tesseract.
```

### `"sequential"`

```python
apply_tesseract(tess, inputs, vmap_method="sequential")
```

Calls the Tesseract once per batch element, with the batch dimension removed. This works with any schema, including one that fixes the number of dimensions, such as `Array[(None,), Float64]`. Each element is a separate request, so it can be slow for large batches.

### `"expand_dims"`

```python
apply_tesseract(tess, inputs, vmap_method="expand_dims")
```

Moves the batch dimension of every batched input to the front, adds a leading dimension of size 1 to every unbatched array input, and calls the Tesseract once. The Tesseract has to accept the extra dimension, so its schema must use `Array[..., dtype]` rather than a fixed number of dimensions, and it has to broadcast `(1, ...)` against `(batch_size, ...)`. No data is duplicated.

A schema with a fixed number of dimensions fails the Tesseract's own input validation (`Array has wrong number of dimensions`). Use `"sequential"` for those.

### `"broadcast_all"`

```python
apply_tesseract(tess, inputs, vmap_method="broadcast_all")
```

Like `"expand_dims"`, but every unbatched array input is broadcast to `(batch_size, ...)`, so all array inputs share the same leading dimension. This sends redundant data to the Tesseract. Use it when the Tesseract checks that its input shapes match rather than broadcasting them.

## What gets batched

Tensors and NumPy arrays are array inputs. Other values are passed through unchanged under every method. Under `"sequential"` unbatched inputs are passed through as they are, so the table only applies to `"expand_dims"` and `"broadcast_all"`.

| Input                  | Example                           | Given a batch dimension by `expand_dims` / `broadcast_all`? |
| ---------------------- | --------------------------------- | ----------------------------------------------------------- |
| Tensor                 | `torch.ones(3)`                   | Yes                                                         |
| NumPy array            | `np.ones(3)`                      | Yes                                                         |
| Python or NumPy scalar | `1.0`, `3`, `True`, `np.int64(2)` | No                                                          |
| String or other value  | `"hello"`                         | No                                                          |

Tensors and NumPy arrays on non-differentiable fields can be batched too, under every method, including ones inside a nested model with no differentiable field.

## Outputs

- Differentiable outputs come back as batched tensors.
- Non-differentiable array outputs come back as tensors inside `torch.vmap`, since a NumPy array cannot carry the batch dimension. Outside `torch.vmap` they stay NumPy arrays.
- Outputs that are not arrays, such as a `bool` or `str` field, cannot carry a batch dimension and are returned once. Under `"sequential"` the value from the first element is returned, with a `UserWarning` if another element returned a different one. Under `"expand_dims"` and `"broadcast_all"` the value comes from the single batched call, so it describes the whole batch rather than any one element. A value that depends on the inputs belongs in the schema as an array instead.
- Under `"broadcast_all"`, every array output must come back with a leading dimension of size `batch_size`. Under `"expand_dims"` it may also be 1, for an output that depends on no batched input. Any other shape raises a `ValueError`.
- Under `"expand_dims"`, an output that depends on no input at all is read by its leading dimension, so one whose leading dimension happens to be 1 or `batch_size` is misread as batched. Use `"sequential"` for such outputs.

## Interaction with autodiff

Batched calls stay differentiable through PyTorch's standard autograd API.

- **Reverse mode** (`.backward()`, `torch.autograd.grad`): the Tesseract's `vector_jacobian_product` is called once per call made in the forward pass, so once for `"expand_dims"` and `"broadcast_all"`, and once per element for `"sequential"`. It receives the same batched inputs as `apply`, and cotangents with the batch dimension in front. For an input that was given a size-1 batch dimension, the Tesseract may return a gradient with the full batch dimension; PyTorch sums it over the batch.
- **Forward mode** (`torch.autograd.forward_ad`): dual tensors work the same way, through `jacobian_vector_product`.

The other `torch.func` transforms (`torch.func.grad`, `torch.func.vjp`, `torch.func.jvp`, `torch.func.jacrev`, `torch.func.jacfwd`) are still not supported, inside `torch.vmap` or not. Neither are batched cotangents, `torch.autograd.grad(..., is_grads_batched=True)` and `torch.autograd.functional.jacobian(..., vectorize=True)`, since they vmap the backward pass. See [Troubleshooting](troubleshooting.md).

### Example: per-element gradients

Every output row depends only on its own input row, so the gradient of the summed output holds one gradient per element. With `"expand_dims"` this is one `apply` call and one `vector_jacobian_product` call:

```python
def f(x):
    return apply_tesseract(tess, {"x": x, "y": y}, vmap_method="expand_dims")["result"]


y = torch.randn(3)
xs = torch.randn(8, 3, requires_grad=True)
torch.vmap(f)(xs).sum().backward()
```

`xs.grad` then has shape `(8, 3)`, one row per element.

## Nested `torch.vmap` and `chunk_size`

Nested `torch.vmap` works with every method. Under `"expand_dims"` and `"broadcast_all"` each level adds a leading dimension, so the Tesseract sees shapes like `(batch_1, batch_2, ...)`.

`torch.vmap(f, chunk_size=k)` splits the batch into chunks of `k` elements. Under `"expand_dims"` and `"broadcast_all"` it makes one batched call per chunk; `"sequential"` still calls once per element.

## GPU transport

`vmap_method` cannot be combined with `gpu_transport` yet. Passing both raises a `ValueError`.
