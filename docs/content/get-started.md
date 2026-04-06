# Get started

## Quick start

```{note}
Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+).
```

```{seealso}
For more detailed installation instructions, please refer to the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html).
```

1. Install Tesseract-Torch:

   ```bash
   $ pip install tesseract-torch
   ```

2. Build an example Tesseract:

   ```bash
   $ git clone https://github.com/pasteurlabs/tesseract-torch
   $ tesseract build tesseract-torch/examples/simple/vectoradd_torch
   ```

3. Use it as part of a PyTorch program:

   ```python
   import torch
   from tesseract_core import Tesseract
   from tesseract_torch import apply_tesseract

   # Load the Tesseract
   t = Tesseract.from_image("vectoradd_torch")
   t.serve()

   # Run it with PyTorch tensors
   x = torch.ones(1000, requires_grad=True)
   y = torch.ones(1000)

   def vector_sum(x, y):
       res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}})
       return res["vector_add"]["result"].sum()

   loss = vector_sum(x, y)
   loss.backward()
   print(x.grad)  # gradients via the Tesseract's VJP endpoint

   # Forward-mode AD is also supported
   import torch.autograd.forward_ad as fwAD
   with fwAD.dual_level():
       x_dual = fwAD.make_dual(x.detach(), torch.ones_like(x))
       result = apply_tesseract(t, {"a": {"v": x_dual}, "b": {"v": y}})
       _, tangent = fwAD.unpack_dual(result["vector_add"]["result"])
   ```

```{tip}
Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-torch/tree/main/examples) for ways to use Tesseract-Torch.
```

## Sharp edges

- **Required endpoints**: Using `apply_tesseract` with reverse-mode AD (`.backward()`, `torch.autograd.grad`) requires the Tesseract to define a [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) endpoint. Forward-mode AD (`torch.autograd.forward_ad`) requires [`jacobian_vector_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#jacobian-vector-product).

- **Non-differentiable inputs/outputs**: Only inputs and outputs marked as `Differentiable[...]` in the Tesseract schema participate in gradient computation. See the [Handling Differentiability](handling-differentiability.md) page for details.
