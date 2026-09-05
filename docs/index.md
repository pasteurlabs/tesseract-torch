# Tesseract-Torch

Tesseract-Torch is a lightweight extension to [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) that wraps Tesseracts as differentiable [PyTorch](https://pytorch.org/) operations, with full support for reverse-mode and forward-mode automatic differentiation.

The API of Tesseract-Torch consists of a single function, [`apply_tesseract(tesseract, inputs)`](tesseract_torch.apply_tesseract), which integrates any Tesseract into PyTorch's autograd graph:

```python
result = apply_tesseract(my_tesseract, {"x": x_tensor})
result["y"].sum().backward()  # reverse-mode AD
x_tensor.grad  # gradients flow through the Tesseract
```

Want to learn more? See how to [get started](content/get-started.md) with Tesseract-Torch, explore the [API reference](content/api.md), or learn by [example](examples/simple/demo.ipynb).

## License

Tesseract-Torch is licensed under the [Apache License 2.0](https://github.com/pasteurlabs/tesseract-torch/blob/main/LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.

```{toctree}
:caption: Usage
:maxdepth: 2
:hidden:

content/get-started
content/handling-differentiability
content/troubleshooting
content/api
```

```{toctree}
:caption: Examples
:maxdepth: 2
:hidden:

examples/simple/demo.ipynb
```

```{toctree}
:caption: See also
:maxdepth: 2
:hidden:

Tesseract User Forums <https://si-tesseract.discourse.group/>
```
