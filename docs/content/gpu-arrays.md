# CUDA tensors

When you pass CUDA tensors to `apply_tesseract`, they stay on the device whenever possible and are copied through the host otherwise. The results are the same either way. Only where they live and how fast the call runs differ. An array the Tesseract returns on the GPU arrives as a CUDA tensor on its device. Anything that comes through the host (CPU arrays, or GPU arrays when no transport works) arrives on the CPU, as a tensor for differentiable outputs and as a NumPy array otherwise. Gradients always land on the device of the input they belong to. GPU transports are an experimental tesseract-core feature.

## Serving a Tesseract with `cuda_ipc`

To exchange CUDA tensors with a served Tesseract without a host copy, serve it with the `cuda_ipc` transport. This tells the Tesseract's runtime that its endpoints accept GPU arrays as inputs, so only enable it for Tesseracts written to handle them:

```python
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

with Tesseract.from_source("tesseract_api.py", gpu_transport="cuda_ipc") as tess:
    out = apply_tesseract(tess, {"x": x.cuda()})
```

`Tesseract.from_image(..., gpus=["all"], gpu_transport="cuda_ipc")` works the same way for a containerized Tesseract.

The Tesseract must run tesseract-core 1.15 or newer, since the `cuda_ipc` wire format changed in that release, so rebuild images built with an older version. It must also be served with a single worker (the default), as tesseract-core rejects GPU transports with `num_workers > 1`.

## How the transport is picked

The first call with CUDA tensors asks the Tesseract which transport to use (via `Tesseract.resolve_gpu_transport`). This checks once per connection that the transport works between your process and the server by exchanging a small GPU array.

- A Tesseract created with `gpu_transport="cuda_ipc"` uses it. If it does not work from your process, the call raises with the reason.
- A Tesseract that requests no transport, such as one from `Tesseract.from_url`, uses `cuda_ipc` if the server offers it and it works. Otherwise its tensors are copied through the host, with a one-time warning saying why. This happens, for example, when the server runs on another machine.
- Calls with only CPU tensors always go through the host and never run the check.

To always copy CUDA tensors through the host, pass a view of the Tesseract:

```python
apply_tesseract(tess.with_encoding(gpu_transport="none"), inputs)
```

## In-process Tesseracts

A Tesseract loaded with `Tesseract.from_tesseract_api` receives NumPy arrays by default, and tensors its endpoints return come back unchanged. Create it with `gpu_transport="cuda_ipc"` to hand its endpoints the CUDA tensors as they are. Its endpoints must then handle device arrays, for example by computing with torch:

```python
tess = Tesseract.from_tesseract_api("tesseract_api.py", gpu_transport="cuda_ipc")
```

The endpoints must not modify these input tensors in place. They share memory with the caller's tensors, so an in-place change such as `x -= x.mean()` silently changes the caller's values. Compute into a new tensor instead (`x = x - x.mean()`). A served Tesseract is not affected, since it receives its own copy of each input.
