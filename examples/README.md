# Tesseract-Torch examples

This directory contains example Tesseract configurations, notebooks, and scripts demonstrating how to use Tesseract-Torch in various contexts. Each example is self-contained and can be run independently.

## Examples

- [Simple](simple/demo.ipynb): A basic example of using Tesseract-Torch with a simple vector addition task. It demonstrates how to build a Tesseract and execute it within PyTorch, including reverse-mode and forward-mode automatic differentiation.
- [Nested](nested/demo.ipynb): A Tesseract whose inputs and outputs are nested Pydantic models, mixing differentiable arrays with plain settings. It demonstrates calling it with nested dictionaries and taking reverse-mode and forward-mode gradients through a nested output field.
