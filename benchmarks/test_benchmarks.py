# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for tesseract-torch.

Measures execution time for noop and vectoradd Tesseracts:
- Apply (warm cache)
- Dtype casting (float64 -> float32)
- Reverse-mode AD (vjp via .backward())
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import DEFAULT_ARRAY_SIZES, create_test_array

from tesseract_torch import apply_tesseract


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests based on --array-sizes."""
    if "array_size" in metafunc.fixturenames:
        raw = metafunc.config.getoption("--array-sizes", default=None)
        if raw:
            sizes = [int(s.strip()) for s in raw.split(",")]
        else:
            sizes = DEFAULT_ARRAY_SIZES

        ids = [f"{size:,}" for size in sizes]
        metafunc.parametrize("array_size", sizes, ids=ids)


def _make_vectoradd_inputs(a_v, b_v):
    """Create vectoradd inputs from pre-built arrays."""
    return {
        "a": torch.as_tensor(a_v, dtype=torch.float32),
        "b": np.asarray(b_v, dtype=np.float32),
    }


class TestNoopApi:
    """Benchmarks for noop Tesseract via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, noop_tesseract_api, array_size):
        self.tess = noop_tesseract_api
        self.inputs = {
            "data": torch.as_tensor(create_test_array(array_size), dtype=torch.float32)
        }
        self.inputs_f64 = {
            "data": torch.as_tensor(
                create_test_array(array_size, dtype="float64"), dtype=torch.float64
            )
        }

    def test_noop_api_apply(self, benchmark):
        """Benchmark apply via from_tesseract_api."""
        benchmark(apply_tesseract, self.tess, self.inputs)

    def test_noop_api_cast_float64(self, benchmark):
        """Benchmark apply with float64 input (expects float32)."""
        benchmark(apply_tesseract, self.tess, self.inputs_f64)


class TestVectoraddApi:
    """Benchmarks for vectoradd Tesseract via from_tesseract_api."""

    @pytest.fixture(autouse=True)
    def setup_inputs(self, vectoradd_tesseract_api, array_size):
        self.tess = vectoradd_tesseract_api
        self.a_v = create_test_array(array_size)
        self.b_v = create_test_array(array_size)

    def test_vectoradd_api_apply(self, benchmark):
        """Benchmark forward pass of vectoradd."""
        inputs = _make_vectoradd_inputs(self.a_v, self.b_v)
        benchmark(apply_tesseract, self.tess, inputs)

    def test_vectoradd_api_backward(self, benchmark):
        """Benchmark reverse-mode AD (.backward()) of vectoradd c w.r.t. a."""
        a = torch.as_tensor(self.a_v, dtype=torch.float32).requires_grad_(True)
        b = np.asarray(self.b_v, dtype=np.float32)

        def do_backward():
            result = apply_tesseract(self.tess, {"a": a, "b": b})
            result["c"].sum().backward()
            if a.grad is not None:
                a.grad = None

        benchmark(do_backward)
