# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests
from tesseract_core import Tesseract

here = Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


@pytest.fixture(scope="module")
def vectoradd_tess() -> Tesseract:
    """Load the simple vector-addition test Tesseract."""
    return Tesseract.from_tesseract_api(
        here / "vectoradd_tesseract" / "tesseract_api.py"
    )


@pytest.fixture(scope="module")
def nested_tess() -> Tesseract:
    """Load the nested-schema test Tesseract."""
    return Tesseract.from_tesseract_api(here / "nested_tesseract" / "tesseract_api.py")


@pytest.fixture(scope="module")
def forwardonly_tess() -> Tesseract:
    """Load a Tesseract that only has an apply endpoint (no JVP/VJP)."""
    return Tesseract.from_tesseract_api(
        here / "forwardonly_tesseract" / "tesseract_api.py"
    )


@pytest.fixture(scope="module")
def dict_tess() -> Tesseract:
    """Tesseract with dict-valued differentiable fields on both sides."""
    return Tesseract.from_tesseract_api(here / "dict_tesseract" / "tesseract_api.py")


# ---------------------------------------------------------------------------
# GPU (cuda_ipc) serving
# ---------------------------------------------------------------------------
#
# Cross-process CUDA IPC needs the Tesseract (producer) and the test process
# (consumer) to be *separate* processes sharing the GPU -- a process cannot
# open an IPC handle it exported itself. So the GPU test Tesseract is served
# via a bare ``tesseract-runtime serve`` subprocess (not Docker): running on
# the host trivially shares the GPU and IPC namespace with the test process,
# with no ``--ipc=host`` container flag needed.


def _find_free_port() -> int:
    """Find a free port to use for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _serve_tesseract(
    tmp_path_factory, api_path, *, name: str, extra_env: dict | None = None
):
    """Start a tesseract-runtime server and yield its URL.

    ``extra_env`` merges additional environment variables into the server
    process (e.g. the cuda_ipc opt-in for the GPU fixture).
    """
    port = _find_free_port()
    timeout = 10

    output_dir = tmp_path_factory.mktemp(f"tesseract_output_{name}")

    env = os.environ.copy()
    env["TESSERACT_API_PATH"] = str(api_path)
    env["TESSERACT_OUTPUT_PATH"] = str(output_dir)
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        ["tesseract-runtime", "serve", "--host", "localhost", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def _server_output() -> str:
        """Collect whatever the server process has printed so far."""
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        return (
            f"--- {name} server stdout ---\n{stdout.decode(errors='replace')}\n"
            f"--- {name} server stderr ---\n{stderr.decode(errors='replace')}"
        )

    try:
        start_time = time.time()
        while True:
            # Fail fast (and surface the reason) if the server already crashed.
            if process.poll() is not None:
                raise RuntimeError(
                    f"Tesseract {name!r} server exited early with code "
                    f"{process.returncode}\n{_server_output()}"
                )
            try:
                requests.get(f"http://localhost:{port}/health")
                break
            except requests.exceptions.ConnectionError as exc:
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Tesseract {name!r} did not start in time\n{_server_output()}"
                    ) from exc
                time.sleep(0.1)

        yield f"http://localhost:{port}"
    finally:
        process.terminate()
        process.communicate()


@pytest.fixture(scope="module")
def served_gpu_tesseract(tmp_path_factory):
    """A served GPU Tesseract with cuda_ipc enabled. Skips without a CUDA GPU."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")

    gen = _serve_tesseract(
        tmp_path_factory,
        here / "gpu_tesseract" / "tesseract_api.py",
        name="gpu",
        extra_env={
            # cuda_ipc output is an experimental opt-in in tesseract-core.
            "TESSERACT_ENABLE_EXPERIMENTAL_CUDA_IPC": "1",
        },
    )
    url = next(gen)
    try:
        yield Tesseract.from_url(url)
    finally:
        gen.close()
