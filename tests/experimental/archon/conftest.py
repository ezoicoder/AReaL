"""Pytest configuration for archon tests.

NOTE: Archon engine requires PyTorch >= 2.9.1.
"""

from types import SimpleNamespace

import pytest
import torch

# Require PyTorch >= 2.9.1 for archon tests
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3])
_MIN_TORCH_VERSION = (2, 9, 1)

if _TORCH_VERSION < _MIN_TORCH_VERSION:
    collect_ignore_glob = ["test_*.py"]


def pytest_addoption(parser):
    parser.addoption(
        "--dta-data",
        type=str,
        default=None,
        help="Path to .pt file with DTA token sequences (list[Tensor]).",
    )
    parser.addoption(
        "--no-dta",
        action="store_true",
        default=False,
        help="Disable DTA.",
    )
    parser.addoption(
        "--max-tokens-per-mb",
        type=int,
        default=5596,
        help="Cap sequence length and set mb_spec.max_tokens_per_mb for archon tests.",
    )
    parser.addoption(
        "--dta-limit",
        type=int,
        default=-1,
        help="Use at most N sequences from --dta-data; -1 keeps all sequences.",
    )
    parser.addoption(
        "--use-hf",
        action="store_true",
        default=False,
        help="Use HuggingFace model for Archon DTA tests.",
    )
    parser.addoption(
        "--model-path",
        type=str,
        default="/storage/openpsi/models/Qwen__Qwen2.5-0.5B-Instruct/",
        help="Path to model.",
    )


@pytest.fixture(scope="module")
def archon_test_config(request) -> SimpleNamespace:
    """Expose archon runtime config to tests/fixtures."""
    Ans = SimpleNamespace(
        max_tokens_per_mb=int(request.config.getoption("--max-tokens-per-mb")),
        enable_dta=not request.config.getoption("--no-dta"),
        dta_data=request.config.getoption("--dta-data"),
        dta_limit=int(request.config.getoption("--dta-limit")),
        use_hf=request.config.getoption("--use-hf"),
        model_path=request.config.getoption("--model-path"),
    )
    assert not Ans.use_hf or Ans.enable_dta, (
        "Use HuggingFace model for Archon DTA tests must enable DTA."
    )
    return Ans


def pytest_collection_modifyitems(config, items):
    """Skip all archon tests if PyTorch version is too old."""
    if _TORCH_VERSION >= _MIN_TORCH_VERSION:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Archon tests require PyTorch >= 2.9.1, but found {torch.__version__}"
    )
    for item in items:
        if "experimental/archon" in str(item.fspath):
            item.add_marker(skip_marker)
