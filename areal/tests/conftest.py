"""pytest configuration for areal tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--max-tokens-per-mb",
        action="store",
        default=16384,
        type=int,
        help="Maximum tokens per microbatch for tree packing tests",
    )
    parser.addoption(
        "--disable-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Disable gradient checkpointing for all tests",
    )
    parser.addoption(
        "--prefix-len",
        action="store",
        default=-1,
        type=int,
        help="Number of sequences to keep (-1 = all)",
    )
    parser.addoption(
        "--disable-dfn-mask",
        action="store_true",
        default=False,
        help="Disable DFN (depth-first numbering) O(B) mask; fall back to dense O(B^2) mask",
    )
    parser.addoption(
        "--model-path",
        action="store",
        default="/data/jiarui/dta/models/Qwen2.5-0.5B",
        type=str,
        help="Path to the model checkpoint (local path or HuggingFace ID fallback)",
    )
    parser.addoption(
        "--data-path",
        action="store",
        default="/data/jiarui/dta/AReaL/DynamicTreeAttn/data/call10.pt",
        type=str,
        help="Path to tree training data (.pt file)",
    )


@pytest.fixture(scope="session")
def max_tokens_per_mb(request):
    """Fixture to get max_tokens_per_mb from command line."""
    return request.config.getoption("--max-tokens-per-mb")


@pytest.fixture(scope="session")
def is_gradient_checkpointing(request):
    """Fixture: True unless --disable-gradient-checkpointing is passed."""
    return not request.config.getoption("--disable-gradient-checkpointing")


@pytest.fixture(scope="session")
def prefix_len(request):
    """Fixture to get prefix_len from command line."""
    return request.config.getoption("--prefix-len")


@pytest.fixture(scope="session")
def use_dfn_mask(request):
    """Fixture: True (DFN mask enabled) unless --disable-dfn-mask is passed."""
    return not request.config.getoption("--disable-dfn-mask")


@pytest.fixture(scope="session")
def model_path(request):
    """Fixture to get resolved model_path with HuggingFace fallback."""
    from areal.tests.utils import get_model_path

    path = request.config.getoption("--model-path")
    return get_model_path(path, "Qwen/Qwen2-0.5B")


@pytest.fixture(scope="session")
def data_path(request):
    """Fixture to get data_path from command line."""
    return request.config.getoption("--data-path")
