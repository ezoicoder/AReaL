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
        help="Prefix length for tree training tests",
    )


@pytest.fixture(scope="session")
def max_tokens_per_mb(request):
    """Fixture to get max_tokens_per_mb from command line or use default."""
    return request.config.getoption("--max-tokens-per-mb")

@pytest.fixture(scope="session")
def is_gradient_checkpointing(request):
    """Fixture to get disable_gradient_checkpointing from command line or use default."""
    return not request.config.getoption("--disable-gradient-checkpointing")

@pytest.fixture(scope="session")
def prefix_len(request):
    """Fixture to get prefix_len from command line or use default."""
    return request.config.getoption("--prefix-len")

