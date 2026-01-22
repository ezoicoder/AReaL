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


@pytest.fixture
def max_tokens_per_mb(request):
    """Fixture to get max_tokens_per_mb from command line or use default."""
    return request.config.getoption("--max-tokens-per-mb")

