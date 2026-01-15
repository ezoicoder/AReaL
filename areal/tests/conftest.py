"""Pytest configuration for AReaL tests.

This file contains shared fixtures and configuration options for all tests.
"""

import pytest

# Default max_tokens_per_mb for tree training tests
DEFAULT_MAX_TOKENS_PER_MB = 2048


def pytest_addoption(parser):
    """Add custom command line options for pytest.
    
    Usage:
        python -m pytest areal/tests/test_tree_training.py::test_fsdp_tree_training_forward --max-tokens-per-mb=4096 -v -s
    """
    parser.addoption(
        "--max-tokens-per-mb",
        action="store",
        default=DEFAULT_MAX_TOKENS_PER_MB,
        type=int,
        help=f"Specify max_tokens_per_mb for tree training tests (default: {DEFAULT_MAX_TOKENS_PER_MB})"
    )


@pytest.fixture(scope="module")
def max_tokens_per_mb(request):
    """Fixture to get max_tokens_per_mb from command line argument."""
    from areal.utils import logging
    logger = logging.getLogger("pytest")
    value = request.config.getoption("--max-tokens-per-mb")
    logger.info(f"Using max_tokens_per_mb={value}")
    return value

