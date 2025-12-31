"""Pytest configuration and shared fixtures."""

import pytest

# Don't import fixtures here - let pytest discover them via pytest_plugins
# from tests.fixtures.dataset_manager import (...)  # ❌ This causes early import

# Make fixtures available to all tests via plugin system
pytest_plugins = ["tests.fixtures.dataset_manager"]


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--skip-download",
        action="store_true",
        default=False,
        help="Skip dataset download (unit tests only)",
    )
    parser.addoption(
        "--force-download",
        action="store_true",
        default=False,
        help="Force re-download of all datasets",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_dataset: mark test as requiring dataset download"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip tests that require datasets if --skip-download is used."""
    if config.getoption("--skip-download"):
        skip_dataset = pytest.mark.skip(reason="--skip-download option used")
        for item in items:
            if "requires_dataset" in item.keywords:
                item.add_marker(skip_dataset)
