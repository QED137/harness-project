"""Shared pytest configuration.

Tests marked `live` call the real OpenAI API and cost money. They are skipped unless
pytest is started with --live, no matter which -m filter is used.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", help="run tests that call the real OpenAI API (costs money)"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="calls the real OpenAI API; run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
