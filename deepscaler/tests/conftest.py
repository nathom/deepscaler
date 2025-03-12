"""
Configuration file for pytest.

This file contains global fixtures and configurations for the DeepScaler test suite.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root directory to the path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Define custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "optional: mark test as optional (may be skipped if dependencies not available)"
    )

# Skip optional tests by default unless the --run-optional flag is provided
def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--run-optional", action="store_true", default=False, 
        help="run optional tests that require additional dependencies"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--run-optional"):
        skip_optional = pytest.mark.skip(reason="needs --run-optional option to run")
        for item in items:
            if "optional" in item.keywords:
                item.add_marker(skip_optional)