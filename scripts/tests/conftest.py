"""
Pytest conftest for scripts/tests.

Ensures the tests/ directory is on sys.path so that the attention package
(scripts/tests/attention/) can be imported when running pytest from project root.
"""
import os
import sys

# Add scripts/tests to path so that "attention" is a top-level package
# when test files use: from .common import ...
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
