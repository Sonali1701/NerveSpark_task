#!/usr/bin/env python3
"""
Test runner for the E-commerce Recommendation System.

This script runs all unit and integration tests.
"""
import unittest
import sys
import os
from pathlib import Path

def run_tests():
    """Discover and run all tests in the tests directory."""
    # Add the src directory to the Python path
    src_dir = str(Path(__file__).parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return success (0) or failure (1)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    # Run the tests
    sys.exit(run_tests())
