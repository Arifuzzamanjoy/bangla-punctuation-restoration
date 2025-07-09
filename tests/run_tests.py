#!/usr/bin/env python3
"""
Test runner script for the Bangla punctuation restoration system
"""

import unittest
import sys
import os
from io import StringIO
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_test_suite(test_type="all", verbose=True):
    """
    Run test suite
    
    Args:
        test_type: Type of tests to run ("all", "unit", "integration", "api", "data", "models")
        verbose: Whether to run in verbose mode
    """
    
    # Discover tests based on type
    if test_type == "all":
        test_dir = os.path.dirname(__file__)
        suite = unittest.TestLoader().discover(test_dir, pattern='test_*.py')
    elif test_type == "unit":
        suite = unittest.TestSuite()
        from test_data import TestDataProcessor, TestDatasetLoader, TestDatasetGenerator
        from test_models import TestBaselineModel, TestModelEvaluator
        
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataProcessor))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatasetLoader))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatasetGenerator))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBaselineModel))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModelEvaluator))
    elif test_type == "integration":
        from test_integration import TestPipelineIntegration, TestEndToEndScenarios
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPipelineIntegration))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEndToEndScenarios))
    elif test_type == "api":
        from test_api import TestFastAPIServer, TestGradioInterface
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFastAPIServer))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGradioInterface))
    elif test_type == "data":
        from test_data import TestDataProcessor, TestDatasetLoader, TestDatasetGenerator, TestAdversarialAttackGenerator
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataProcessor))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatasetLoader))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDatasetGenerator))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdversarialAttackGenerator))
    elif test_type == "models":
        from test_models import TestBaselineModel, TestAdvancedModel, TestModelEvaluator
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBaselineModel))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdvancedModel))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModelEvaluator))
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    print(f"Running {test_type} tests...")
    print("=" * 50)
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Test Summary for {test_type} tests:")
    print(f"Total tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Print failure details
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


def run_coverage_analysis():
    """Run test coverage analysis"""
    try:
        import coverage
        
        # Initialize coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        success = run_test_suite("all", verbose=False)
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        print("\n" + "=" * 50)
        print("Coverage Report:")
        print("=" * 50)
        
        # Generate console report
        cov.report(show_missing=True)
        
        # Generate HTML report
        cov.html_report(directory='coverage_html_report')
        print("\nHTML coverage report generated in 'coverage_html_report' directory")
        
        return success
        
    except ImportError:
        print("Coverage.py not installed. Install with: pip install coverage")
        return run_test_suite("all")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Bangla punctuation restoration system")
    parser.add_argument("--type", "-t", 
                       choices=["all", "unit", "integration", "api", "data", "models"],
                       default="all",
                       help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Run in verbose mode")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Run with coverage analysis")
    
    args = parser.parse_args()
    
    if args.coverage:
        success = run_coverage_analysis()
    else:
        success = run_test_suite(args.type, args.verbose)
    
    sys.exit(0 if success else 1)
