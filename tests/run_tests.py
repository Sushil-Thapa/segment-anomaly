"""
Test runner script that executes all test suites.
"""

import sys
import subprocess
from pathlib import Path


def run_test_file(test_file):
    """Run a specific test file."""
    test_path = Path(__file__).parent / test_file

    if not test_path.exists():
        print(f"Test file not found: {test_file}")
        return False

    print(f"\n{'='*50}")
    print(f"Running {test_file}")
    print("=" * 50)

    try:
        # Run the test file
        result = subprocess.run(
            [sys.executable, str(test_path)], capture_output=False, check=True
        )
        print(f"✓ {test_file} passed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ {test_file} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {test_file} failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("Swin-UNet Wafer Defect Segmentation - Test Suite")
    print("=" * 60)

    test_files = [
        "test_tiling.py",
        "test_metrics.py",
        "test_memory.py",
        "test_integration.py",
    ]

    passed = 0
    total = len(test_files)

    for test_file in test_files:
        if run_test_file(test_file):
            passed += 1

    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} test files passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        print("\nNote: Import errors are expected if dependencies are not installed.")
        print("Run 'pip install -r requirements.txt' to install dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
