#!/usr/bin/env python3
"""Unified test runner for BirdNET India.

Executes all unit tests, CLI tests, model verification, and real-audio tests.
"""

import subprocess
import sys
import time


def main():
    print("=" * 65)
    print(" AvianAI India — Comprehensive Test Suite")
    print(" Developed & Engineered by Mrutyunjay Joshi")
    print("=" * 65)

    start_t = time.perf_counter()

    print("\n[Running Pytest Suite]...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    res = subprocess.run(cmd)

    print("\n" + "=" * 65)
    elapsed = time.perf_counter() - start_t
    if res.returncode == 0:
        print(f" [PASS] ALL TESTS PASSED SUCCESSFULLY! (Completed in {elapsed:.2f}s)")
    else:
        print(f" [FAIL] Some tests failed. Exit code: {res.returncode}")
    print("=" * 65)

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
