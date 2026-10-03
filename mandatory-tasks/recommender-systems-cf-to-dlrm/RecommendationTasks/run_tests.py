#!/usr/bin/env python
"""Run every unit test of all three tasks without needing pytest:  python run_tests.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
suites = [
    ("Task 1 (collaborative filtering)", os.path.join(ROOT, "01_CollaborativeFiltering", "tests", "test_cf.py")),
    ("Tasks 2-3 models / gradient checks", os.path.join(ROOT, "tests", "test_models.py")),
    ("Tasks 2-3 data / metrics / training", os.path.join(ROOT, "tests", "test_data_metrics.py")),
]
failed = 0
for name, path in suites:
    print(f"== {name}")
    r = subprocess.run([sys.executable, path], cwd=ROOT)
    failed += r.returncode != 0
print("\nALL TESTS PASSED" if not failed else f"\n{failed} test file(s) FAILED")
sys.exit(1 if failed else 0)
