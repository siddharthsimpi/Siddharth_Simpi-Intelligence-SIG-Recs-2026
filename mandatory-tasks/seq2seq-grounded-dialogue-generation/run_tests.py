#!/usr/bin/env python
"""Run every unit test without needing pytest:   python run_tests.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
suites = ["test_autograd.py", "test_text.py", "test_models.py", "test_train_embed.py", "test_data.py"]
failed = 0
for s in suites:
    print(f"== {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", s)], cwd=ROOT)
    failed += r.returncode != 0
print("\nALL TESTS PASSED" if not failed else f"\n{failed} test file(s) FAILED")
sys.exit(1 if failed else 0)
