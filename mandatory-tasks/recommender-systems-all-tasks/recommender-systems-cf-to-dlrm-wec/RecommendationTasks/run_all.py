#!/usr/bin/env python
"""Run the whole pipeline in order: tests -> Task 1 -> Task 2 -> Task 3 (Task 3 then includes Tasks 1-2 in its comparison).

    python run_all.py            # full run (~25-40 min on one CPU core)
    python run_all.py --quick    # smoke run (~3 min)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
quick = ["--quick"] if "--quick" in sys.argv else []
steps = [
    ("Unit tests", [sys.executable, "run_tests.py"], ROOT),
    ("Task 1: collaborative filtering", [sys.executable, "run_experiment.py"] + quick, os.path.join(ROOT, "01_CollaborativeFiltering")),
    ("Task 2: neural CTR", [sys.executable, "run_task2.py"] + quick, os.path.join(ROOT, "02_NeuralCTR")),
    ("Task 3: DLRM + final comparison", [sys.executable, "run_task3.py"] + quick, os.path.join(ROOT, "03_DLRM")),
]
for title, cmd, cwd in steps:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"step failed: {title}")
print("\nDone. Reports: 01_CollaborativeFiltering/results/report.md, 02_NeuralCTR/results/report.md, 03_DLRM/results/report.md")
