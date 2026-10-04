#!/usr/bin/env python
"""Run everything in order: tests -> sub-task 1 -> 2 -> 3 -> final report.

    python run_all.py                  # full run (hours on a CPU: see the time budgets)
    python run_all.py --quick --toy    # ~5 minute smoke test on SYNTHETIC data (not real results)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
extra = [a for a in sys.argv[1:] if a in ("--quick", "--toy")]
steps = [
    ("Unit tests", [sys.executable, "run_tests.py"], ROOT),
    ("Sub-task 1: basic seq2seq", [sys.executable, "run_subtask1.py"] + extra, os.path.join(ROOT, "subtask1")),
    ("Sub-task 2: attention + decoding", [sys.executable, "run_subtask2.py"] + extra, os.path.join(ROOT, "subtask2")),
    ("Sub-task 3: grounded Hinglish dialogue", [sys.executable, "run_subtask3.py"] + extra, os.path.join(ROOT, "subtask3")),
    ("Final report", [sys.executable, "make_final_report.py"], ROOT),
]
for title, cmd, cwd in steps:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"step failed: {title}")
print("\nDone. See FINAL_REPORT.md and subtask*/results/report.md")
