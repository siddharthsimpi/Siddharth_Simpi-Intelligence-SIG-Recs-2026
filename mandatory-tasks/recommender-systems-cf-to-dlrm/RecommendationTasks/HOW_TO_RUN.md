# How to run everything (Windows PowerShell or cmd; the same commands work on macOS/Linux)

Python 3.9+ is required. From this folder (`RecommendationTasks`):

```
pip install -r requirements.txt
python run_tests.py                       # unit tests of all three tasks (expect ALL TESTS PASSED)
python run_all.py --quick                 # smoke run of Tasks 1-3 (~3 minutes)
python run_all.py                         # full run of Tasks 1-3 (~30-40 minutes)
```
Or task by task:
```
cd 01_CollaborativeFiltering && python run_experiment.py && cd ..
cd 02_NeuralCTR              && python run_task2.py      && cd ..
cd 03_DLRM                   && python run_task3.py      && cd ..
```
Task 3 reads Task 2's selected configurations and (if present) Task 1's metrics, so run them in this order.

Task 1 downloads MovieLens-100K by itself. If that fails (certificate/proxy), download
https://files.grouplens.org/datasets/movielens/ml-100k.zip in a browser and unzip it into `01_CollaborativeFiltering\data\`
so that `01_CollaborativeFiltering\data\ml-100k\u.data` exists. Without it, Task 1 falls back to a clearly labelled synthetic dataset.
Tasks 2-3 find the supplied dataset at `..\datasets\dataset (tasks 2 and 3).zip` (repo layout) or via `--data <folder or zip>`.

Results: `*/results/report.md` in each task folder (tables, figures, conclusions); `*/results/environment.json` has seeds, versions and hardware.
