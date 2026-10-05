# Task 3 solution notes: DLRM

(Assignment text: `README.md`. Paper summary: `PAPER_NOTES.md`. Generated results: `results/report.md`.)

## Run (after Task 2, which provides the selected baseline configurations)
```
python run_task3.py --quick      # ~1 min smoke run
python run_task3.py              # full run, ~15 min on one CPU core
```

## What is implemented
`ctrlib/models.py::DLRM`: bottom MLP on dense features -> vector of dimension d; one embedding table per categorical field (dimension d);
interaction layer = all pairwise dot products between the 27 vectors (351 values, self-products excluded); top MLP over
concat[dense vector, dot products] -> logit. Forward and backward (including the interaction layer) are hand-written and checked
against finite differences. No pretrained weights, no DLRM library.

## Experiments
1. **Tuning** (validation only, same budget as the MLP/DCN grids): embedding dimension, MLP sizes, dropout, weight decay.
2. **Fair final comparison**: LogReg, FM (the matrix-factorization family generalised to all fields), vanilla MLP and DCN (configs selected in Task 2),
   and DLRM, on the same split, preprocessing, seeds and metrics, with parameter counts, memory, training time and inference time.
3. **Ablations** around the selected DLRM: no interaction (concatenation instead of dot products), embedding dimension {4, 8, 16, 32},
   dense-MLP depth, top-MLP size.
4. **Cross-task summary**: how the progression neighbours -> matrix factorization -> MLP -> DCN -> DLRM behaves, what each step adds and costs.

## Output files (`results/`)
`report.md`, `dlrm_grid.csv`, `final_comparison.csv`, `ablations.csv`, `environment.json`, `figures/`.
