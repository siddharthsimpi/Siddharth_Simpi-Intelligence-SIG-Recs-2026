@echo off
REM Windows version of run_all.sh : build + execute the 4 notebooks on the REAL dataset.
REM Optional before running:  set SHOPEE_DATA_DIR=D:\data\shopee-product-matching   and/or   set SHOPEE_MAX_GROUPS=3000
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python tools\build_notebooks.py || exit /b 1
for %%P in (PartA PartB PartC Finale) do (
  echo === %%P ===
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 %%P\notebook.ipynb || (echo FAILED in %%P & exit /b 1)
)
echo Done. See the results folders and Finale\report.pdf
