@echo off
REM Windows version of run_smoke.sh : quick self-test on synthetic data (no GPU, no downloads).
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
python tools\make_synthetic_data.py --out data\synthetic --groups 300 || exit /b 1
set SHOPEE_SMOKE=1
set SHOPEE_DATA_DIR=%cd%\data\synthetic
set SHOPEE_CACHE_DIR=%cd%\cache_smoke
set MPLBACKEND=Agg
for %%P in (PartA PartB PartC Finale) do (
  echo === %%P ===
  pushd %%P
  python notebook.py > NUL || (echo FAILED in %%P & popd & exit /b 1)
  popd
)
echo SMOKE TEST PASSED
