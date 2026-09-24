#!/bin/bash
# Real runs of the CPU-sized notebooks, then smoke tests of the GPU-sized ones.
cd "$(dirname "$0")"
export MAML_DATA_ROOT=${MAML_DATA_ROOT:-/home/user/abhijeetckarve/maml_ae}
for n in 01_SMAP_Data_Exploration 02_SMAP_Preprocessing 03_SWaT_WADI_Cleaning 04_LSTM_Autoencoder 05_MAML_Training_Pilot; do
  s=$(date +%s); python3 _run_notebook.py Notebook_$n.ipynb 7200 2>&1 | grep -v WARNING | tail -3; echo "REAL $n $(( $(date +%s)-s ))s"
done
for n in 06_SMAP_MAML_vs_Baselines 07_SWaT_MAML_vs_Baselines 08_WADI_MAML_vs_Baselines 09_Transfer_SWaT_WADI 10_Transfer_Significance; do
  cp Notebook_$n.ipynb _smoke_$n.ipynb
  s=$(date +%s); SMAP_SMOKE=1 python3 _run_notebook.py _smoke_$n.ipynb 3600 2>&1 | grep -v WARNING | tail -3; echo "SMOKE $n $(( $(date +%s)-s ))s"
done
