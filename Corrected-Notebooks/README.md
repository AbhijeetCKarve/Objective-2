# Corrected Notebooks — MAML-AE for few-shot anomaly detection (SMAP, SWaT, WADI)

Corrected versions of the original MAML-AE notebooks, renumbered 01 to 10. The originals
(in the `MAML_AE` repository) are **not modified**. Each notebook's first cell says what it
does, what it needs, what it saves, and lists every correction made to the original.

## The notebooks

| Notebook | Replaces (original name) | Status of the saved outputs |
|---|---|---|
| `Notebook_01_SMAP_Data_Exploration` | `01_dataExploration` | Run in full (CPU) |
| `Notebook_02_SMAP_Preprocessing` | `02_preprocessing`, `02c_smap_generate_channel_data_kaggle` | Run in full (CPU) |
| `Notebook_03_SWaT_WADI_Cleaning` | `02b_swat_wadi_raw_cleaning` + the missing regime code | Run in full (CPU) |
| `Notebook_04_LSTM_Autoencoder` | `03_lstm_autoencoder` | Run in full (CPU) |
| `Notebook_05_MAML_Training_Pilot` | `04_maml_training`, `04-maml-training (3)` | Run in full (CPU, 500-step pilot) |
| `Notebook_06_SMAP_MAML_vs_Baselines` | `06_smap_maml_clean`, `06-smap-maml-clean (1)`, `05-evaluation (4)` | **Needs a GPU run** (about 12–13 h on Kaggle for 5 seeds) |
| `Notebook_07_SWaT_MAML_vs_Baselines` | `07-swat-maml-clean (1)` | **Needs a GPU run** (about 12–13 h for 5 seeds) |
| `Notebook_08_WADI_MAML_vs_Baselines` | `08-wadi-maml-clean (1)` | **Needs a GPU run** (about 2–3 h for 5 seeds) |
| `Notebook_09_Transfer_SWaT_WADI` | `notebook46c429edf0 (1)` (the original transfer notebook) + the training part of `10-transfer-significance (1)` | **Needs a GPU run** (about 2.5 h for 6 seeds) |
| `Notebook_10_Transfer_Significance` | `10-transfer-significance (1)` (statistics part) | Runs in seconds once Notebook_09's results exist |

Notebooks 06 to 10 have passed a smoke test (tiny settings, CPU); smoke numbers are not
results and are not saved. Their outputs will be added when the full runs are done.

Other files:

| File | What it is |
|---|---|
| `maml_common.py` | The single shared copy of the data code, models, MAML loop, checks, metrics and statistics. Every notebook imports it. |
| `tests/test_maml_common.py` | 22 unit tests (parameter count 69,481; detection of the broken training loop; resume; early stopping; label cleaning; downsampling; window labels; regimes; support-only projection; point-wise scoring; thresholds; paired tests). |
| `audit_old_smap_protocol.py` → `audit/audit_old_smap_protocol.json` | Re-runs the original SMAP evaluation with the original checkpoints and measures its problems. |
| `audit/verify_outer_loops.py` + `_output.txt` | Tests the three MAML training loops found in the original notebooks. |
| `outputs/` | Figures and JSON summaries written by the notebooks (large data files are not committed). |
| `_build_notebooks.py`, `_run_notebook.py`, `_run_all_local.sh` | Generate the notebooks from their cell text, execute one, or execute all. |

## Main findings of the check

1. **The rebuilt pipelines reproduce the originals exactly.** SMAP: with the original
   checkpoints, the manuscript's SMAP table (its "Version B") is reproduced to three
   decimals. SWaT and WADI: all six cleaned arrays of the original are identical value for
   value, the window counts match (SWaT 4,948 / 4,497 / 648; WADI 7,843 / 1,726 / 140), and
   the rebuilt operating regimes are the original ones (same k, regime numbers and sizes).
2. **Only the Kaggle copy of the training loop was broken.** The learn2learn notebook
   `04_maml_training` trains correctly; `04-maml-training (3)` gives the meta-model no
   gradient (0 of 20 parameter tensors). Notebook 05 of the original evaluated a checkpoint
   from the broken loop against an untrained "Static-AE", the likely origin of the
   manuscript's other SMAP table (0.579 against 0.580).
3. **The original SMAP evaluation did not measure detection.** It compared normal windows
   from the training file with anomaly windows from the test file. An untrained network scores
   0.542 macro ROC-AUC under it, and the mean value of a window 0.575, both above the trained
   MAML (about 0.48) and Static (about 0.45). On four channels, flagging everything would give
   F1 0.88–0.93. 12–48% of normal query windows overlapped the support windows.
4. **Within-plant comparisons were unequal.** SWaT/WADI MAML was trained on the meta-training
   regimes only, but Static-AE and MLP-AE on all normal data including the test regimes; all
   from one training run.
5. **The transfer experiment leaked target data** (the target's scaling and PCA were fitted on
   its whole normal recording), its "Scratch" baseline was an untrained network, and its
   p-values were one-sided.
6. **Silhouette over a wider range:** the best k over 2–30 is 2 for SWaT (silhouette 0.886)
   and 3 for WADI (0.223); the original searched only 8–30. With the original regime split,
   WADI's two test regimes have 3 and 4 anomalous windows, so a held-out-regime test is not
   possible on WADI; SWaT's two test regimes have 28 and 20.

## How to run

### Locally (CPU)

```
pip install torch numpy pandas scikit-learn scipy matplotlib openpyxl nbformat nbclient ipykernel pytest
python -m pytest tests -q
export MAML_DATA_ROOT=/path/to/folder/with/the/raw/data     # SMAP - NASA, SWAT, WADI folders
python _run_notebook.py Notebook_01_SMAP_Data_Exploration.ipynb
```

Data is searched for under `/kaggle/input`, the folders in `MAML_DATA_ROOT` (separated by
`:`), and the folder above this repository. If the original `MAML_AE` arrays
(`swat_normal.npy`, ...) are also on that path, Notebook_03 checks the new arrays against
them. Set `SMAP_SMOKE=1` for a tiny test run; its files go to `outputs/SMOKE/`.

### On Kaggle (notebooks 06 to 10)

1. Create private Kaggle Datasets: (a) this repository folder, with a file `CODE_VERSION`
   holding the commit hash; (b) the SMAP release; (c) the SWaT and WADI raw files (iTrust
   licence: keep private); (d) the output of Notebook_03 (`*_clean.npz`, `*_regimes.npz`,
   `plant_data_summary.json`).
2. Attach what each notebook needs: 06 needs (a)+(b); 07, 08 and 09 need (a)+(d); 10 needs
   (a) plus the output of 09. Turn on a GPU.
3. Use Save & Run All (Commit). Runs are split by training seed; after a timeout, attach the
   notebook's previous output and commit again: finished seeds are skipped and unfinished
   training resumes from its checkpoint. `TRAIN_SEEDS` in the configuration cell splits the
   work across sessions.

## What these notebooks are, and are not

They fix the original experiments while keeping their settings (steps, learning rates,
schedules), so that the corrections alone explain any change in the numbers. They are not
the pre-registered v2 study (see the v2 plan), which adds a hyperparameter grid, Reptile,
ANIL, the dose-response experiments and early stopping everywhere.
