# Corrected SMAP notebooks (01, 02, 03, 04, 06)

This folder holds corrected versions of the five original SMAP notebooks. The originals
are **not modified**; they stay in the `MAML_AE` repository. Every correction is listed
below, and each notebook's first cell repeats the list for that notebook.

| File | What it is |
|---|---|
| `smap_common.py` | The one shared copy of the data code, models, FOMAML loop, checks and metrics. All five notebooks import it. |
| `01_dataExploration_corrected.ipynb` | Data exploration. Executed on the real SMAP files (CPU). |
| `02_preprocessing_corrected.ipynb` | Scaling, windows, labels, split. Executed (CPU). |
| `03_lstm_autoencoder_corrected.ipynb` | Model check and sanity training. Executed (CPU). |
| `04_maml_training_corrected.ipynb` | 500-step MAML pilot on CPU, meta-train/meta-val only. Executed (CPU). |
| `06_smap_maml_corrected.ipynb` | The full SMAP experiment, 5 training seeds. **Not run yet: it needs a Kaggle GPU.** It passed a smoke test with tiny settings on CPU; smoke numbers are not results and are not saved here. |
| `tests/test_smap_common.py` | 15 unit tests (parameter count, broken-loop detection, resume, point-wise mapping, thresholds, split). |
| `audit_old_smap_protocol.py` → `audit/audit_old_smap_protocol.json` | Re-runs the **original** evaluation with the **original** checkpoints and measures its problems. Usage: `python corrected_legacy/audit_old_smap_protocol.py <folder with the old MAML_AE .pt files>` |
| `audit/verify_outer_loops.py` + `_output.txt` | Tests the three MAML training loops found in the old notebooks. |
| `_build_notebooks.py` | Generates the notebooks (edit cells here). `_run_notebook.py` executes one. |

## What the check found (plain summary)

**1. The data code reproduces the manuscript's SMAP table exactly.** With the original
checkpoints (`smap_maml_best.pt`, step 18,000; `smap_static_ae.pt`) and the original
evaluation code, the rebuilt data gives the Version B table to three decimals (one value:
0.488 against 0.487, a rounding difference). So the manuscript's SMAP table is Version B
and it came from `06-smap-maml-clean (1).ipynb`.

**2. The uploaded 04 was correct; the Kaggle copy was broken.**

| Loop | Parameters with a gradient | Meta-model changed? |
|---|---|---|
| Uploaded `04_maml_training.ipynb` (learn2learn, `first_order=True`) | 20 / 20 | yes |
| Kaggle `04-maml-training (3).ipynb` (deepcopy, `.backward()` on the copy) | 0 / 20 | **no** |
| `06_smap_maml_clean.ipynb` corrected loop | 20 / 20 | yes |

The learn2learn gradient and the 06 gradient agree to a relative difference of 1e-7. The
checkpoint notebook 05 evaluated (`maml_best_kaggle.pt`, step 5,500) came from the broken
Kaggle loop. Notebook 05's "Static-AE" was `lstm_ae_init.pt`, an untrained network. That is
the most likely origin of Version A (MAML 0.579 against Static 0.580: two untrained
models). The saved run of 05 itself crashed, so this remains likely, not confirmed.

**3. The original SMAP evaluation did not measure detection.** Numbers from
`audit/audit_old_smap_protocol.json`:

- Under the original protocol, an **untrained random LSTM-AE scores 0.542** macro ROC-AUC
  and the **mean value of a window scores 0.575**, both higher than the trained MAML
  (0.475–0.488) and Static (0.450–0.457). The reason: normal query windows come from the
  training file and anomaly windows from the test file, so a score can separate the two
  files without detecting anything.
- On four of the seven channels (E-3, D-7, T-2, D-3), 78–87% of the query windows are
  anomalies. Flagging every window would give F1 = 0.88–0.93, far above the reported F1
  (0.10–0.32). The original code also set F1 to 0 whenever every window was flagged.
- 12–15% of normal query windows (48% on A-6) share timesteps with the support windows the
  model had just adapted on.
- Point-wise scoring of the test file is not easy to beat either: the untrained network
  reaches 0.667 macro ROC-AUC (1.000 on A-6). The corrected notebook 06 therefore reports
  this untrained network as a floor that trained models must beat.

**4. Adaptation barely moves the weights.** In the 500-step CPU pilot (notebook 04,
meta-validation channels, normal data only), 10 inner SGD steps at learning rate 0.01
changed the weights by 0.004% to 0.03% of their size, and lowered the query loss by about
0.3% to 4% depending on the channel. The pilot itself trained normally (validation loss
0.0178 at step 50, 0.0097 at step 500). Whether such small adaptation changes what is
detected is what the zero-step controls in 06 will show on the full run. This is a pilot
observation, not a test result.

## Corrections, notebook by notebook

### 01 — data exploration
1. P-2 is listed twice in `labeled_anomalies.csv`: 81 unique channels, **54 SMAP** (not 82 / 55). Rows merged.
2. "Already normalised: Yes" came from checking one channel's maximum. All channels checked: values run from -1.48 to 258, so scaling is needed.
3. Anomaly ranges are inclusive; the original mask dropped the last timestep of each range.
4. Shot feasibility counted anomaly *sequences* (concluding no channel supports 5-shot). The design adapts on K *normal* windows, so feasibility is now counted in normal windows (only D-12 is short: 29).
5. The "12% proves supervised learning is impossible" claim was removed; pooled and per-channel fractions are both reported.
6. No `os.chdir('..')` or Windows paths.

### 02 — preprocessing
1. The split is written out explicitly (the same 39 / 6 / 9 channels as before). Originally it depended on two reshuffles of a global random state and on notebook 04 later removing MSL channels.
2. SMAP only (MSL has 55 features).
3. P-2 rows merged (the original used only the first row; P-2 is meta-train, so no evaluation result changes).
4. `ast.literal_eval` instead of `eval`.
5. Full scaled test series and per-timestep labels are saved, so the test file can be scored point by point.
6. Stale outputs (`channel_data.pkl`, a `config.json` with unused query sizes) are no longer produced.
7. Exclusions recorded with reasons. D-12: too few training timesteps (label-free). P-4: found by looking at test anomalies, so marked post hoc and still scored as a sensitivity row.
8. Kept identical: scaling fitted on the training file, clipping, window length 30, normal stride 10, legacy anomaly windows. (The old 500-window cap never applied; the largest SMAP channel has 286.) Clipping is now reported per channel.

### 03 — LSTM autoencoder
1. The model is imported from `smap_common.py`; 69,481 parameters asserted.
2. The untrained-model check was misread ("similar" scores that were not similar). It now also scores normal windows from the test file, which shows how much of the difference comes from the file and not the anomaly.
3. The separation check compared anomalies with the same normal windows used for training. It now uses held-out normal windows and a point-wise score of the test file.
4. The saved initialisation is seeded right before creation and labelled as untrained. (Notebook 05 had used the old `lstm_ae_init.pt` as its "Static-AE".)

### 04 — MAML training pilot
1. The uploaded learn2learn loop was correct (see above). The corrected notebook uses the equivalent plain-PyTorch loop from `smap_common.py`, because learn2learn no longer installs on current Python.
2. Built-in checks: all meta-parameters get a gradient and move; adaptation moves the weights; a flat validation curve raises a warning. The notebook runs the broken Kaggle loop once to show the check catches it.
3. **No meta-test channel is touched.** The original checked separation on meta-test anomaly windows, decided "needs more steps" from them and diagnosed P-4 from them. All diagnostics now use meta-validation normal data.
4. Validation uses 4 fixed episodes per channel with their own seed. The original drew one fresh episode per check from the training random stream, which was noisy and changed the training samples.
5. 10 inner steps (the pilot used 5; all later notebooks use 10).
6. Plain weight keys in the checkpoint; the final cell calling an undefined `load_lstm` was removed.

### 06 — full SMAP experiment
1. **Point-wise scoring of each test file is the main result** (stride-1 windows; a timestep's score is the mean error of the windows covering it). The old window protocol is kept only as a clearly labelled bridge to the old table.
2. Support windows no longer overlap the normal data being scored.
3. Evaluation adapts for 10 steps, as in meta-training (the original evaluated after 5).
4. Zero-step controls for MAML and Static.
5. "MLP-AE" is now a trained MLP. The old "MLP-AE" (random MLP + 50 steps on the support windows) is kept as "MLP-scratch (legacy)".
6. An untrained LSTM-AE is scored as a floor.
7. F1 is never zeroed; the anomaly fraction and the flag-everything F1 are reported beside it; label-free F1 and oracle best-F1 are separate.
8. Isolation Forest at 1 shot is reported as not applicable.
9. Five training seeds (the original trained once), with separate, shared support seeds so comparisons are paired. Summary gives mean, SD and 95% CI across training seeds, plus paired differences with two-sided t-tests and Wilcoxon tests.
10. Fixed validation episodes; checkpoints store the random state, so a resumed run continues the same sample stream (tested).
11. Every result file records configuration, seeds, code version and time. Resume works per seed from `/kaggle/working` or attached inputs.

Kept as in the original, so only the corrections change: the split, scaling and windows;
FOMAML settings (inner SGD 0.01 x 10, Adam 0.001, 4 tasks, 20/20, clip 1.0, up to 30,000
steps, validation every 500 with learning-rate halving, best checkpoint); the static recipe.
These differ from the pre-registered v2 protocol (early stopping every 250 steps, no
schedule); v2 will supersede this notebook.

## How to run

Locally (CPU), from the repository root:

```
pip install torch numpy pandas scikit-learn scipy matplotlib nbformat nbclient ipykernel pytest
python -m pytest corrected_legacy/tests -q
cd corrected_legacy && python _run_notebook.py 01_dataExploration_corrected.ipynb
```

The SMAP files are found in `SMAP - NASA/` automatically (or set `SMAP_ROOT`). Set
`SMAP_SMOKE=1` for a tiny test run; its files go to `corrected_legacy_out/SMOKE/`.

On Kaggle (notebook 06): upload this `corrected_legacy` folder and the SMAP release as two
Datasets, attach both, turn on a GPU, and use Save & Run All. Expect about 2.5 GPU hours
per training seed, so 12–13 hours for five: plan one resume, or set `TRAIN_SEEDS` to split
the seeds across sessions. Put the commit hash in a file called `CODE_VERSION` inside the
uploaded folder so the result files record it (Kaggle has no git).
