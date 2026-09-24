"""Generates the corrected notebooks from the cell text below.
Run:  python Corrected-Notebooks/_build_notebooks.py
Editing the cells here (not in the .ipynb files) keeps the notebooks reviewable."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = r'''import os, sys

def _find_common():
    """Find maml_common.py: this folder when run locally, /kaggle/input on Kaggle."""
    for root in [os.getcwd(), "/kaggle/input"]:
        if os.path.isdir(root):
            for d, _, files in os.walk(root):
                if "maml_common.py" in files:
                    return d
    raise FileNotFoundError("maml_common.py not found. Run from the Corrected-Notebooks folder, "
                            "or attach that folder to Kaggle as a Dataset.")

sys.path.insert(0, _find_common())
import maml_common as sc
SMOKE = os.environ.get("SMAP_SMOKE") == "1"     # tiny settings for testing only
OUT = sc.output_dir(smoke=SMOKE)
print("shared code:", sc.__file__)
print("outputs go to:", OUT)
print("code version:", sc.git_commit())'''


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, os.path.join(HERE, name))
    print("wrote", name)


# =============================================================================
# 01 — data exploration
# =============================================================================
write("Notebook_01_SMAP_Data_Exploration.ipynb", [
md(r'''
# Notebook 01 — SMAP data exploration

*Corrected version of the original `01_dataExploration.ipynb`.*

**What this notebook does.** It looks at the NASA SMAP telemetry release before any
modelling: how many channels there are, their shapes, their value ranges, how much of each
test file is labelled anomalous, and how many normal windows each channel can provide.

**Why.** Every later notebook depends on these facts. The original exploration got several
of them wrong, and some of its conclusions were carried forward.

**Input.** `labeled_anomalies.csv` and the `train/` and `test/` folders of the telemanom
release. Found automatically (local repository folder `SMAP - NASA`, or `/kaggle/input`).

**Output.** Two figures and `exploration_summary.json` in the output folder.

### What was corrected

1. **P-2 is listed twice** in `labeled_anomalies.csv`, with two different anomaly ranges.
   The original counted 82 channels and 55 SMAP channels. There are 81 unique channels and
   **54 unique SMAP channels**. The two P-2 rows are now merged.
2. **"Already normalised: Yes" was wrong.** The original checked only the maximum of one
   channel (P-1, range -1 to 1). Across all channels the values run from about -1.5 to 258.
   All channels are now checked, minimum and maximum.
3. **Anomaly ranges are inclusive** at both ends. The original mask dropped the last
   timestep of every range.
4. **Shot feasibility was measured on the wrong thing.** The original counted anomaly
   *sequences* per channel and concluded that no channel supports 5-shot. The final design
   adapts on K **normal** windows, so feasibility is now measured by normal windows.
5. The claim that 12% anomalies "proves" supervised learning is impossible was removed;
   the numbers are reported without that interpretation.
6. Paths no longer depend on `os.chdir('..')` and a Windows folder layout.
'''),
code(BOOT + r'''

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pd.set_option("display.width", 140)'''),
md(r'''
## 1 — The labels file

Each row describes one channel: its spacecraft (SMAP or MSL) and the anomalous ranges in
its test file. We first read the raw rows, look for channels listed more than once, and
then build one merged row per channel.
'''),
code(r'''
csv, TRAIN_DIR, TEST_DIR = sc.find_smap_raw()
raw = pd.read_csv(csv)
print("rows in the CSV:", len(raw))
print(raw["spacecraft"].value_counts().to_string())
dups = raw[raw["chan_id"].duplicated(keep=False)]
print("\nchannels listed more than once:")
print(dups[["chan_id", "spacecraft", "anomaly_sequences", "num_values"]].to_string(index=False))

labels = sc.load_labels(csv)
print("\nunique channels:", len(labels))
print(labels["spacecraft"].value_counts().to_string())
print("\nmerged P-2 ranges:", labels.loc["P-2", "sequences"])'''),
md(r'''
## 2 — Shapes and number of features

SMAP channels should all have 25 features; MSL channels have 55. The model in this project
takes 25 features, so only SMAP channels are used later.
'''),
code(r'''
rows = []
for ch in labels.index:
    tr = np.load(os.path.join(TRAIN_DIR, f"{ch}.npy"))
    te = np.load(os.path.join(TEST_DIR, f"{ch}.npy"))
    rows.append({"channel": ch, "spacecraft": labels.loc[ch, "spacecraft"],
                 "train_len": tr.shape[0], "test_len": te.shape[0], "n_features": tr.shape[1],
                 "train_min": tr.min(), "train_max": tr.max(), "test_min": te.min(), "test_max": te.max()})
stats = pd.DataFrame(rows).set_index("channel")
print(stats.groupby("spacecraft").agg(channels=("train_len", "size"),
                                      features=("n_features", lambda s: sorted(set(s))),
                                      mean_train_len=("train_len", "mean"),
                                      mean_test_len=("test_len", "mean")).round(0).to_string())
assert (stats.loc[stats.spacecraft == "SMAP", "n_features"] == 25).all()
assert (stats.spacecraft == "SMAP").sum() == 54'''),
md(r'''
## 3 — Are the values already scaled to [0, 1]?

The original looked at one channel and concluded yes. Here every channel is checked.
'''),
code(r'''
lo = stats[["train_min", "test_min"]].min(axis=1)
hi = stats[["train_max", "test_max"]].max(axis=1)
outside = stats[(lo < 0) | (hi > 1)]
print(f"overall range: {lo.min():.4f} to {hi.max():.4f}")
print(f"channels with values outside [0, 1]: {len(outside)} of {len(stats)} "
      f"(SMAP: {(outside.spacecraft == 'SMAP').sum()})")
print("channels with the largest maximum:")
print(hi.sort_values(ascending=False).head(5).round(3).to_string())
print("\nConclusion from the data:", "values are NOT all in [0, 1]; scaling is needed"
      if len(outside) else "all values are in [0, 1]")'''),
md(r'''
## 4 — Two example channels

The blue line is the first feature of the test file; red bands are the labelled anomalies.
P-1 is the channel the original plotted; D-3 is one of the evaluation channels.
'''),
code(r'''
def plot_channel(ch, path):
    te = np.load(os.path.join(TEST_DIR, f"{ch}.npy"))
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(te[:, 0], lw=0.8, color="steelblue", label="feature 0")
    for s, e in labels.loc[ch, "sequences"]:
        ax.axvspan(s, e + 1, color="red", alpha=0.25)
    ax.set_title(f"{ch}: test file, feature 0, labelled anomalies shaded")
    ax.set_xlabel("timestep"); ax.set_ylabel("raw value")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print("saved", path)

for ch in ["P-1", "D-3"]:
    plot_channel(ch, os.path.join(OUT, f"nb01_channel_{ch}.png"))'''),
md(r'''
## 5 — How much of each test file is anomalous

Labels are built per timestep with inclusive range ends. We report the pooled fraction
(all SMAP test timesteps together) and the mean of per-channel fractions, because they
differ. For the seven evaluation channels we also note whether the anomaly runs to the end
of the file, which gives those channels a high anomaly fraction.
'''),
code(r'''
prev = []
for ch in labels.index:
    n = int(stats.loc[ch, "test_len"])
    y = sc.point_labels(labels.loc[ch, "sequences"], n)
    last_end = max(e for _, e in labels.loc[ch, "sequences"])
    prev.append({"channel": ch, "spacecraft": labels.loc[ch, "spacecraft"], "test_len": n,
                 "anomalous_points": int(y.sum()), "fraction": y.mean(),
                 "anomaly_reaches_end": last_end >= n - 1})
prev = pd.DataFrame(prev).set_index("channel")
for sp, g in prev.groupby("spacecraft"):
    print(f"{sp}: pooled anomaly fraction {g.anomalous_points.sum() / g.test_len.sum():.4f}, "
          f"mean of per-channel fractions {g.fraction.mean():.4f}")
print("\nevaluation channels (and P-4, reported separately):")
print(prev.loc[sc.EVAL_CHANNELS + sc.SENSITIVITY_CHANNELS,
               ["test_len", "anomalous_points", "fraction", "anomaly_reaches_end"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(prev.loc[prev.spacecraft == "SMAP", "fraction"], bins=20, color="steelblue", edgecolor="white")
ax.set_xlabel("anomalous fraction of the test file"); ax.set_ylabel("SMAP channels")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "nb01_anomaly_fraction.png"), dpi=120); plt.close(fig)'''),
md(r'''
## 6 — What the few-shot design actually needs

The model adapts on K **normal** windows (length 30) taken from the channel's training
file, with K = 1, 5 or 10. Anomalies are only used for scoring. So the question is how many
normal windows each channel has. Windows are cut every 10 timesteps, as in the original.
Channels with fewer than 40 normal windows cannot give a 10-shot support set and still
keep normal data aside.
'''),
code(r'''
smap = stats[stats.spacecraft == "SMAP"].copy()
smap["normal_windows_stride10"] = (smap.train_len - sc.WINDOW) // sc.STRIDE_NORMAL + 1
print(smap["normal_windows_stride10"].describe().round(1).to_string())
print("\nSMAP channels with fewer than 40 normal windows:")
print(smap.loc[smap.normal_windows_stride10 < 40, ["train_len", "normal_windows_stride10"]].to_string())'''),
md(r'''
## 7 — Save the summary
'''),
code(r'''
summary = {
    "csv_rows": int(len(raw)), "unique_channels": int(len(labels)),
    "unique_smap_channels": int((labels.spacecraft == "SMAP").sum()),
    "duplicate_rows": dups["chan_id"].tolist(),
    "value_range_all_channels": [float(lo.min()), float(hi.max())],
    "channels_outside_0_1": int(len(outside)),
    "smap_pooled_anomaly_fraction": float(prev[prev.spacecraft == "SMAP"].anomalous_points.sum()
                                          / prev[prev.spacecraft == "SMAP"].test_len.sum()),
    "smap_mean_channel_anomaly_fraction": float(prev[prev.spacecraft == "SMAP"].fraction.mean()),
    "eval_channels": {c: {"test_fraction": float(prev.loc[c, "fraction"]),
                          "anomaly_reaches_end": bool(prev.loc[c, "anomaly_reaches_end"]),
                          "normal_windows_stride10": int(smap.loc[c, "normal_windows_stride10"])}
                      for c in sc.EVAL_CHANNELS + sc.SENSITIVITY_CHANNELS},
    "smap_channels_under_40_normal_windows": smap.index[smap.normal_windows_stride10 < 40].tolist(),
}
print(sc.save_json(os.path.join(OUT, "exploration_summary.json"), summary))'''),
])

# =============================================================================
# 02 — preprocessing
# =============================================================================
write("Notebook_02_SMAP_Preprocessing.ipynb", [
md(r'''
# Notebook 02 — SMAP preprocessing

*Corrected version of the original `02_preprocessing.ipynb`. It also replaces
`02c_smap_generate_channel_data_kaggle.ipynb`, which was a Kaggle copy of the same steps:
this notebook finds its inputs on Kaggle by itself.*

**What this notebook does.** It scales every SMAP channel, cuts windows, builds per-timestep
labels for the test files, fixes the channel split, and saves everything for the later
notebooks.

**Why.** The original preprocessing produced the data behind the manuscript's SMAP table.
The corrected version keeps the same scaling and window rules, so the old numbers can be
reproduced, and adds what a fair evaluation needs: the full scaled test series with
per-timestep labels.

**Input.** The raw SMAP release (found automatically).

**Output.** `smap_prepared.pkl`, `smap_split.json` and `smap_preprocessing_config.json`.

### What was corrected

1. **The split no longer depends on cell order.** The original shuffled all 81 channels
   (SMAP and MSL) twice with Python's global random state, and the original notebook 04 then removed the
   MSL channels, leaving a 39 / 6 / 9 split. That split is kept (so results stay comparable)
   but is now written out explicitly in `maml_common.py`.
2. **Only SMAP is processed.** MSL channels have 55 features and were never usable by the
   25-feature model.
3. **P-2's two CSV rows are merged.** The original used only the first row. P-2 is a
   meta-training channel, so this does not change any evaluation result.
4. **`ast.literal_eval` replaces `eval`** for reading the anomaly ranges.
5. **The full scaled test series and per-timestep labels are saved.** The original saved
   only windows: anomaly windows from the test file and normal windows from the training
   file. Scoring normal windows from one file against anomaly windows from another mixes
   up "anomalous" with "comes from the test period" (see the audit in the README).
6. The stale `channel_data.pkl` (unscaled) and a `config.json` whose query sizes (10
   anomalies, 50 normals) were never used are no longer produced.
7. The exclusions of D-12 and P-4 are recorded with their reasons. The P-4 reason came from
   looking at test anomalies, so it is marked as post hoc and P-4 is still scored separately.

**Kept exactly as before:** MinMax scaling fitted on each channel's training file only,
clipping to [0, 1], window length 30, normal windows every 10 steps from the training
file, legacy anomaly windows every 5 steps from `test[start : end + 30]`. (The original
also capped normal windows at 500 per channel; no SMAP channel has more than 286, so the
cap never applied.)
'''),
code(BOOT + r'''

import pickle
import numpy as np, pandas as pd'''),
md(r'''
## 1 — Load and prepare all SMAP channels
'''),
code(r'''
data, labels = sc.build_smap_dataset()
print("SMAP channels prepared:", len(data))
assert len(data) == 54
assert all(d["train"].shape[1] == 25 for d in data.values())'''),
md(r'''
## 2 — The channel split

Meta-train channels are used to learn the starting point, meta-validation channels to pick
checkpoints, and meta-test channels only for the final scoring.
'''),
code(r'''
split = {"meta_train": sc.META_TRAIN, "meta_val": sc.META_VAL, "meta_test": sc.META_TEST,
         "eval_channels": sc.EVAL_CHANNELS, "sensitivity_channels": sc.SENSITIVITY_CHANNELS,
         "excluded": sc.EXCLUDED}
tr, va, te = map(set, (sc.META_TRAIN, sc.META_VAL, sc.META_TEST))
assert not (tr & va or tr & te or va & te), "split overlaps"
assert tr | va | te == set(data), "split does not cover exactly the 54 SMAP channels"
print({k: len(v) for k, v in split.items() if isinstance(v, list)})
for ch, why in sc.EXCLUDED.items():
    print(f"excluded {ch}: {why}")'''),
md(r'''
## 3 — How much clipping happens

Scaling uses the training file's range. Test values outside that range are clipped to
[0, 1], as in the original. Clipping can shrink large anomalies, so we report how often it
happens on the evaluation channels.
'''),
code(r'''
csv, TRAIN_DIR, TEST_DIR = sc.find_smap_raw()
rows = []
for ch in sc.EVAL_CHANNELS + sc.SENSITIVITY_CHANNELS:
    d = data[ch]
    te_raw = np.load(os.path.join(TEST_DIR, f"{ch}.npy"))
    rng = np.where(d["scaler_max"] > d["scaler_min"], d["scaler_max"] - d["scaler_min"], 1.0)
    unclipped = (te_raw - d["scaler_min"]) / rng
    clipped = (unclipped < 0) | (unclipped > 1)
    y = d["labels"].astype(bool)
    rows.append({"channel": ch, "clipped_values_all": clipped.mean(),
                 "clipped_values_in_anomalies": clipped[y].mean() if y.any() else np.nan,
                 "clipped_values_in_normal": clipped[~y].mean()})
print(pd.DataFrame(rows).set_index("channel").round(4).to_string())'''),
md(r'''
## 4 — What each evaluation channel contains
'''),
code(r'''
rows = []
for ch in sc.EVAL_CHANNELS + sc.SENSITIVITY_CHANNELS:
    d = data[ch]
    rows.append({"channel": ch, "train_len": len(d["train"]), "test_len": len(d["test"]),
                 "normal_windows": len(d["normal_windows"]),
                 "legacy_anomaly_windows": len(d["legacy_anomaly_windows"]),
                 "test_anomaly_fraction": d["labels"].mean(),
                 "ranges": labels.loc[ch, "sequences"]})
print(pd.DataFrame(rows).set_index("channel").round(3).to_string())'''),
md(r'''
## 5 — Save
'''),
code(r'''
cfg = {"window": sc.WINDOW, "stride_normal": sc.STRIDE_NORMAL, "stride_anomaly_legacy": sc.STRIDE_ANOMALY,
       "scaling": "MinMax per channel, fitted on the training file only; clipped to [0, 1]",
       "labels": "per timestep, anomaly ranges inclusive at both ends; P-2 CSV rows merged",
       "channels": len(data)}
with open(os.path.join(OUT, "smap_prepared.pkl"), "wb") as f:
    pickle.dump({"data": data, "sequences": labels["sequences"].to_dict(), "config": cfg}, f)
sc.save_json(os.path.join(OUT, "smap_split.json"), split)
sc.save_json(os.path.join(OUT, "smap_preprocessing_config.json"), cfg)
print(sorted(f for f in os.listdir(OUT) if f.startswith("smap_")))'''),
])

# =============================================================================
# 03 — LSTM autoencoder
# =============================================================================
write("Notebook_04_LSTM_Autoencoder.ipynb", [
md(r'''
# Notebook 04 — The LSTM autoencoder

*Corrected version of the original `03_lstm_autoencoder.ipynb`.*

**What this notebook does.** It builds the LSTM autoencoder used everywhere in the project,
checks its size and shapes, and runs a short sanity training on one meta-training channel.

**Why.** It confirms the model is exactly the one described in the manuscript, and shows
what a fair sanity check looks like.

**Input.** The raw SMAP release. **Output.** `lstm_ae_init_seed42.pt` and one figure.

### What was corrected

1. **The model definition now lives in `maml_common.py`** and is imported, not re-typed.
   The architecture is unchanged: 69,481 parameters at 25 features (checked by an assert).
2. **The untrained-model check was misread.** The original printed clearly different errors
   for anomaly and normal windows from an *untrained* model (0.0415 vs 0.0256) and called
   them "similar". An untrained model has learned nothing about anomalies, so any such
   difference comes from the windows themselves: which file they are cut from, and their
   amplitude. The corrected check adds normal windows from the test file to separate the
   two effects.
3. **The separation check used training windows.** The original compared anomaly windows
   against the same normal windows the model had just been trained on. The corrected check
   holds out the last 20% of the normal windows.
4. **The saved starting weights were not reproducible.** The original re-created the model
   after the sanity training without resetting the seed. The seed is now set right before.
5. **Warning added:** the saved file is an untrained starting point. In the original
   pipeline, the original notebook 05 loaded this file as its "Static-AE" baseline, so that baseline was
   an untrained network.
'''),
code(BOOT + r'''

import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sc.seed_everything(42)
print("device:", DEVICE)'''),
md(r'''
## 1 — Build the model and count parameters
'''),
code(r'''
model = sc.LSTMAutoencoder(input_size=25).to(DEVICE)
print(model)
n_params = sc.count_params(model)
print(f"trainable parameters: {n_params:,}")
assert n_params == 69481'''),
md(r'''
## 2 — Trace one batch through the model
'''),
code(r'''
x = torch.randn(8, 30, 25, device=DEVICE)
with torch.no_grad():
    h1, _ = model.encoder.lstm1(x)
    _, (h2, _) = model.encoder.lstm2(h1)
    z = model.encoder.fc(h2[-1])
    xh = model(x)
    s = model.reconstruction_error(x)
for name, t in [("input", x), ("encoder LSTM 1", h1), ("encoder LSTM 2, last state", h2[-1]),
                ("code", z), ("reconstruction", xh), ("scores (one per window)", s)]:
    print(f"{name:28s} {tuple(t.shape)}")
assert xh.shape == x.shape and s.shape == (8,)'''),
md(r'''
## 3 — An untrained model on real data

We use P-3, a meta-training channel (not an evaluation channel). Three groups of windows
are compared:

- normal windows from the **training file**,
- normal windows from the **test file** (windows with no labelled anomalous timestep),
- anomaly windows from the test file, built as in the original.

An untrained network knows nothing about anomalies. If it still scores the two test-file
groups differently from the training-file group, the difference comes from the files, not
from the anomalies.
'''),
code(r'''
data, labels = sc.build_smap_dataset(["P-3"])
d = data["P-3"]
test_w = sc.create_windows(d["test"], 30, 10)
starts = np.arange(len(test_w)) * 10
covered = np.array([d["labels"][s:s + 30].any() for s in starts])
groups = {"normal, training file": d["normal_windows"],
          "normal, test file": test_w[~covered],
          "anomaly windows, test file": d["legacy_anomaly_windows"]}
torch.manual_seed(42)
untrained = sc.LSTMAutoencoder(25).to(DEVICE)
for name, w in groups.items():
    e = sc.window_errors(untrained, w, DEVICE)
    print(f"{name:28s} n={len(w):4d}  mean error {e.mean():.4f}  sd {e.std():.4f}")'''),
md(r'''
## 4 — Short training with held-out normal windows

The model is trained for 20 epochs on the first 80% of P-3's normal training windows, in
time order. It is then checked on the held-out 20%, on normal test-file windows and on
anomaly windows, and scored point by point on the whole test file.
'''),
code(r'''
nw = d["normal_windows"]
cut = int(0.8 * len(nw))
train_w, held_w = nw[:cut], nw[cut:]
sc.seed_everything(42)
m = sc.LSTMAutoencoder(25).to(DEVICE)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
X = torch.as_tensor(train_w)
losses = []
for ep in range(20):
    m.train(); perm = torch.randperm(len(X)); tot = 0; nb = 0
    for i in range(0, len(X), 32):
        b = X[perm[i:i + 32]].to(DEVICE)
        opt.zero_grad(); l = torch.nn.functional.mse_loss(m(b), b); l.backward(); opt.step()
        tot += l.item(); nb += 1
    losses.append(tot / nb)
print(f"training loss: first epoch {losses[0]:.5f}, last epoch {losses[-1]:.5f}")
for name, w in {"held-out normal, training file": held_w, **{k: v for k, v in groups.items() if "test" in k}}.items():
    e = sc.window_errors(m, w, DEVICE)
    print(f"{name:32s} mean error {e.mean():.5f}")
pw, _ = sc.pointwise_scores(m, d["test"], DEVICE)
print(f"point-wise ROC-AUC on P-3's test file: {roc_auc_score(d['labels'], pw):.3f} "
      f"(anomalous fraction {d['labels'].mean():.3f})")'''),
md(r'''
## 5 — Learning curve and one reconstruction
'''),
code(r'''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(range(1, 21), losses, marker="o"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE")
ax[0].set_title("training loss, P-3")
with torch.no_grad():
    r = m(torch.as_tensor(held_w[:1], device=DEVICE)).cpu().numpy()
ax[1].plot(held_w[0, :, 0], label="held-out window"); ax[1].plot(r[0, :, 0], "--", label="reconstruction")
ax[1].set_title("feature 0 of a held-out normal window"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "nb04_autoencoder_quick_test.png"), dpi=120); plt.close(fig)'''),
md(r'''
## 6 — Save a reproducible untrained starting point

This file is an **untrained** random initialisation. It is useful as a fixed starting point.
It must never be used as a trained baseline.
'''),
code(r'''
sc.seed_everything(42)
init = sc.LSTMAutoencoder(25)
path = os.path.join(OUT, "lstm_ae_init_seed42.pt")
torch.save({"model_state_dict": init.state_dict(),
            "architecture": {"input_size": 25, "seq_len": 30, "hidden1": 64, "hidden2": 32, "latent": 16},
            "note": "UNTRAINED random initialisation, torch seed 42. Not a baseline.",
            "git_commit": sc.git_commit()}, path)
print("saved", path)'''),
])

# =============================================================================
# 04 — MAML training pilot
# =============================================================================
write("Notebook_05_MAML_Training_Pilot.ipynb", [
md(r'''
# Notebook 05 — MAML meta-training pilot on CPU

*Corrected version of the original `04_maml_training.ipynb` (and its Kaggle copy
`04-maml-training (3).ipynb`).*

**What this notebook does.** It meta-trains the LSTM autoencoder with first-order MAML for
a short pilot run (500 outer steps), checks that training really updates the model, and
looks at what adaptation does on the meta-validation channels.

**Why.** It is a quick, cheap check that the training loop works before the long run in
Notebook_06.

**Input.** The raw SMAP release. **Output.** `maml_pilot_seed42.pt`,
`maml_pilot_summary.json` and one figure.

### What the check of the original found

- **The uploaded `04_maml_training.ipynb` (learn2learn version) was correct.** It used
  `learn2learn.MAML(first_order=True)`, whose `clone()` keeps the link to the meta-model.
  A direct test gave gradients on all 20 parameter tensors, and the gradient matched the
  corrected loop of the original notebook 06 to a relative difference of 1e-7.
- **The Kaggle copy `04-maml-training (3).ipynb` was broken.** It deep-copied the model,
  adapted the copy, and called `.backward()` on the copy. The meta-model received no
  gradient (0 of 20 tensors) and never changed. The checkpoint that the original notebook 05 loaded
  (`maml_best_kaggle.pt`, step 5,500) came from this broken loop.

### What was corrected

1. The loop comes from `maml_common.py` (plain PyTorch, no learn2learn). The maths is the
   same first-order MAML. learn2learn is no longer maintained and does not install on
   current Python versions.
2. **Built-in checks:** after the first outer step every meta-parameter must have a
   non-zero gradient and the parameters must have moved; adaptation must move the weights;
   a flat validation curve raises a warning. The notebook also runs the broken Kaggle loop
   once to show that the checks catch it.
3. **No meta-test channel is used.** The original ran "sanity checks" on meta-test channels
   with their anomaly windows, used them to decide on more training, and diagnosed P-4 from
   them. That is selection with test labels. All checks here use meta-validation channels
   and normal data only.
4. **Validation is less noisy and no longer affects training.** The original drew one new
   random episode per validation channel at each check, from the same random stream as
   training. Validation now uses 4 fixed episodes per channel, drawn once.
5. **Inner steps set to 10**, as in the original notebooks 06–10 and in Notebook_06 to Notebook_10. The original pilot used 5.
6. The checkpoint stores plain model weights (the original stored learn2learn's
   `module.`-prefixed keys). The final cell that called an undefined `load_lstm` was removed.
'''),
code(BOOT + r'''

import copy, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
CFG = dict(seed=42, n_outer=500, val_every=50, inner_lr=0.01, inner_steps=10, outer_lr=1e-3,
           tasks_per_batch=4, support_size=20, query_size=20, val_episodes_per_task=4,
           val_seed=2024, use_scheduler=False)
if SMOKE:
    CFG.update(n_outer=4, val_every=2, val_episodes_per_task=1)
print("device:", DEVICE, "| smoke test:" if SMOKE else "|", CFG)'''),
md(r'''
## 1 — Data: meta-training and meta-validation channels only
'''),
code(r'''
data, _ = sc.build_smap_dataset(sc.META_TRAIN + sc.META_VAL)
train_windows = {c: data[c]["normal_windows"] for c in sc.META_TRAIN}
val_episodes = sc.fixed_episodes({c: data[c]["normal_windows"] for c in sc.META_VAL},
                                 CFG["val_episodes_per_task"], CFG["val_seed"],
                                 CFG["support_size"], CFG["query_size"])
print(f"meta-train channels {len(train_windows)} | meta-val channels {len(sc.META_VAL)} | "
      f"fixed validation episodes {len(val_episodes)}")'''),
md(r'''
## 2 — Check the training step before training

One outer step with the corrected loop, then one with the broken loop from the Kaggle
copy of this notebook, both starting from the same weights. The check should pass for the
first and fail for the second.
'''),
code(r'''
sc.seed_everything(0)
probe = sc.LSTMAutoencoder(25).to(DEVICE)
rng = np.random.RandomState(0)
eps = [tuple(sc.to_tensor(a, DEVICE) for a in sc.sample_episode(train_windows[c], rng))
       for c in sc.META_TRAIN[:4]]

good = copy.deepcopy(probe); opt = torch.optim.Adam(good.parameters(), lr=1e-3)
before = [p.detach().clone() for p in good.parameters()]
sc.fomaml_outer_step(good, opt, eps, 0.01, 10)
print("corrected loop:", sc.check_outer_step(good, before))

bad = copy.deepcopy(probe); opt = torch.optim.Adam(bad.parameters(), lr=1e-3)
before = [p.detach().clone() for p in bad.parameters()]
meta = 0
for s, q in eps:                         # the loop from '04-maml-training (3)'
    learner = sc.inner_adapt(bad, s, 0.01, 10)
    meta = meta + torch.nn.functional.mse_loss(learner(q), q)
opt.zero_grad(); (meta / len(eps)).backward(); opt.step()
try:
    sc.check_outer_step(bad, before)
    raise RuntimeError("the check did not catch the broken loop")
except AssertionError as e:
    print("broken loop caught:", e)'''),
md(r'''
## 3 — Meta-train (pilot)
'''),
code(r'''
sc.seed_everything(CFG["seed"])
model = sc.LSTMAutoencoder(25).to(DEVICE)
model, info = sc.train_maml(
    model, train_windows, val_episodes, DEVICE, seed=CFG["seed"], n_outer=CFG["n_outer"],
    val_every=CFG["val_every"], inner_lr=CFG["inner_lr"], inner_steps=CFG["inner_steps"],
    outer_lr=CFG["outer_lr"], tasks_per_batch=CFG["tasks_per_batch"],
    support_size=CFG["support_size"], query_size=CFG["query_size"],
    use_scheduler=CFG["use_scheduler"], ckpt_path=os.path.join(OUT, "maml_pilot_ckpt.pt"))
print("checks:", info["checks"])
print(f"best validation loss {info['best_val']:.6f} at step {info['best_step']} of {info['steps_reached']}")'''),
md(r'''
## 4 — Training curves
'''),
code(r'''
h = info["history"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot([r["step"] for r in h], [r["train_loss"] for r in h], marker=".", label="training batch loss")
ax.plot([r["step"] for r in h], [r["val_loss"] for r in h], marker="o", label="validation loss (fixed episodes)")
ax.axvline(info["best_step"], ls="--", color="green", label=f"best step {info['best_step']}")
ax.set_xlabel("outer step"); ax.set_ylabel("query MSE after adaptation"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "nb05_maml_pilot_curves.png"), dpi=120); plt.close(fig)'''),
md(r'''
## 5 — What adaptation does (meta-validation channels, normal data only)

For each validation episode we adapt the pilot model with 0 and with 10 inner steps and
compare the query loss. We also record how far the weights move. If 10 steps barely move
the weights or barely change the loss, adaptation is close to inert.
'''),
code(r'''
rows = []
for ch, s, q in val_episodes:
    st, qt = sc.to_tensor(s, DEVICE), sc.to_tensor(q, DEVICE)
    a = sc.inner_adapt(model, st, CFG["inner_lr"], CFG["inner_steps"])
    rep = sc.adaptation_report(model, a, st)
    with torch.no_grad():
        q0 = torch.nn.functional.mse_loss(model(qt), qt).item()
        q10 = torch.nn.functional.mse_loss(a(qt), qt).item()
    rows.append({"channel": ch, "query_loss_0_steps": q0, "query_loss_10_steps": q10, **rep})
import pandas as pd
adapt = pd.DataFrame(rows).groupby("channel").mean()
print(adapt.round(6).to_string())'''),
md(r'''
## 6 — Save
'''),
code(r'''
torch.save({"model_state_dict": model.state_dict(), "step": info["best_step"], "val_loss": info["best_val"],
            "config": CFG, "git_commit": sc.git_commit()}, os.path.join(OUT, "maml_pilot_seed42.pt"))
print(sc.save_json(os.path.join(OUT, "maml_pilot_summary.json"),
                   {"kind": "pilot run on meta-train/meta-val channels only; no test data used",
                    "smoke_test": SMOKE, "config": CFG, "training": info,
                    "adaptation_on_meta_val": adapt.reset_index().to_dict(orient="records")}))'''),
])

# =============================================================================
# 06 — full SMAP experiment
# =============================================================================
write("Notebook_06_SMAP_MAML_vs_Baselines.ipynb", [
md(r'''
# Notebook 06 — SMAP: MAML-AE against baselines

*Corrected version of the original `06_smap_maml_clean.ipynb` / `06-smap-maml-clean (1).ipynb`.
It also replaces the original `05-evaluation (4).ipynb`, which evaluated the checkpoint from
the broken training loop and whose saved run crashed.*

**What this notebook does.** For each training seed it meta-trains MAML, trains a static
LSTM autoencoder and an MLP autoencoder on the same meta-training channels, and scores all
methods on the seven held-out SMAP channels at 1, 5 and 10 shots. It then summarises the
results across training seeds with confidence intervals and paired differences.

**Why.** This is the experiment behind the manuscript's SMAP table. The original had flaws
that made its numbers hard to interpret (listed below).

**Input.** The raw SMAP release and the `Corrected-Notebooks` repository folder (for `maml_common.py`).
Both are found automatically under `/kaggle/input`.

**Output (in `/kaggle/working`).** One `smap_corrected_seed<seed>.json` per training
seed, `smap_corrected_summary.json`, and model checkpoints.

### How to run on Kaggle

1. Attach two Datasets: the SMAP release, and the `Corrected-Notebooks` folder of this
   repository. Turn on a GPU.
2. Use **Save Version -> Save & Run All (Commit)**.
3. The run is split by training seed. If a session times out, attach this notebook's
   previous output as an extra input and commit again: finished seeds are skipped, and an
   unfinished MAML run resumes from its last checkpoint.
4. Expected time on a Kaggle GPU: about 2.5 hours per training seed (up to 30,000 outer
   steps, based on the original run), so about 12 to 13 hours for five seeds. That exceeds
   one 12-hour session, so plan for one resume. To split the work, set `TRAIN_SEEDS` in
   the configuration cell.

### What was corrected

1. **Point-wise scoring on the test file is the main result.** The original compared
   normal windows from the *training* file with anomaly windows from the *test* file. An
   untrained random network scores 0.542 macro ROC-AUC under that protocol, higher than
   the trained models (see the audit in the README). Now every timestep of each test file
   is scored (stride-1 windows; a timestep's score is the mean error of the windows that
   cover it) against per-timestep labels.
2. **Support and query no longer overlap.** In the original, 12–48% of the normal query
   windows shared timesteps with the support windows the model had just adapted on.
3. **Adaptation steps match training.** MAML was meta-trained with 10 inner steps but
   evaluated after 5. Evaluation now uses 10 steps for every adapted model.
4. **Zero-step controls.** MAML and Static are also scored with no adaptation at all, to
   show whether adaptation changes anything.
5. **"MLP-AE" is now a trained MLP.** The original "MLP-AE" was a random MLP given 50 SGD
   steps on the K support windows, a scratch model. That model is kept under the name
   "MLP-scratch (legacy)"; the new "MLP-AE" is trained like the static LSTM-AE.
6. **An untrained random LSTM-AE is scored** as a floor that any trained model should beat.
7. **F1 is no longer set to 0** when every window is flagged; the anomaly fraction and the
   F1 of "flag everything" are reported next to every F1, and the label-free F1 is kept
   separate from the oracle best-F1.
8. **Isolation Forest at 1 shot is reported as not applicable** (one training point),
   instead of an AUROC of 0.500.
9. **Five independent training seeds** (the original trained once; its "seeds" were only
   support draws). Support draws use separate seeds and are identical across methods and
   training seeds, so comparisons are paired.
10. **Validation uses fixed episodes** and its own random stream.
11. Every result file records the configuration, seeds, code version and time.

**Kept as in the original, so that only the corrections change:** the channel split,
scaling and windows; FOMAML with inner SGD 0.01 x 10 steps, Adam 0.001, 4 tasks per
batch, 20/20 support/query, gradient clipping 1.0, up to 30,000 outer steps, validation
every 500 steps with the learning-rate halving schedule, best-validation checkpoint; the
static recipe (random 90/10 split, Adam 0.001, batch 128, up to 150 epochs, patience 15).

A **legacy view** re-scores every model with the original window protocol and 5
adaptation steps, only so that the new models can be compared with the old table. It is
not a valid test (see point 1).
'''),
code(BOOT + r'''

import copy, json, time, numpy as np, torch
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| SMOKE TEST (numbers are meaningless)" if SMOKE else "")'''),
md(r'''
## 1 — Configuration
'''),
code(r'''
CFG = dict(
    train_seeds=[42, 123, 456, 789, 1024], support_seeds=[0, 1, 2, 3, 4],
    legacy_support_seeds=[42, 123, 456, 789, 1024], k_shots=[1, 5, 10],
    channels=sc.EVAL_CHANNELS + sc.SENSITIVITY_CHANNELS,
    n_outer=30000, val_every=500, inner_lr=0.01, inner_steps=10, outer_lr=1e-3, tasks_per_batch=4,
    support_size=20, query_size=20, use_scheduler=True, val_episodes_per_task=4, val_seed=2024,
    adapt_steps=10, adapt_lr=0.01, legacy_adapt_steps=5, mlp_scratch_steps=50,
    static_max_epochs=150, static_patience=15,
)
if SMOKE:
    CFG.update(train_seeds=[42], support_seeds=[0], legacy_support_seeds=[42], k_shots=[1, 5],
               channels=["A-6", "D-6"], n_outer=4, val_every=2, val_episodes_per_task=1,
               static_max_epochs=1, static_patience=1)
TRAIN_SEEDS = CFG["train_seeds"]      # edit to split the work across Kaggle sessions
print(CFG)'''),
md(r'''
## 2 — Data
'''),
code(r'''
data, labels = sc.build_smap_dataset()
train_windows = {c: data[c]["normal_windows"] for c in sc.META_TRAIN}
pool = np.concatenate([train_windows[c] for c in sc.META_TRAIN]).astype(np.float32)
val_episodes = sc.fixed_episodes({c: data[c]["normal_windows"] for c in sc.META_VAL},
                                 CFG["val_episodes_per_task"], CFG["val_seed"],
                                 CFG["support_size"], CFG["query_size"])
test_windows = {c: sc.create_windows(data[c]["test"], sc.WINDOW, 1) for c in CFG["channels"]}
print(f"meta-train pool {pool.shape} | validation episodes {len(val_episodes)}")
for c in CFG["channels"]:
    print(f"  {c}: normal windows {len(data[c]['normal_windows'])}, test length {len(data[c]['test'])}, "
          f"anomalous fraction {data[c]['labels'].mean():.3f}")'''),
md(r'''
## 3 — Finding earlier results and checkpoints (for resuming)
'''),
code(r'''
def find_all(name):
    hits = [os.path.join(OUT, name)] if os.path.exists(os.path.join(OUT, name)) else []
    if os.path.isdir("/kaggle/input"):
        hits += [os.path.join(d, name) for d, _, f in os.walk("/kaggle/input") if name in f]
    return hits

def latest_checkpoint(name):
    best, best_step = None, -1
    for p in find_all(name):
        try:
            s = torch.load(p, map_location="cpu", weights_only=False).get("step", 0)
        except Exception:
            continue
        if s > best_step:
            best, best_step = p, s
    return best

def result_name(seed):
    return f"smap_corrected_seed{seed}{'_SMOKE' if SMOKE else ''}.json"'''),
md(r'''
## 4 — Training for one seed

MAML, the static LSTM-AE and the MLP-AE all learn from the same 39 meta-training
channels. Finished models are saved and reused.
'''),
code(r'''
def train_models(seed):
    models, info = {}, {}
    name = f"smap_maml_seed{seed}"
    done = find_all(f"{name}_best.pt")
    sc.seed_everything(seed)
    maml = sc.LSTMAutoencoder(25).to(DEVICE)
    if done:
        ck = torch.load(done[0], map_location=DEVICE, weights_only=False)
        maml.load_state_dict(ck["model_state_dict"]); info["maml"] = ck["info"]
        print("loaded", done[0])
    else:
        maml, info["maml"] = sc.train_maml(
            maml, train_windows, val_episodes, DEVICE, seed=seed, n_outer=CFG["n_outer"],
            val_every=CFG["val_every"], inner_lr=CFG["inner_lr"], inner_steps=CFG["inner_steps"],
            outer_lr=CFG["outer_lr"], tasks_per_batch=CFG["tasks_per_batch"],
            support_size=CFG["support_size"], query_size=CFG["query_size"],
            use_scheduler=CFG["use_scheduler"], ckpt_path=os.path.join(OUT, f"{name}_ckpt.pt"),
            resume_from=latest_checkpoint(f"{name}_ckpt.pt"))
        torch.save({"model_state_dict": maml.state_dict(), "info": info["maml"]},
                   os.path.join(OUT, f"{name}_best.pt"))
    models["maml"] = maml
    for key, cls in [("static", sc.LSTMAutoencoder), ("mlp", sc.MLPAutoencoder)]:
        fname = f"smap_{key}_seed{seed}.pt"
        sc.seed_everything(seed)
        m = cls(25).to(DEVICE)
        if find_all(fname):
            ck = torch.load(find_all(fname)[0], map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck["model_state_dict"]); info[key] = ck["info"]
        else:
            m, info[key] = sc.train_conventional(m, pool, DEVICE, seed=seed,
                                                 max_epochs=CFG["static_max_epochs"],
                                                 patience=CFG["static_patience"])
            torch.save({"model_state_dict": m.state_dict(), "info": info[key]}, os.path.join(OUT, fname))
        models[key] = m
    return models, info'''),
md(r'''
## 5 — Scoring for one seed

For every channel, shot count and support seed, the same K normal support windows (from
the channel's training file) are given to every method. Each adapted model scores the whole
test file point by point. The label-free threshold is the mean + 2 standard deviations of
the support-window errors (1.20 x the mean at 1 shot).
'''),
code(r'''
def support_indices(ch, k, s):
    rng = np.random.RandomState([s, k, sc.META_TEST.index(ch)])
    return np.sort(rng.choice(len(data[ch]["normal_windows"]), k, replace=False))

def score_model(model, sup, ch, steps, lr):
    adapted = sc.inner_adapt(model, sc.to_tensor(sup, DEVICE), lr, steps) if steps else model
    errs = sc.window_errors(adapted, test_windows[ch], DEVICE)
    pw = sc.pointwise_from_window_errors(errs, len(data[ch]["test"]))
    tau = sc.label_free_threshold(sc.window_errors(adapted, sup, DEVICE))
    return sc.detection_metrics(pw, data[ch]["labels"], tau), adapted

def score_iforest(sup, ch, seed):
    if len(sup) == 1:
        return {"not_applicable": "one support window: Isolation Forest cannot be fitted meaningfully"}
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed).fit(sup.reshape(len(sup), -1))
    tw = test_windows[ch]
    errs = np.concatenate([-iso.score_samples(tw[i:i + 4096].reshape(len(tw[i:i + 4096]), -1))
                           for i in range(0, len(tw), 4096)])
    pw = sc.pointwise_from_window_errors(errs, len(data[ch]["test"]))
    tau = sc.label_free_threshold(-iso.score_samples(sup.reshape(len(sup), -1)))
    return sc.detection_metrics(pw, data[ch]["labels"], tau)

def old_f1_rule(y, s, tau):
    p = (s > tau).astype(int)
    return float(f1_score(y, p, zero_division=0)) if 0 < p.sum() < len(p) else 0.0

def evaluate(models, seed):
    res = {"pointwise": {}, "legacy_window_view": {}, "adaptation": {}}
    for ch in CFG["channels"]:
        for k in CFG["k_shots"]:
            for s in CFG["support_seeds"]:
                sup = data[ch]["normal_windows"][support_indices(ch, k, s)]
                key = f"{ch}|{k}|{s}"
                r = {}
                r["MAML-AE"], a = score_model(models["maml"], sup, ch, CFG["adapt_steps"], CFG["adapt_lr"])
                res["adaptation"][f"MAML-AE|{key}"] = sc.adaptation_report(models["maml"], a, sc.to_tensor(sup, DEVICE))
                r["MAML-AE (0 steps)"], _ = score_model(models["maml"], sup, ch, 0, CFG["adapt_lr"])
                r["Static-AE"], a = score_model(models["static"], sup, ch, CFG["adapt_steps"], CFG["adapt_lr"])
                res["adaptation"][f"Static-AE|{key}"] = sc.adaptation_report(models["static"], a, sc.to_tensor(sup, DEVICE))
                r["Static-AE (0 steps)"], _ = score_model(models["static"], sup, ch, 0, CFG["adapt_lr"])
                r["MLP-AE"], _ = score_model(models["mlp"], sup, ch, CFG["adapt_steps"], CFG["adapt_lr"])
                torch.manual_seed(seed * 1000 + s)
                r["MLP-scratch (legacy)"], _ = score_model(sc.MLPAutoencoder(25).to(DEVICE), sup, ch,
                                                           CFG["mlp_scratch_steps"], CFG["adapt_lr"])
                torch.manual_seed(seed * 1000 + s)
                r["LSTM-AE untrained (floor)"], _ = score_model(sc.LSTMAutoencoder(25).to(DEVICE), sup, ch, 0, 0.0)
                r["Isolation-Forest"] = score_iforest(sup, ch, s)
                res["pointwise"][key] = r
            for s in CFG["legacy_support_seeds"]:
                d = data[ch]
                sup, q, y, _, _ = sc.legacy_eval_query(d["normal_windows"], d["legacy_anomaly_windows"], k, s)
                r = {"query_anomaly_fraction": float(y.mean())}
                for name, base, steps in [("MAML-AE", models["maml"], CFG["legacy_adapt_steps"]),
                                          ("Static-AE", models["static"], CFG["legacy_adapt_steps"])]:
                    a = sc.inner_adapt(base, sc.to_tensor(sup, DEVICE), 0.01, steps)
                    e = sc.window_errors(a, q, DEVICE)
                    tau = sc.label_free_threshold(sc.window_errors(a, sup, DEVICE))
                    r[name] = {"roc_auc": float(roc_auc_score(y, e)), "f1_old_rule": old_f1_rule(y, e, tau)}
                res["legacy_window_view"][f"{ch}|{k}|{s}"] = r
        print(f"  seed {seed}: {ch} scored")
    return res'''),
md(r'''
## 6 — Run all training seeds (finished seeds are skipped)
'''),
code(r'''
for seed in TRAIN_SEEDS:
    prior = find_all(result_name(seed))
    if prior:
        print(f"seed {seed}: result found at {prior[0]}, skipping"); continue
    t0 = time.time()
    models, info = train_models(seed)
    res = evaluate(models, seed)
    sc.save_json(os.path.join(OUT, result_name(seed)),
                 {"experiment": "SMAP, corrected (Notebook_06)", "smoke_test": SMOKE,
                  "train_seed": seed, "config": CFG, "training": info, "results": res,
                  "device": str(DEVICE), "torch": torch.__version__,
                  "wall_seconds": time.time() - t0})
    print(f"seed {seed}: saved {result_name(seed)} ({time.time() - t0:.0f}s)")'''),
md(r'''
## 7 — Summary across training seeds

For each method and shot count: the macro mean over the seven evaluation channels of the
mean over support draws, computed per training seed, then the mean, standard deviation and
95% confidence interval across training seeds. Paired differences use the training seed as
the unit, with two-sided t-tests and Wilcoxon tests. With fewer than two seeds only means
are shown. P-4 is summarised separately.
'''),
code(r'''
runs = {seed: json.load(open(find_all(result_name(seed))[0])) for seed in CFG["train_seeds"]
        if find_all(result_name(seed))}
print("training seeds available:", sorted(runs))
METHODS = ["MAML-AE", "MAML-AE (0 steps)", "Static-AE", "Static-AE (0 steps)", "MLP-AE",
           "MLP-scratch (legacy)", "LSTM-AE untrained (floor)", "Isolation-Forest"]
METRICS = ["roc_auc", "pr_auc", "f1_at_tau", "oracle_best_f1", "trivial_all_positive_f1"]
eval_ch = [c for c in sc.EVAL_CHANNELS if c in CFG["channels"]]

def per_seed(run, method, k, metric, channels):
    vals = []
    for ch in channels:
        xs = [run["results"]["pointwise"][f"{ch}|{k}|{s}"][method].get(metric) for s in CFG["support_seeds"]]
        xs = [x for x in xs if x is not None]
        if xs: vals.append(np.mean(xs))
    return float(np.mean(vals)) if len(vals) == len(channels) else None

def describe(xs):
    xs = np.array([x for x in xs if x is not None], dtype=float)
    out = {"n_seeds": len(xs), "mean": float(xs.mean()) if len(xs) else None}
    if len(xs) > 1:
        sd = xs.std(ddof=1); h = stats.t.ppf(0.975, len(xs) - 1) * sd / np.sqrt(len(xs))
        out.update(sd=float(sd), ci95=[float(xs.mean() - h), float(xs.mean() + h)])
    return out

summary = {"pointwise_macro_eval_channels": {}, "pointwise_P-4": {}, "paired_differences_roc_auc": {},
           "legacy_view_macro_roc_auc": {}}
for k in CFG["k_shots"]:
    for m in METHODS:
        for met in METRICS:
            summary["pointwise_macro_eval_channels"][f"{m}|{k}|{met}"] = describe([per_seed(r, m, k, met, eval_ch) for r in runs.values()])
            if "P-4" in CFG["channels"]:
                summary["pointwise_P-4"][f"{m}|{k}|{met}"] = describe([per_seed(r, m, k, met, ["P-4"]) for r in runs.values()])
    for a, b in [("MAML-AE", "Static-AE"), ("MAML-AE", "MAML-AE (0 steps)"), ("Static-AE", "Static-AE (0 steps)"),
                 ("MAML-AE", "LSTM-AE untrained (floor)")]:
        d = [per_seed(r, a, k, "roc_auc", eval_ch) - per_seed(r, b, k, "roc_auc", eval_ch) for r in runs.values()]
        entry = describe(d)
        if len(d) > 1:
            entry["t_test_p_two_sided"] = float(stats.ttest_1samp(d, 0.0).pvalue)
            try: entry["wilcoxon_p_two_sided"] = float(stats.wilcoxon(d).pvalue)
            except ValueError: entry["wilcoxon_p_two_sided"] = None
        summary["paired_differences_roc_auc"][f"{a} minus {b}|{k}"] = entry
    for m in ["MAML-AE", "Static-AE"]:
        summary["legacy_view_macro_roc_auc"][f"{m}|{k}"] = describe([
            np.mean([np.mean([r["results"]["legacy_window_view"][f"{ch}|{k}|{s}"][m]["roc_auc"]
                              for s in CFG["legacy_support_seeds"]]) for ch in eval_ch]) for r in runs.values()])

def fmt(e):
    if e["mean"] is None: return "n/a"
    return f"{e['mean']:.3f}" + (f" [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]" if "ci95" in e else "")

for k in CFG["k_shots"]:
    print(f"\n{k}-shot, point-wise, macro over {len(eval_ch)} channels (mean [95% CI] over training seeds)")
    print(f"{'method':28s} {'ROC-AUC':>22s} {'PR-AUC':>22s} {'F1 at tau':>22s} {'oracle best-F1':>22s}")
    for m in METHODS:
        g = lambda met: fmt(summary["pointwise_macro_eval_channels"][f"{m}|{k}|{met}"])
        print(f"{m:28s} {g('roc_auc'):>22s} {g('pr_auc'):>22s} {g('f1_at_tau'):>22s} {g('oracle_best_f1'):>22s}")
    print(f"flag-everything F1 for reference: {fmt(summary['pointwise_macro_eval_channels'][f'MAML-AE|{k}|trivial_all_positive_f1'])}")
    for key, e in summary["paired_differences_roc_auc"].items():
        if key.endswith(f"|{k}"):
            print(f"  {key.split('|')[0]:45s} {fmt(e)}  p(t)={e.get('t_test_p_two_sided')}")
sc.save_json(os.path.join(OUT, f"smap_corrected_summary{'_SMOKE' if SMOKE else ''}.json"),
             {"smoke_test": SMOKE, "train_seeds_used": sorted(runs), "config": CFG, "summary": summary})'''),
md(r'''
## 8 — How to read the results

- **MAML-AE minus Static-AE**: counts as a difference only if its 95% interval excludes 0.
- **MAML-AE minus MAML-AE (0 steps)**: if close to 0, adaptation does not change what the
  meta-learned model detects.
- **Untrained LSTM-AE (floor)**: a trained model that does not beat this floor has not
  shown that its training helps detection on these channels.
- **F1 at tau** is the only deployable F1. Compare it with the flag-everything F1, which
  depends only on the anomaly fraction. **Oracle best-F1** uses the labels to set the
  threshold and is an upper bound.
- **Legacy window view** exists only to line up with the old table. It is not a valid test.
'''),
])

# =============================================================================
# 03 — SWaT and WADI cleaning, windows, regimes
# =============================================================================
write("Notebook_03_SWaT_WADI_Cleaning.ipynb", [
md(r'''
# Notebook 03 — SWaT and WADI: cleaning, windows and operating regimes

*Corrected version of the original `02b_swat_wadi_raw_cleaning.ipynb`. It also rebuilds the
operating regimes (`swat_tasks.pkl`, `wadi_tasks.pkl`), whose code was not in the repository.*

**What this notebook does.** It reads the raw SWaT and WADI files, cleans them, downsamples
them by 10, scales them, cuts 30-step windows every 10 steps, and groups the normal windows
into operating regimes that serve as meta-learning tasks.

**Why.** Notebooks 07 to 10 depend on these arrays and regimes. The transfer experiment also
needs the cleaned but **unscaled** data, which the original never saved.

**Input.** `SWaT_Dataset_Normal_v1.xlsx`, `SWaT_Dataset_Attack_v0.xlsx` (SWaT.A1 & A2, Dec 2015)
and `WADI_14days_new.csv`, `WADI_attackdataLABLE.csv` (WADI.A2, 19 Nov 2019), found
automatically. If the arrays saved by the original pipeline are also present
(`swat_normal.npy`, ...), the new arrays are compared with them value by value.

**Output.** `swat_clean.npz`, `wadi_clean.npz` (unscaled and scaled arrays and labels),
`swat_regimes.npz`, `wadi_regimes.npz` and `plant_data_summary.json`. The `.npz` files are
derived from licensed iTrust data, so they are not committed to the repository.

### What was corrected
1. **Label cleaning is checked, not assumed.** The SWaT attack file spells its label three
   ways (`Normal`, `Attack`, `A ttack`). All whitespace is now removed before matching, every
   row must then read `normal` or `attack`, and the attack count is asserted against the
   count of all non-normal spellings (54,621 rows). The original's `!= "Normal"` rule gave
   the right count but would have silently turned any unexpected value into an attack.
2. **WADI's empty and gappy columns are found from the data** and asserted to be the four
   known empty columns, instead of being typed in by hand. The -1 = attack polarity is
   asserted (only 1 and -1 may occur) and the attack count is checked (9,977 rows).
3. **Unscaled downsampled arrays are saved** next to the scaled ones, so the transfer
   notebook can fit the target plant's scaling on the few support windows only.
4. **The regimes are rebuilt with visible code:** per-window feature means and standard
   deviations, standardised, k-means with 10 starts and a fixed seed. k is chosen by the
   silhouette score over k = 8 to 30, the range the original used (as its split files
   show); the curve from k = 2 is also reported, because the original's WADI choice (k = 8)
   sat at the lower edge of its range. Regimes under 80 windows are dropped, as before.
5. **Attack windows are assigned to regimes** (nearest centroid), so the held-out regimes
   can be used for testing. The original defined test regimes but never used them. When the
   rebuilt regimes are identical to the original ones (same k, same regime numbers and
   sizes), the original train / validation / test split of the regimes is used.
6. Window counts are reported next to the earlier ones, together with a stricter label rule
   (a window is anomalous only if at least half of its steps are attacks).
'''),
code(BOOT + r'''

import time
import numpy as np, pandas as pd
from sklearn.metrics import silhouette_score
K_GRID_ORIGINAL = list(range(8, 31))
K_GRID_DIAGNOSTIC = list(range(2, 31))
if SMOKE:
    K_GRID_ORIGINAL, K_GRID_DIAGNOSTIC = [8, 9], [2, 8, 9]
paths = {p: [sc.find_data_file(f) for f in files] for p, files in sc.PLANT_FILES.items()}
for p, fs in paths.items():
    print(p, fs)
    assert all(fs), f"raw {p} files not found; set MAML_DATA_ROOT or attach them on Kaggle"'''),
md(r'''
## 1 — SWaT: load the two workbooks and clean the labels

Reading the Excel files takes a few minutes.
'''),
code(r'''
t0 = time.time()
swat_n_raw, swat_n_lab, swat_sensors = sc.load_swat_workbook(paths["swat"][0])
swat_a_raw, swat_a_lab, swat_sensors_a = sc.load_swat_workbook(paths["swat"][1])
print(f"read in {time.time() - t0:.0f}s: normal {swat_n_raw.shape}, attack {swat_a_raw.shape}")
assert swat_sensors == swat_sensors_a and len(swat_sensors) == sc.SWAT_EXPECTED["sensors"]
assert len(swat_n_raw) == sc.SWAT_EXPECTED["normal_rows"] and len(swat_a_raw) == sc.SWAT_EXPECTED["attack_rows"]
swat_n_attack, n_variants = sc.swat_attack_mask(swat_n_lab)
swat_a_attack, a_variants = sc.swat_attack_mask(swat_a_lab)
print("label spellings in the normal file:", n_variants)
print("label spellings in the attack file:", a_variants)
assert swat_n_attack.sum() == 0, "the normal file contains attack rows"
assert swat_a_attack.sum() == sc.SWAT_EXPECTED["attack_labelled_rows"]
print("attack rows before downsampling:", int(swat_a_attack.sum()))
print("any missing values:", int(np.isnan(swat_n_raw).sum() + np.isnan(swat_a_raw).sum()))'''),
md(r'''
## 2 — WADI: drop empty columns, interpolate gaps, remap the label
'''),
code(r'''
t0 = time.time()
wadi_n_raw, wadi_a_raw, wadi_a_attack, wadi_sensors, wadi_report = sc.load_wadi(*paths["wadi"])
print(f"read in {time.time() - t0:.0f}s: normal {wadi_n_raw.shape}, attack {wadi_a_raw.shape}")
for k, v in wadi_report.items():
    print(f"  {k}: {v}")
assert len(wadi_sensors) == sc.WADI_EXPECTED["sensors"]
assert len(wadi_n_raw) == sc.WADI_EXPECTED["normal_rows"] and len(wadi_a_raw) == sc.WADI_EXPECTED["attack_rows"]
assert wadi_a_attack.sum() == sc.WADI_EXPECTED["attack_labelled_rows"]
print("attack rows before downsampling:", int(wadi_a_attack.sum()))'''),
md(r'''
## 3 — Downsample by 10 and scale

Values: every 10th row. Labels: a downsampled step is an attack if any of its 10 rows is.
Scaling: MinMax fitted on the normal stream only; attack data clipped to [0, 1]. The
unscaled downsampled arrays are kept as well.
'''),
code(r'''
plants = {}
for p, (Xn, Xa, ya, sensors) in {"swat": (swat_n_raw, swat_a_raw, swat_a_attack, swat_sensors),
                                 "wadi": (wadi_n_raw, wadi_a_raw, wadi_a_attack, wadi_sensors)}.items():
    n_u, a_u = sc.downsample_values(Xn), sc.downsample_values(Xa)
    y = sc.downsample_labels_or(ya).astype(np.int8)
    n_s, a_s, scaler = sc.minmax_scale_plant(n_u, a_u)
    plants[p] = {"normal_unscaled": n_u, "attack_unscaled": a_u, "attack_labels": y,
                 "normal_scaled": n_s, "attack_scaled": a_s, "sensors": np.array(sensors), **scaler}
    print(f"{p}: normal {n_u.shape}, attack {a_u.shape}, attack steps {int(y.sum())} "
          f"({y.mean():.3%}), attack values clipped {np.mean((a_s == 0) | (a_s == 1)) - np.mean((n_s == 0) | (n_s == 1)):+.3%} vs normal")'''),
md(r'''
## 4 — Compare with the arrays of the original pipeline (if available)

The original saved `swat_normal.npy`, `swat_attack.npy`, `swat_attack_labels.npy`,
`wadi_normal.npy`, `wadi_attack.npy` and `wadi_attack_labels.npy`. If they can be found, the
new scaled arrays must match them exactly.
'''),
code(r'''
compare = {}
for p in plants:
    for key, name in [("normal_scaled", f"{p}_normal.npy"), ("attack_scaled", f"{p}_attack.npy"),
                      ("attack_labels", f"{p}_attack_labels.npy")]:
        f = sc.find_data_file(name)
        if f is None:
            compare[name] = "old file not available"
            continue
        old = np.load(f)
        same = old.shape == plants[p][key].shape and np.array_equal(old, plants[p][key].astype(old.dtype))
        compare[name] = "identical" if same else f"DIFFERENT (max abs diff {np.abs(old.astype(float) - plants[p][key]).max():.3g})"
for k, v in compare.items():
    print(f"  {k:24s} {v}")'''),
md(r'''
## 5 — Windows and window labels

Length 30, a new window every 10 steps. "any" is the earlier rule (at least one attack step);
"half" is the stricter rule (at least half the steps are attacks).
'''),
code(r'''
windows = {}
for p, d in plants.items():
    Wn, n_starts = sc.windows_with_labels(d["normal_scaled"])
    Wa, a_starts, y_any, y_half = sc.windows_with_labels(d["attack_scaled"], d["attack_labels"])
    windows[p] = {"normal": Wn, "attack": Wa, "any": y_any, "half": y_half,
                  "normal_starts": n_starts, "attack_starts": a_starts}
    old = sc.OLD_WINDOW_COUNTS[p]
    print(f"{p}: normal windows {len(Wn)} (earlier {old['normal']}), attack-recording windows {len(Wa)} "
          f"(earlier {old['attack']}), anomalous 'any' {y_any.sum()} (earlier {old['anomalous']}), "
          f"anomalous 'half' {y_half.sum()}")'''),
md(r'''
## 6 — Operating regimes

Each normal window is summarised by its per-feature mean and standard deviation (so 102
numbers for SWaT and 246 for WADI), the summaries are standardised, and k-means is run for
each k. The silhouette score measures how well separated the clusters are (from -1 to 1).
k is chosen as the best silhouette over k = 8 to 30, the original range; the best over
k = 2 to 30 is also shown.
'''),
code(r'''
old_splits = {p: (lambda f: __import__("json").load(open(f)) if f else None)(sc.find_data_file(f"{p}_task_splits.json"))
              for p in plants}
regimes = {}
for p in plants:
    t0 = time.time()
    fit = sc.fit_regimes(windows[p]["normal"], K_GRID_DIAGNOSTIC, seed=42, n_init=10)
    curve = {k: f["silhouette"] for k, f in fit["fits"].items()}
    k_best = max(K_GRID_ORIGINAL, key=lambda k: curve[k])
    k_best_any = max(K_GRID_DIAGNOSTIC, key=lambda k: curve[k])
    f = fit["fits"][k_best]
    split = sc.split_regimes(f["labels"], min_size=80, n_val=1, n_test=2, seed=42)
    orig = sc.ORIGINAL_REGIMES[p]
    kept = {g: n for g, n in split["sizes"].items() if g not in split["dropped_small"]}
    same = k_best == orig["k"] and kept == orig["sizes"]
    if same:   # identical regimes: use the original train/val/test split
        split.update({key: orig[key] for key in ["meta_train", "meta_val", "meta_test"]},
                     source="original split (regimes reproduced exactly)")
    else:
        split["source"] = "new seeded split (regimes differ from the original)"
    regimes[p] = {"fit": fit, "k": k_best, "curve": curve, "split": split, "labels": f["labels"],
                  "centroids": f["centroids"]}
    print(f"\n{p}: {time.time() - t0:.0f}s, summary dimensions {fit['summary_dims']}")
    print(f"  chosen k (range 8-30): {k_best}, silhouette {curve[k_best]:.4f}")
    print(f"  best k over 2-30: {k_best_any}, silhouette {curve[k_best_any]:.4f}")
    print("  silhouette by k:", {k: round(v, 3) for k, v in curve.items()})
    print(f"  regime sizes: {split['sizes']} | dropped (<80): {split['dropped_small']}")
    print(f"  regimes identical to the original (same k, ids and sizes): {same}")
    print(f"  split used ({split['source']}): meta-train {split['meta_train']}, meta-val {split['meta_val']}, "
          f"meta-test {split['meta_test']}")
    if old_splits[p]:
        o = old_splits[p]
        print(f"  earlier: k {o['chosen_k']}, silhouette {o['silhouette']:.4f}, kept regime sizes "
              f"{sorted(o['regime_sizes'].values(), reverse=True)}")
        print(f"  now:     kept regime sizes {sorted([v for k, v in split['sizes'].items() if k not in split['dropped_small']], reverse=True)}")'''),
md(r'''
## 7 — Assign attack windows to regimes

Each attack-recording window goes to its nearest regime centroid. A held-out regime can be
tested only if it has at least 20 anomalous and 20 normal attack-recording windows.
'''),
code(r'''
for p in plants:
    r = regimes[p]
    r["attack_regime"] = sc.assign_regimes(windows[p]["attack"], r["fit"]["scaler_mean"],
                                           r["fit"]["scaler_scale"], r["centroids"])
    rows = []
    for g in sorted(r["split"]["sizes"]):
        m = r["attack_regime"] == g
        role = next((k for k in ["meta_train", "meta_val", "meta_test"] if g in r["split"][k]), "dropped")
        rows.append({"regime": g, "role": role, "normal_windows": r["split"]["sizes"][g],
                     "attack_windows": int(m.sum()), "anomalous_any": int(windows[p]["any"][m].sum()),
                     "normal_in_attack_file": int((m & (windows[p]["any"] == 0)).sum())})
    df = pd.DataFrame(rows).set_index("regime")
    df["testable"] = (df.anomalous_any >= 20) & (df.normal_in_attack_file >= 20)
    r["table"] = df
    print(f"\n{p}\n{df.to_string()}")'''),
md(r'''
## 8 — Save
'''),
code(r'''
summary = {"label_spellings": {"swat_attack_file": a_variants, "swat_normal_file": n_variants},
           "wadi_cleaning": wadi_report, "comparison_with_original_arrays": compare}
for p in plants:
    d, w, r = plants[p], windows[p], regimes[p]
    np.savez_compressed(os.path.join(OUT, f"{p}_clean.npz"), **d)
    np.savez_compressed(os.path.join(OUT, f"{p}_regimes.npz"), normal_regime=r["labels"],
                        attack_regime=r["attack_regime"], centroids=r["centroids"],
                        scaler_mean=r["fit"]["scaler_mean"], scaler_scale=r["fit"]["scaler_scale"])
    summary[p] = {"normal_steps": len(d["normal_scaled"]), "attack_steps": len(d["attack_scaled"]),
                  "attack_steps_labelled": int(d["attack_labels"].sum()),
                  "windows": {"normal": len(w["normal"]), "attack": len(w["attack"]),
                              "anomalous_any": int(w["any"].sum()), "anomalous_half": int(w["half"].sum()),
                              "earlier": sc.OLD_WINDOW_COUNTS[p]},
                  "regimes": {"chosen_k": r["k"], "k_range": [8, 30], "silhouette": r["curve"][r["k"]],
                              "silhouette_curve": r["curve"], "split": r["split"],
                              "summary_dims": r["fit"]["summary_dims"],
                              "table": r["table"].reset_index().to_dict(orient="records")}}
print(sc.save_json(os.path.join(OUT, "plant_data_summary.json"), summary))
print(sorted(f for f in os.listdir(OUT) if f.endswith(".npz")))'''),
])

# =============================================================================
# 07 / 08 — within-plant experiments (SWaT, WADI)
# =============================================================================
OLD_WITHIN = {
    "swat": {"Static-AE": (0.8402, 0.6933, 4.559), "MLP-AE": (0.8438, 0.7074, 7.055),
             "Isolation-Forest": (0.7983, 0.6871, 1.260), "MAML-AE 20": (0.8170, 0.6612, 6.560),
             "MAML-AE 50": (0.8168, 0.6612, 6.526), "MAML-AE 100": (0.8175, 0.6612, 6.608)},
    "wadi": {"Static-AE": (0.7167, 0.4096, 1.931), "MLP-AE": (0.6760, 0.4274, 1.531),
             "Isolation-Forest": (0.7305, 0.3760, 1.105), "MAML-AE 20": (0.7156, 0.3929, 1.599),
             "MAML-AE 50": (0.7144, 0.3941, 1.595), "MAML-AE 100": (0.7148, 0.3964, 1.597)}}


def plant_notebook(plant, number, orig, n_outer, hours):
    P = plant.upper() if plant == "wadi" else "SWaT"
    write(f"Notebook_{number:02d}_{P}_MAML_vs_Baselines.ipynb", [
md(rf'''
# Notebook {number:02d} — {P}: MAML-AE against baselines within one plant

*Corrected version of the original `{orig}`.*

**What this notebook does.** For each training seed it meta-trains MAML on the {P}
meta-training regimes, trains the baselines, and scores everything on the {P} attack
recording at 20, 50 and 100 support windows. It then summarises across training seeds.

**Why.** The original ran one training run and compared MAML with baselines that had seen
more data. Its "MAML below Static" (SWaT) and "MAML about equal to Static" (WADI) findings
need a fair and repeated test.

**Input.** The output of Notebook_03 (`{plant}_clean.npz`, `{plant}_regimes.npz`,
`plant_data_summary.json`) and this repository folder, found automatically.

**Output.** `{plant}_within_seed<seed>.json` per training seed, `{plant}_within_summary.json`,
and model checkpoints. Expected time on a Kaggle GPU: about {hours} per training seed; the
run resumes after a timeout (attach the previous output and commit again).

### What was corrected
1. **Same training data for MAML and its main comparison.** The original MAML saw only the
   meta-training regimes, while Static-AE and MLP-AE were trained on all normal windows,
   including the validation and test regimes. The main Static-AE and MLP-AE now train on
   the same meta-training windows as MAML. The original "all normal data" versions are kept,
   labelled legacy, so the old numbers can be lined up.
2. **Five independent training seeds.** The original trained once; its five "seeds" only
   changed the support draw. Support draws now use separate seeds, shared by all methods.
3. **Isolation Forest is given the same K support windows** as the other methods. The
   original fitted it on 2,000 normal windows at every shot count; that version is kept as
   legacy.
4. **Zero-step controls** for MAML and Static, and an **untrained LSTM-AE floor**.
5. **Label-free F1** (threshold = mean + 2 SD of the support errors) is reported next to the
   oracle best-F1, which uses the labels. The original reported only the oracle value.
6. **Stricter label rule** reported as a sensitivity check: a window counts as anomalous only
   if at least half its steps are attacks (the original rule: at least one step).
7. **Held-out-regime test.** The original defined test regimes but never used them. Here
   the model also adapts on normal windows from a regime it never saw in training and is
   scored on the attack windows of that regime, where at least 20 anomalous and 20 normal
   windows exist.
8. **Validation uses 16 fixed episodes** with their own seed (the single validation regime
   gave one random episode per check in the original).
9. Every result file records configuration, seeds, code version and time.

**Kept as in the original:** cleaning, windows and scaling (Notebook_03); FOMAML with inner
SGD 0.01 x 10 steps, Adam 0.001, 4 tasks, 20/20 support/query, clipping 1.0, up to
{n_outer:,} outer steps, validation every 500 steps with the learning-rate halving schedule,
best checkpoint; evaluation adapts 10 steps at 0.01; the conventional-training recipe.
'''),
code(BOOT + r'''

import json, time, numpy as np, torch
from sklearn.ensemble import IsolationForest
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| SMOKE TEST (numbers are meaningless)" if SMOKE else "")'''),
md(r'''
## 1 — Configuration
'''),
code(rf'''
PLANT = "{plant}"
CFG = dict(train_seeds=[42, 123, 456, 789, 1024], support_seeds=[0, 1, 2, 3, 4], k_shots=[20, 50, 100],
           n_outer={n_outer}, val_every=500, inner_lr=0.01, inner_steps=10, outer_lr=1e-3,
           tasks_per_batch=4, support_size=20, query_size=20, use_scheduler=True,
           val_episodes_per_task=16, val_seed=2024, adapt_steps=10, adapt_lr=0.01,
           static_max_epochs=150, static_patience=15, iforest_legacy_windows=2000,
           min_windows_per_class_heldout=20, tost_margin=0.02)
if SMOKE:
    CFG.update(train_seeds=[42], support_seeds=[0], k_shots=[20], n_outer=4, val_every=2,
               val_episodes_per_task=1, static_max_epochs=1, static_patience=1, iforest_legacy_windows=100)
TRAIN_SEEDS = CFG["train_seeds"]      # edit to split the work across Kaggle sessions
print(CFG)'''),
md(r'''
## 2 — Data and regimes
'''),
code(r'''
D = sc.load_plant(PLANT)
sp = D["split"]
train_windows = {g: D["normal"][D["normal_regime"] == g] for g in sp["meta_train"]}
val_episodes = sc.fixed_episodes({g: D["normal"][D["normal_regime"] == g] for g in sp["meta_val"]},
                                 CFG["val_episodes_per_task"], CFG["val_seed"],
                                 CFG["support_size"], CFG["query_size"])
pool_same = np.concatenate(list(train_windows.values()))
pool_all = D["normal"]
print(f"normal windows {len(D['normal'])}, attack windows {len(D['attack'])}, anomalous (any) {D['any'].sum()}, (half) {D['half'].sum()}")
print(f"regimes: train {sp['meta_train']} ({len(pool_same)} windows), val {sp['meta_val']}, test {sp['meta_test']}")
heldout = {}
for g in sp["meta_test"]:
    m = D["attack_regime"] == g
    n_anom, n_norm = int(D["any"][m].sum()), int((D["any"][m] == 0).sum())
    ok = n_anom >= CFG["min_windows_per_class_heldout"] and n_norm >= CFG["min_windows_per_class_heldout"]
    heldout[g] = {"mask": m, "pool": D["normal"][D["normal_regime"] == g], "testable": ok,
                  "anomalous": n_anom, "normal": n_norm}
    print(f"  test regime {g}: {len(heldout[g]['pool'])} normal windows; attack windows {int(m.sum())} "
          f"(anomalous {n_anom}, normal {n_norm}) -> {'testable' if ok else 'NOT testable (fewer than 20 of a class)'}")'''),
md(r'''
## 3 — Finding earlier results and checkpoints (for resuming)
'''),
code(r'''
def find_all(name):
    hits = [os.path.join(OUT, name)] if os.path.exists(os.path.join(OUT, name)) else []
    if os.path.isdir("/kaggle/input"):
        hits += [os.path.join(d, name) for d, _, f in os.walk("/kaggle/input") if name in f]
    return hits

def latest_checkpoint(name):
    best, best_step = None, -1
    for p in find_all(name):
        try:
            s = torch.load(p, map_location="cpu", weights_only=False).get("step", 0)
        except Exception:
            continue
        if s > best_step:
            best, best_step = p, s
    return best

def result_name(seed):
    return f"{PLANT}_within_seed{seed}{'_SMOKE' if SMOKE else ''}.json"'''),
md(r'''
## 4 — Training for one seed

MAML, Static-AE and MLP-AE learn from the meta-training regimes only. The legacy
Static-AE and MLP-AE learn from all normal windows, as in the original.
'''),
code(r'''
def train_conv(key, cls, pool, seed):
    fname = f"{PLANT}_{key}_seed{seed}.pt"
    sc.seed_everything(seed)
    m = cls(D["normal"].shape[2]).to(DEVICE)
    hit = find_all(fname)
    if hit:
        ck = torch.load(hit[0], map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); return m, ck["info"]
    m, info = sc.train_conventional(m, pool, DEVICE, seed=seed, max_epochs=CFG["static_max_epochs"],
                                    patience=CFG["static_patience"])
    torch.save({"model_state_dict": m.state_dict(), "info": info}, os.path.join(OUT, fname))
    return m, info

def train_models(seed):
    models, info = {}, {}
    name = f"{PLANT}_maml_seed{seed}"
    sc.seed_everything(seed)
    maml = sc.LSTMAutoencoder(D["normal"].shape[2]).to(DEVICE)
    done = find_all(f"{name}_best.pt")
    if done:
        ck = torch.load(done[0], map_location=DEVICE, weights_only=False)
        maml.load_state_dict(ck["model_state_dict"]); info["maml"] = ck["info"]
    else:
        maml, info["maml"] = sc.train_maml(
            maml, train_windows, val_episodes, DEVICE, seed=seed, n_outer=CFG["n_outer"],
            val_every=CFG["val_every"], inner_lr=CFG["inner_lr"], inner_steps=CFG["inner_steps"],
            outer_lr=CFG["outer_lr"], tasks_per_batch=min(CFG["tasks_per_batch"], len(train_windows)),
            support_size=CFG["support_size"], query_size=CFG["query_size"],
            use_scheduler=CFG["use_scheduler"], ckpt_path=os.path.join(OUT, f"{name}_ckpt.pt"),
            resume_from=latest_checkpoint(f"{name}_ckpt.pt"))
        torch.save({"model_state_dict": maml.state_dict(), "info": info["maml"]}, os.path.join(OUT, f"{name}_best.pt"))
    models["maml"] = maml
    for key, cls, pool in [("static", sc.LSTMAutoencoder, pool_same), ("mlp", sc.MLPAutoencoder, pool_same),
                           ("static_all_legacy", sc.LSTMAutoencoder, pool_all),
                           ("mlp_all_legacy", sc.MLPAutoencoder, pool_all)]:
        models[key], info[key] = train_conv(key, cls, pool, seed)
    return models, info'''),
md(r'''
## 5 — Scoring for one seed

**Whole-plant view** (the original set-up, a sanity baseline rather than a few-shot test):
support windows are drawn from all normal windows and every attack-recording window is
scored. **Held-out-regime view** (the few-shot test): support windows come from a test
regime never used in training, and only that regime's attack-recording windows are scored.
'''),
code(r'''
def metrics_both(errs, tau, mask=None):
    y_any, y_half = D["any"], D["half"]
    if mask is not None:
        errs_, y_any, y_half = errs[mask], y_any[mask], y_half[mask]
    else:
        errs_ = errs
    return {"any": sc.detection_metrics(errs_, y_any, tau), "half": sc.detection_metrics(errs_, y_half, tau)}

def adapted_scores(model, sup, steps):
    a = sc.inner_adapt(model, sc.to_tensor(sup, DEVICE), CFG["adapt_lr"], steps) if steps else model
    return sc.window_errors(a, D["attack"], DEVICE), sc.label_free_threshold(sc.window_errors(a, sup, DEVICE)), a

def iforest(fit_windows, seed):
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
    iso.fit(fit_windows.reshape(len(fit_windows), -1))
    s = -iso.score_samples(D["attack"].reshape(len(D["attack"]), -1))
    return s, sc.label_free_threshold(-iso.score_samples(fit_windows.reshape(len(fit_windows), -1)))

def score_all(models, sup, s, seed, mask=None, legacy=True):
    r, adapt = {}, {}
    for name, key, steps in [("MAML-AE", "maml", CFG["adapt_steps"]), ("MAML-AE (0 steps)", "maml", 0),
                             ("Static-AE", "static", CFG["adapt_steps"]), ("Static-AE (0 steps)", "static", 0),
                             ("MLP-AE", "mlp", CFG["adapt_steps"])]:
        e, tau, a = adapted_scores(models[key], sup, steps)
        r[name] = metrics_both(e, tau, mask)
        if steps and name in ("MAML-AE", "Static-AE"):
            adapt[name] = sc.adaptation_report(models[key], a, sc.to_tensor(sup, DEVICE))
    torch.manual_seed(seed * 1000 + s)
    floor = sc.LSTMAutoencoder(D["normal"].shape[2]).to(DEVICE)
    e, tau, _ = adapted_scores(floor, sup, 0)
    r["LSTM-AE untrained (floor)"] = metrics_both(e, tau, mask)
    e, tau = iforest(sup, s)
    r["Isolation-Forest (K support)"] = metrics_both(e, tau, mask)
    if legacy:
        for name, key in [("Static-AE all-normal (legacy)", "static_all_legacy"),
                          ("MLP-AE all-normal (legacy)", "mlp_all_legacy")]:
            e, tau, _ = adapted_scores(models[key], sup, 0)
            r[name] = metrics_both(e, tau, mask)
        pick = np.random.RandomState(s).choice(len(pool_all), min(CFG["iforest_legacy_windows"], len(pool_all)), replace=False)
        e, tau = iforest(pool_all[pick], s)
        r["Isolation-Forest 2000 normal (legacy)"] = metrics_both(e, tau, mask)
    return r, adapt

def evaluate(models, seed):
    res = {"whole_plant": {}, "heldout_regime": {}, "adaptation": {}}
    for k in CFG["k_shots"]:
        for s in CFG["support_seeds"]:
            sup = pool_all[np.sort(np.random.RandomState([s, k]).choice(len(pool_all), k, replace=False))]
            res["whole_plant"][f"{k}|{s}"], res["adaptation"][f"whole|{k}|{s}"] = score_all(models, sup, s, seed)
            for g, h in heldout.items():
                if not h["testable"] or len(h["pool"]) < k:
                    continue
                sup = h["pool"][np.sort(np.random.RandomState([s, k, g]).choice(len(h["pool"]), k, replace=False))]
                res["heldout_regime"][f"{g}|{k}|{s}"], res["adaptation"][f"heldout|{g}|{k}|{s}"] = \
                    score_all(models, sup, s, seed, mask=h["mask"], legacy=False)
        print(f"  seed {seed}: K={k} scored")
    return res'''),
md(r'''
## 6 — Run all training seeds (finished seeds are skipped)
'''),
code(r'''
for seed in TRAIN_SEEDS:
    if find_all(result_name(seed)):
        print(f"seed {seed}: result exists, skipping"); continue
    t0 = time.time()
    models, info = train_models(seed)
    res = evaluate(models, seed)
    sc.save_json(os.path.join(OUT, result_name(seed)),
                 {"experiment": f"{PLANT} within-plant, corrected", "smoke_test": SMOKE, "train_seed": seed,
                  "config": CFG, "split": sp, "heldout_testable": {str(g): {k: v for k, v in h.items() if k in ("testable", "anomalous", "normal")} for g, h in heldout.items()},
                  "training": info, "results": res, "device": str(DEVICE), "torch": torch.__version__,
                  "wall_seconds": time.time() - t0})
    print(f"seed {seed}: saved {result_name(seed)} ({time.time() - t0:.0f}s)")'''),
md(r'''
## 7 — Summary across training seeds

Values are averaged over support draws within each training seed, then summarised across
training seeds (mean, SD, 95% CI). Paired differences use the training seed as the unit:
two-sided t-test and Wilcoxon test, and a TOST equivalence test for a band of +/- 0.02
ROC-AUC. The original single-run numbers are shown for comparison.
'''),
code(rf'''
OLD = {OLD_WITHIN[plant]!r}
runs = {{s: json.load(open(find_all(result_name(s))[0])) for s in CFG["train_seeds"] if find_all(result_name(s))}}
print("training seeds available:", sorted(runs))
METHODS = ["MAML-AE", "MAML-AE (0 steps)", "Static-AE", "Static-AE (0 steps)", "MLP-AE",
           "Static-AE all-normal (legacy)", "MLP-AE all-normal (legacy)", "LSTM-AE untrained (floor)",
           "Isolation-Forest (K support)", "Isolation-Forest 2000 normal (legacy)"]
METRICS = ["roc_auc", "pr_auc", "f1_at_tau", "oracle_best_f1", "separation_ratio"]

def seed_value(run, view, method, k, metric, rule="any", regime=None):
    block = run["results"][view]
    keys = [key for key in block if key.split("|")[-2] == str(k) and (regime is None or key.split("|")[0] == str(regime))]
    vals = [block[key][method][rule].get(metric) for key in keys if method in block[key]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

summary = {{"whole_plant": {{}}, "heldout_regime": {{}}, "paired": {{}}}}
for k in CFG["k_shots"]:
    print(f"\n=== whole-plant view, K = {{k}} (mean [95% CI] across training seeds; rule 'any') ===")
    print(f"{{'method':40s}} {{'ROC-AUC':>22s}} {{'F1 at tau':>22s}} {{'oracle best-F1':>22s}} {{'separation':>12s}}")
    for m in METHODS:
        row = {{met: sc.describe_values([seed_value(r, "whole_plant", m, k, met) for r in runs.values()]) for met in METRICS}}
        row["roc_auc_half_rule"] = sc.describe_values([seed_value(r, "whole_plant", m, k, "roc_auc", "half") for r in runs.values()])
        summary["whole_plant"][f"{{m}}|{{k}}"] = row
        f = lambda e: "n/a" if e["mean"] is None else f"{{e['mean']:.4f}}" + (f" [{{e['ci95'][0]:.3f}},{{e['ci95'][1]:.3f}}]" if "ci95" in e else "")
        print(f"{{m:40s}} {{f(row['roc_auc']):>22s}} {{f(row['f1_at_tau']):>22s}} {{f(row['oracle_best_f1']):>22s}} {{f(row['separation_ratio']):>12s}}")
    pairs = [("MAML-AE", "Static-AE"), ("MAML-AE", "MAML-AE (0 steps)"), ("Static-AE", "Static-AE (0 steps)"),
             ("MAML-AE", "Static-AE all-normal (legacy)"), ("MAML-AE", "LSTM-AE untrained (floor)")]
    for a, b in pairs:
        pc = sc.paired_comparison([seed_value(r, "whole_plant", a, k, "roc_auc") for r in runs.values()],
                                  [seed_value(r, "whole_plant", b, k, "roc_auc") for r in runs.values()], CFG["tost_margin"])
        summary["paired"][f"whole|{{a}} minus {{b}}|{{k}}"] = pc
        print(f"  {{a}} minus {{b}}: mean {{pc['mean']:+.4f}}" + (f", 95% CI [{{pc['ci95'][0]:+.4f}}, {{pc['ci95'][1]:+.4f}}], "
              f"p(t) {{pc['t_test_p_two_sided']:.3f}}, TOST p {{pc.get('tost_p', float('nan')):.3f}}" if "ci95" in pc else ""))
    for g in [g for g, h in heldout.items() if h["testable"]]:
        print(f"\n--- held-out regime {{g}}, K = {{k}} ---")
        for m in METHODS[:5] + ["LSTM-AE untrained (floor)", "Isolation-Forest (K support)"]:
            e = sc.describe_values([seed_value(r, "heldout_regime", m, k, "roc_auc", regime=g) for r in runs.values()])
            summary["heldout_regime"][f"{{g}}|{{m}}|{{k}}"] = e
            if e["mean"] is not None:
                print(f"  {{m:40s}} ROC-AUC {{e['mean']:.4f}}" + (f" [{{e['ci95'][0]:.3f}}, {{e['ci95'][1]:.3f}}]" if "ci95" in e else ""))
        pc = sc.paired_comparison([seed_value(r, "heldout_regime", "MAML-AE", k, "roc_auc", regime=g) for r in runs.values()],
                                  [seed_value(r, "heldout_regime", "Static-AE", k, "roc_auc", regime=g) for r in runs.values()],
                                  CFG["tost_margin"])
        summary["paired"][f"heldout {{g}}|MAML-AE minus Static-AE|{{k}}"] = pc

print("\n=== the original single-run numbers (ROC-AUC / oracle best-F1 / separation) ===")
for m, v in OLD.items():
    print(f"  {{m:22s}} {{v}}")
sc.save_json(os.path.join(OUT, f"{{PLANT}}_within_summary{{'_SMOKE' if SMOKE else ''}}.json"),
             {{"smoke_test": SMOKE, "train_seeds_used": sorted(runs), "config": CFG, "summary": summary, "original_numbers": OLD}})'''),
md(r'''
## 8 — How to read the results

- A difference between methods counts only if its 95% interval excludes 0.
- "About equal" is supported only if the TOST p-value is below 0.05 (difference shown to lie
  within +/- 0.02 ROC-AUC); otherwise the result is inconclusive.
- MAML minus MAML (0 steps) near 0 means adaptation does not change what is detected.
- The whole-plant view is a sanity baseline; the held-out-regime view is the few-shot test.
- F1 at tau is the deployable F1; oracle best-F1 uses the labels.
'''),
    ])


plant_notebook("swat", 7, "07-swat-maml-clean (1).ipynb", 30000, "2.5 hours")
plant_notebook("wadi", 8, "08-wadi-maml-clean (1).ipynb", 5000, "30 minutes")

# =============================================================================
# 09 — cross-plant transfer
# =============================================================================
write("Notebook_09_Transfer_SWaT_WADI.ipynb", [
md(r'''
# Notebook 09 — Cross-plant transfer between SWaT and WADI

*Corrected version of the original transfer notebook `notebook46c429edf0 (1).ipynb` (called
"Notebook 09" inside it). It also does the training part of the original
`10-transfer-significance (1).ipynb`; Notebook_10 now only analyses the saved results.*

**What this notebook does.** Each plant is projected to a common 32-dimensional space. MAML
and a conventional autoencoder are trained on the source plant, adapted with K normal windows
of the target plant (K = 20, 50, 100), and scored on the target's attack recording. This is
repeated for 6 training seeds and both directions.

**Why.** Transfer was the one setting where the old paper saw a possible MAML advantage. The
original fitted the target plant's scaling and PCA on the target's **entire** normal
recording, which a few-shot method would not have. That leak is removed here.

**Input.** The output of Notebook_03 for both plants, and this repository folder.

**Output.** `transfer_seed<seed>.json` per training seed and model checkpoints. Expected time
on a Kaggle GPU: about 25 minutes per training seed for both directions (up to 5,000 outer
steps each, with early stopping), so about 2.5 hours in total.

### What was corrected
1. **Leak fix.** In the main ("leak-free") view the target's MinMax scaling, PCA and clipping
   ranges are fitted on the K support windows only (K x 30 rows; 600 rows at K = 20), built
   from the unscaled target data. The source plant still uses its full normal data, which is
   allowed. The original, leaky projection is kept as a second view, labelled legacy, so the
   old numbers can be lined up.
2. **"Scratch" is now a real scratch model:** a fresh LSTM-AE trained only on the K support
   windows (Adam, 300 full-batch steps). The original "Scratch" was a random network given 10
   SGD steps, essentially untrained. It is kept as "Scratch (legacy)".
3. **Six training seeds for all K** (the original seeded run used K = 50 only and one-sided
   tests; the single-run table used one training run).
4. **Zero-step controls** for MAML-transfer and Static-transfer, and an untrained floor.
5. **Target-static** (trained on the target's full normal data) is reported as an upper
   reference, not as a few-shot method.
6. PCA explained variance is recorded for the full-data fit and for every support-only fit.
7. Label-free F1 is reported next to the oracle best-F1.

**Kept as in the original seeded run:** PCA to 32 dimensions with min-max scaling and
clipping; FOMAML inner SGD 0.01 x 10, Adam 0.001, 4 tasks, 20/20, clipping 1.0, up to
5,000 outer steps, validation every 250 steps, early stopping after 6 checks without
improvement; conventional recipe with 120 epochs and patience 12; adaptation 10 steps at 0.01.
'''),
code(BOOT + r'''

import json, time, numpy as np, torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| SMOKE TEST (numbers are meaningless)" if SMOKE else "")'''),
md(r'''
## 1 — Configuration
'''),
code(r'''
CFG = dict(directions=[["wadi", "swat"], ["swat", "wadi"]], train_seeds=[42, 123, 456, 789, 1024, 2048],
           support_seeds=[0, 1, 2], k_shots=[20, 50, 100], dim=32,
           n_outer=5000, val_every=250, patience=6, use_scheduler=False, inner_lr=0.01, inner_steps=10,
           outer_lr=1e-3, tasks_per_batch=4, support_size=20, query_size=20, val_episodes_per_task=16,
           val_seed=2024, static_max_epochs=120, static_patience=12, adapt_steps=10, adapt_lr=0.01,
           scratch_steps=300, scratch_lr=1e-3, legacy_scratch_steps=10)
if SMOKE:
    CFG.update(train_seeds=[42], support_seeds=[0], k_shots=[20], n_outer=4, val_every=2, patience=2,
               val_episodes_per_task=1, static_max_epochs=1, static_patience=1, scratch_steps=5)
TRAIN_SEEDS = CFG["train_seeds"]
print(CFG)'''),
md(r'''
## 2 — Data and the full-data projections

For each plant, the full-data projection (MinMax, PCA to 32, min-max, clipping, all fitted on
that plant's normal data) is what the plant uses **as a source**. Used on a target, the same
projection is the original's leaky set-up.
'''),
code(r'''
P = {p: sc.load_plant(p) for p in ["swat", "wadi"]}
full = {}
for p, d in P.items():
    proj = sc.fit_projection(d["normal_rows_unscaled"], CFG["dim"])
    nw = sc.project_windows(proj, d["normal_unscaled"])
    full[p] = {"proj": proj, "normal": nw, "attack": sc.project_windows(proj, d["attack_unscaled"]),
               "tasks": {g: nw[d["normal_regime"] == g] for g in d["split"]["meta_train"]},
               "val": sc.fixed_episodes({g: nw[d["normal_regime"] == g] for g in d["split"]["meta_val"]},
                                        CFG["val_episodes_per_task"], CFG["val_seed"])}
    evr = proj["explained_variance_ratio"]
    print(f"{p}: variance kept by 32 components {proj['explained_variance']:.4%}; PC1 {evr[0]:.2%}, "
          f"first 5 {evr[:5].sum():.2%}, first 10 {evr[:10].sum():.2%}; normal windows {nw.shape}")'''),
md(r'''
## 3 — Finding earlier results and checkpoints (for resuming)
'''),
code(r'''
def find_all(name):
    hits = [os.path.join(OUT, name)] if os.path.exists(os.path.join(OUT, name)) else []
    if os.path.isdir("/kaggle/input"):
        hits += [os.path.join(d, name) for d, _, f in os.walk("/kaggle/input") if name in f]
    return hits

def latest_checkpoint(name):
    best, best_step = None, -1
    for p in find_all(name):
        try:
            s = torch.load(p, map_location="cpu", weights_only=False).get("step", 0)
        except Exception:
            continue
        if s > best_step:
            best, best_step = p, s
    return best

def result_name(seed):
    return f"transfer_seed{seed}{'_SMOKE' if SMOKE else ''}.json"'''),
md(r'''
## 4 — Training for one seed

One MAML model per source plant (meta-trained on its regimes), one conventional
autoencoder per plant (trained on all its normal windows). The conventional model of the
source is the Static-transfer starting point; the conventional model of the target is the
Target-static upper reference.
'''),
code(r'''
def train_models(seed):
    models, info = {}, {}
    for p in ["swat", "wadi"]:
        name = f"transfer_maml_{p}_seed{seed}"
        sc.seed_everything(seed)
        m = sc.LSTMAutoencoder(CFG["dim"]).to(DEVICE)
        hit = find_all(f"{name}_best.pt")
        if hit:
            ck = torch.load(hit[0], map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck["model_state_dict"]); info[f"maml_{p}"] = ck["info"]
        else:
            m, info[f"maml_{p}"] = sc.train_maml(
                m, full[p]["tasks"], full[p]["val"], DEVICE, seed=seed, n_outer=CFG["n_outer"],
                val_every=CFG["val_every"], patience=CFG["patience"], inner_lr=CFG["inner_lr"],
                inner_steps=CFG["inner_steps"], outer_lr=CFG["outer_lr"],
                tasks_per_batch=min(CFG["tasks_per_batch"], len(full[p]["tasks"])),
                use_scheduler=CFG["use_scheduler"], ckpt_path=os.path.join(OUT, f"{name}_ckpt.pt"),
                resume_from=latest_checkpoint(f"{name}_ckpt.pt"))
            torch.save({"model_state_dict": m.state_dict(), "info": info[f"maml_{p}"]}, os.path.join(OUT, f"{name}_best.pt"))
        models[f"maml_{p}"] = m
        fname = f"transfer_static_{p}_seed{seed}.pt"
        sc.seed_everything(seed)
        s = sc.LSTMAutoencoder(CFG["dim"]).to(DEVICE)
        hit = find_all(fname)
        if hit:
            ck = torch.load(hit[0], map_location=DEVICE, weights_only=False)
            s.load_state_dict(ck["model_state_dict"]); info[f"static_{p}"] = ck["info"]
        else:
            s, info[f"static_{p}"] = sc.train_conventional(s, full[p]["normal"], DEVICE, seed=seed,
                                                           max_epochs=CFG["static_max_epochs"],
                                                           patience=CFG["static_patience"])
            torch.save({"model_state_dict": s.state_dict(), "info": info[f"static_{p}"]}, os.path.join(OUT, fname))
        models[f"static_{p}"] = s
    return models, info'''),
md(r'''
## 5 — Scoring for one seed

For every direction, K and support seed, the same K target windows are used by every
method, in both views. In the leak-free view the target projection is refitted on those K
windows each time.
'''),
code(r'''
def score(model, sup, att, y, steps, lr=None):
    a = sc.inner_adapt(model, sc.to_tensor(sup, DEVICE), lr or CFG["adapt_lr"], steps) if steps else model
    tau = sc.label_free_threshold(sc.window_errors(a, sup, DEVICE))
    return sc.detection_metrics(sc.window_errors(a, att, DEVICE), y, tau)

def evaluate(models, seed):
    res = {}
    for src, tgt in CFG["directions"]:
        T = P[tgt]
        for k in CFG["k_shots"]:
            for s in CFG["support_seeds"]:
                idx = np.sort(np.random.RandomState([s, k]).choice(len(T["normal_unscaled"]), k, replace=False))
                sup_u = T["normal_unscaled"][idx]
                proj = sc.fit_projection(sup_u.reshape(-1, sup_u.shape[2]), CFG["dim"])
                views = {"leak_free": (sc.project_windows(proj, sup_u), sc.project_windows(proj, T["attack_unscaled"])),
                         "legacy_full_target_fit": (full[tgt]["normal"][idx], full[tgt]["attack"])}
                entry = {"target_projection_explained_variance": proj["explained_variance"]}
                for view, (sup, att) in views.items():
                    r = {}
                    r["MAML-transfer"] = score(models[f"maml_{src}"], sup, att, T["any"], CFG["adapt_steps"])
                    r["MAML-transfer (0 steps)"] = score(models[f"maml_{src}"], sup, att, T["any"], 0)
                    r["Static-transfer"] = score(models[f"static_{src}"], sup, att, T["any"], CFG["adapt_steps"])
                    r["Static-transfer (0 steps)"] = score(models[f"static_{src}"], sup, att, T["any"], 0)
                    torch.manual_seed(seed * 1000 + s)
                    scratch = sc.train_on_support(sc.LSTMAutoencoder(CFG["dim"]).to(DEVICE), sup, DEVICE,
                                                  seed=seed * 1000 + s, steps=CFG["scratch_steps"], lr=CFG["scratch_lr"])
                    r["Scratch"] = score(scratch, sup, att, T["any"], 0)
                    torch.manual_seed(seed * 1000 + s)
                    r["Scratch (legacy)"] = score(sc.LSTMAutoencoder(CFG["dim"]).to(DEVICE), sup, att, T["any"],
                                                  CFG["legacy_scratch_steps"])
                    torch.manual_seed(seed * 1000 + s + 7)
                    r["LSTM-AE untrained (floor)"] = score(sc.LSTMAutoencoder(CFG["dim"]).to(DEVICE), sup, att, T["any"], 0)
                    if view == "legacy_full_target_fit":
                        r["Target-static (upper reference)"] = score(models[f"static_{tgt}"], sup, att, T["any"], 0)
                    entry[view] = r
                res[f"{src}->{tgt}|{k}|{s}"] = entry
            print(f"  seed {seed}: {src}->{tgt} K={k} scored")
    return res'''),
md(r'''
## 6 — Run all training seeds (finished seeds are skipped)
'''),
code(r'''
for seed in TRAIN_SEEDS:
    if find_all(result_name(seed)):
        print(f"seed {seed}: result exists, skipping"); continue
    t0 = time.time()
    models, info = train_models(seed)
    res = evaluate(models, seed)
    sc.save_json(os.path.join(OUT, result_name(seed)),
                 {"experiment": "SWaT <-> WADI transfer, corrected", "smoke_test": SMOKE, "train_seed": seed,
                  "config": CFG, "full_fit_explained_variance": {p: full[p]["proj"]["explained_variance"] for p in full},
                  "full_fit_variance_ratio": {p: full[p]["proj"]["explained_variance_ratio"] for p in full},
                  "training": info, "results": res, "device": str(DEVICE), "torch": torch.__version__,
                  "wall_seconds": time.time() - t0})
    print(f"seed {seed}: saved {result_name(seed)} ({time.time() - t0:.0f}s)")'''),
])

# =============================================================================
# 10 — transfer significance (analysis only)
# =============================================================================
write("Notebook_10_Transfer_Significance.ipynb", [
md(r'''
# Notebook 10 — Transfer: statistics across training seeds

*Corrected version of the original `10-transfer-significance (1).ipynb`. The training now
happens in Notebook_09; this notebook only reads its result files.*

**What this notebook does.** It reads the per-seed transfer results and reports, for each
direction, K and view: the mean, standard deviation and 95% confidence interval of every
method across training seeds, paired differences with two-sided tests, and equivalence
tests. It then lines the new numbers up with the old ones.

**Input.** `transfer_seed<seed>.json` from Notebook_09 (in the output folder, or attached on
Kaggle). **Output.** `transfer_significance_summary.json`. No GPU needed.

### What was corrected
1. **Two-sided tests.** The original reported one-sided p-values ("MAML > Static"), e.g.
   p = 0.53, without saying so. All tests here are two-sided.
2. **Equivalence is tested, not assumed.** "MAML is about equal to Static" is supported only
   if a TOST equivalence test for +/- 0.02 ROC-AUC gives p < 0.05.
3. **All K and both views.** The original tested K = 50 only, and only with the leaky
   target projection.
4. **Seed-to-seed variability** (SD across training seeds) is reported for every method.
5. **Scratch is compared properly** (a model trained on the support windows), and the gap to
   the Target-static upper reference is reported.
6. PCA explained variance: full-data fit against support-only fits.
'''),
code(BOOT + r'''

import json, glob, numpy as np'''),
md(r'''
## 1 — Load the per-seed results
'''),
code(r'''
def find_results():
    pats = [os.path.join(OUT, "transfer_seed*.json")] + (["/kaggle/input/**/transfer_seed*.json"] if os.path.isdir("/kaggle/input") else [])
    files = {}
    for pat in pats:
        for f in glob.glob(pat, recursive=True):
            if SMOKE == f.endswith("_SMOKE.json"):
                files.setdefault(os.path.basename(f), f)
    return sorted(files.values())

files = find_results()
runs = {json.load(open(f))["train_seed"]: json.load(open(f)) for f in files}
assert runs, "no transfer_seed*.json found: run Notebook_09 first"
CFG = next(iter(runs.values()))["config"]
print("training seeds:", sorted(runs), "| directions:", CFG["directions"], "| K:", CFG["k_shots"])'''),
md(r'''
## 2 — Per-seed values and summaries
'''),
code(r'''
METHODS = ["MAML-transfer", "MAML-transfer (0 steps)", "Static-transfer", "Static-transfer (0 steps)",
           "Scratch", "Scratch (legacy)", "LSTM-AE untrained (floor)", "Target-static (upper reference)"]

def per_seed(run, direction, k, view, method, metric="roc_auc"):
    vals = [run["results"][f"{direction}|{k}|{s}"][view].get(method, {}).get(metric) for s in CFG["support_seeds"]]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

out = {"methods": {}, "paired": {}, "pca": {}}
fmt = lambda e: "n/a" if e["mean"] is None else f"{e['mean']:.4f}" + (f" (SD {e['sd']:.4f}) [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]" if "sd" in e else "")
for src, tgt in CFG["directions"]:
    direction = f"{src}->{tgt}"
    for view in ["leak_free", "legacy_full_target_fit"]:
        for k in CFG["k_shots"]:
            print(f"\n=== {direction}, K = {k}, view: {view} — ROC-AUC across {len(runs)} training seeds ===")
            vals = {m: [per_seed(r, direction, k, view, m) for r in runs.values()] for m in METHODS}
            for m in METHODS:
                e = sc.describe_values(vals[m])
                e["f1_at_tau"] = sc.describe_values([per_seed(r, direction, k, view, m, "f1_at_tau") for r in runs.values()])
                out["methods"][f"{direction}|{view}|{k}|{m}"] = e
                if e["mean"] is not None:
                    print(f"  {m:34s} {fmt(e)}")
            for a, b in [("MAML-transfer", "Static-transfer"), ("MAML-transfer", "Scratch"), ("Static-transfer", "Scratch"),
                         ("MAML-transfer", "MAML-transfer (0 steps)"), ("Static-transfer", "Static-transfer (0 steps)"),
                         ("Target-static (upper reference)", "MAML-transfer")]:
                if all(v is None for v in vals[a]):
                    continue
                pc = sc.paired_comparison(vals[a], vals[b], 0.02)
                out["paired"][f"{direction}|{view}|{k}|{a} minus {b}"] = pc
                if "ci95" in pc:
                    print(f"  {a} minus {b}: {pc['mean']:+.4f} [{pc['ci95'][0]:+.4f}, {pc['ci95'][1]:+.4f}], "
                          f"p(t, two-sided) {pc['t_test_p_two_sided']:.3f}, TOST p {pc.get('tost_p', float('nan')):.3f}")
    for k in CFG["k_shots"]:
        evk = [run["results"][f"{direction}|{k}|{s}"]["target_projection_explained_variance"] for run in runs.values() for s in CFG["support_seeds"]]
        out["pca"][f"{tgt}|support_only|K={k}"] = sc.describe_values(evk)
for p, v in next(iter(runs.values()))["full_fit_explained_variance"].items():
    out["pca"][f"{p}|full_normal_fit"] = v
print("\nPCA variance kept by 32 components:")
for k, v in out["pca"].items():
    print(f"  {k:28s} {v if isinstance(v, float) else fmt(v)}")'''),
md(r'''
## 3 — Next to the old numbers

The old seeded run used K = 50, 6 seeds and the leaky projection (the "legacy" view here),
with one-sided tests. Its scratch numbers came from a single run of an essentially untrained
network ("Scratch (legacy)" here).
'''),
code(r'''
OLD = {"wadi->swat": {"Target-static": 0.8116, "MAML": (0.7962, 0.0099), "Static": (0.7969, 0.0147),
                      "diff": (-0.0007, [-0.0252, 0.0239]), "scratch_single_run": {20: 0.7503, 50: 0.7563, 100: 0.6813}},
       "swat->wadi": {"Target-static": 0.7074, "MAML": (0.5742, 0.0378), "Static": (0.5953, 0.0051),
                      "diff": (-0.0211, [-0.0601, 0.0179]), "scratch_single_run": {20: 0.5039, 50: 0.4762, 100: 0.5052}},
       "pca_full_fit": {"swat": 0.9999, "wadi": 0.9733}}
for d in ["wadi->swat", "swat->wadi"]:
    if 50 not in CFG["k_shots"]:
        print("K = 50 not in this run; comparison skipped"); break
    o = OLD[d]
    for view in ["legacy_full_target_fit", "leak_free"]:
        m = out["methods"][f"{d}|{view}|50|MAML-transfer"]; s = out["methods"][f"{d}|{view}|50|Static-transfer"]
        pc = out["paired"].get(f"{d}|{view}|50|MAML-transfer minus Static-transfer", {})
        print(f"\n{d}, K = 50, {view}:")
        print(f"  MAML   old {o['MAML'][0]:.4f} (SD {o['MAML'][1]:.4f})  new {fmt(m)}")
        print(f"  Static old {o['Static'][0]:.4f} (SD {o['Static'][1]:.4f})  new {fmt(s)}")
        if "ci95" in pc:
            print(f"  MAML - Static old {o['diff'][0]:+.4f} {o['diff'][1]}  new {pc['mean']:+.4f} "
                  f"[{pc['ci95'][0]:+.4f}, {pc['ci95'][1]:+.4f}]")
    print(f"  Target-static old {o['Target-static']:.4f}  new {fmt(out['methods'][f'{d}|legacy_full_target_fit|50|Target-static (upper reference)'])}")
sc.save_json(os.path.join(OUT, f"transfer_significance_summary{'_SMOKE' if SMOKE else ''}.json"),
             {"smoke_test": SMOKE, "train_seeds": sorted(runs), "summary": out, "old_numbers": OLD})'''),
md(r'''
## 4 — How to read the results

- The **leak-free** view is the valid few-shot test; the **legacy** view reproduces the
  original set-up only for comparison.
- A difference counts only if its 95% interval excludes 0. "About equal" needs TOST p < 0.05.
- A large SD across training seeds means a single run could have landed anywhere in that
  range.
'''),
])
