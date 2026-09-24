"""Generates the corrected notebooks from the cell text below.
Run:  python corrected_legacy/_build_notebooks.py
Editing the cells here (not in the .ipynb files) keeps the notebooks reviewable."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

BOOT = r'''import os, sys

def _find_common():
    """Find smap_common.py: this folder when run locally, /kaggle/input on Kaggle."""
    for root in [os.getcwd(), "/kaggle/input"]:
        if os.path.isdir(root):
            for d, _, files in os.walk(root):
                if "smap_common.py" in files:
                    return d
    raise FileNotFoundError("smap_common.py not found. Run from the corrected_legacy folder, "
                            "or attach that folder to Kaggle as a Dataset.")

sys.path.insert(0, _find_common())
import smap_common as sc
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
write("01_dataExploration_corrected.ipynb", [
md(r'''
# 01 — SMAP data exploration (corrected)

**What this notebook does.** It looks at the NASA SMAP telemetry release before any
modelling: how many channels there are, their shapes, their value ranges, how much of each
test file is labelled anomalous, and how many normal windows each channel can provide.

**Why.** Every later notebook depends on these facts. The original exploration got several
of them wrong, and some of its conclusions were carried forward.

**Input.** `labeled_anomalies.csv` and the `train/` and `test/` folders of the telemanom
release. Found automatically (local repository folder `SMAP - NASA`, or `/kaggle/input`).

**Output.** Two figures and `exploration_summary.json` in the output folder.

### What was corrected, compared with the original `01_dataExploration.ipynb`

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
    plot_channel(ch, os.path.join(OUT, f"01_channel_{ch}.png"))'''),
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
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_anomaly_fraction.png"), dpi=120); plt.close(fig)'''),
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
write("02_preprocessing_corrected.ipynb", [
md(r'''
# 02 — SMAP preprocessing (corrected)

**What this notebook does.** It scales every SMAP channel, cuts windows, builds per-timestep
labels for the test files, fixes the channel split, and saves everything for the later
notebooks.

**Why.** The original preprocessing produced the data behind the manuscript's SMAP table.
The corrected version keeps the same scaling and window rules, so the old numbers can be
reproduced, and adds what a fair evaluation needs: the full scaled test series with
per-timestep labels.

**Input.** The raw SMAP release (found automatically).

**Output.** `smap_prepared.pkl`, `smap_split.json` and `smap_preprocessing_config.json`.

### What was corrected, compared with the original `02_preprocessing.ipynb`

1. **The split no longer depends on cell order.** The original shuffled all 81 channels
   (SMAP and MSL) twice with Python's global random state, and notebook 04 then removed the
   MSL channels, leaving a 39 / 6 / 9 split. That split is kept (so results stay comparable)
   but is now written out explicitly in `smap_common.py`.
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
write("03_lstm_autoencoder_corrected.ipynb", [
md(r'''
# 03 — The LSTM autoencoder (corrected)

**What this notebook does.** It builds the LSTM autoencoder used everywhere in the project,
checks its size and shapes, and runs a short sanity training on one meta-training channel.

**Why.** It confirms the model is exactly the one described in the manuscript, and shows
what a fair sanity check looks like.

**Input.** The raw SMAP release. **Output.** `lstm_ae_init_seed42.pt` and one figure.

### What was corrected, compared with the original `03_lstm_autoencoder.ipynb`

1. **The model definition now lives in `smap_common.py`** and is imported, not re-typed.
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
   pipeline, notebook 05 loaded this file as its "Static-AE" baseline, so that baseline was
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
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_autoencoder_quick_test.png"), dpi=120); plt.close(fig)'''),
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
write("04_maml_training_corrected.ipynb", [
md(r'''
# 04 — MAML meta-training pilot on CPU (corrected)

**What this notebook does.** It meta-trains the LSTM autoencoder with first-order MAML for
a short pilot run (500 outer steps), checks that training really updates the model, and
looks at what adaptation does on the meta-validation channels.

**Why.** It is a quick, cheap check that the training loop works before the long run in
notebook 06.

**Input.** The raw SMAP release. **Output.** `maml_pilot_seed42.pt`,
`maml_pilot_summary.json` and one figure.

### What the check of the original found

- **The uploaded `04_maml_training.ipynb` (learn2learn version) was correct.** It used
  `learn2learn.MAML(first_order=True)`, whose `clone()` keeps the link to the meta-model.
  A direct test gave gradients on all 20 parameter tensors, and the gradient matched the
  corrected loop of notebook 06 to a relative difference of 1e-7.
- **The Kaggle copy `04-maml-training (3).ipynb` was broken.** It deep-copied the model,
  adapted the copy, and called `.backward()` on the copy. The meta-model received no
  gradient (0 of 20 tensors) and never changed. The checkpoint that notebook 05 loaded
  (`maml_best_kaggle.pt`, step 5,500) came from this broken loop.

### What was corrected

1. The loop comes from `smap_common.py` (plain PyTorch, no learn2learn). The maths is the
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
5. **Inner steps set to 10**, as in notebooks 06–10. The original pilot used 5.
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
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_maml_pilot_curves.png"), dpi=120); plt.close(fig)'''),
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
write("06_smap_maml_corrected.ipynb", [
md(r'''
# 06 — SMAP: MAML-AE against baselines (corrected)

**What this notebook does.** For each training seed it meta-trains MAML, trains a static
LSTM autoencoder and an MLP autoencoder on the same meta-training channels, and scores all
methods on the seven held-out SMAP channels at 1, 5 and 10 shots. It then summarises the
results across training seeds with confidence intervals and paired differences.

**Why.** This is the experiment behind the manuscript's SMAP table. The original had flaws
that made its numbers hard to interpret (listed below).

**Input.** The raw SMAP release and the `corrected_legacy` folder (for `smap_common.py`).
Both are found automatically under `/kaggle/input`.

**Output (in `/kaggle/working`).** One `smap_corrected_seed<seed>.json` per training
seed, `smap_corrected_summary.json`, and model checkpoints.

### How to run on Kaggle

1. Attach two Datasets: the SMAP release, and the `corrected_legacy` folder of this
   repository. Turn on a GPU.
2. Use **Save Version -> Save & Run All (Commit)**.
3. The run is split by training seed. If a session times out, attach this notebook's
   previous output as an extra input and commit again: finished seeds are skipped, and an
   unfinished MAML run resumes from its last checkpoint.
4. Expected time on a Kaggle GPU: about 2.5 hours per training seed (up to 30,000 outer
   steps, based on the original run), so about 12 to 13 hours for five seeds. That exceeds
   one 12-hour session, so plan for one resume. To split the work, set `TRAIN_SEEDS` in
   the configuration cell.

### What was corrected, compared with the original `06_smap_maml_clean.ipynb`

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
                 {"experiment": "SMAP corrected legacy (notebook 06)", "smoke_test": SMOKE,
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
