"""Shared code for the corrected SMAP notebooks (01, 02, 03, 04, 06).

Every corrected notebook imports from this file instead of re-defining the model,
the meta-learning loop or the metrics. The original notebooks each carried their own
copy of this code, which is how a broken training loop (Kaggle notebook
"04-maml-training (3)") went unnoticed.
"""
import ast
import copy
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler

WINDOW = 30
STRIDE_NORMAL = 10     # normal windows from the training file (as in the original 02)
STRIDE_ANOMALY = 5     # legacy anomaly windows from the test file (as in the original 02)

# ---------------------------------------------------------------------------
# Channel split. This is the exact split the original work used
# (task_splits_25feat.json). It is written out here so it no longer depends on the
# order in which cells of the original 02/04 were run.
# ---------------------------------------------------------------------------
META_TRAIN = ["P-3", "A-9", "G-2", "G-7", "E-5", "S-1", "T-3", "G-4", "E-10", "B-1",
              "D-13", "E-1", "A-7", "A-2", "T-1", "G-6", "A-4", "D-9", "D-4", "P-1",
              "P-2", "D-2", "F-1", "E-11", "P-7", "E-9", "E-2", "E-4", "G-3", "F-3",
              "E-13", "A-3", "R-1", "F-2", "D-8", "A-5", "A-1", "D-1", "E-8"]
META_VAL = ["G-1", "E-12", "D-5", "E-7", "A-8", "D-11"]
META_TEST = ["E-3", "D-7", "E-6", "D-12", "D-6", "T-2", "A-6", "D-3", "P-4"]
EXCLUDED = {
    "D-12": "only 312 training timesteps (29 normal windows at stride 10): too few to "
            "draw support sets and hold out normal data. Label-free reason.",
    "P-4": "sensor-dropout (flat-line) anomaly that a reconstruction model cannot flag. "
           "This reason was found by inspecting the test anomalies in the original "
           "notebook 04, so the exclusion is post hoc. P-4 is still scored and reported "
           "separately as a sensitivity row.",
}
EVAL_CHANNELS = [c for c in META_TEST if c not in EXCLUDED]
SENSITIVITY_CHANNELS = ["P-4"]


# ---------------------------------------------------------------------------
# Paths, seeds, provenance
# ---------------------------------------------------------------------------
def find_file(name, extra_roots=()):
    """Search /kaggle/input, then the given roots, for a file called `name`."""
    roots = [r for r in ["/kaggle/input", *extra_roots] if r and os.path.isdir(r)]
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return None


def find_smap_raw(extra_roots=()):
    """Locate labeled_anomalies.csv and the train/ and test/ folders of the SMAP release."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.environ.get("SMAP_ROOT"), *extra_roots,
             os.path.join(here, "..", "SMAP - NASA"), os.getcwd()]
    csv = find_file("labeled_anomalies.csv", roots)
    if csv is None:
        raise FileNotFoundError("labeled_anomalies.csv not found under /kaggle/input or "
                                f"{roots}. Set SMAP_ROOT to the SMAP folder.")
    base = os.path.dirname(csv)
    for dirpath, dirs, _ in os.walk(base):
        if "train" in dirs and "test" in dirs:
            return csv, os.path.join(dirpath, "train"), os.path.join(dirpath, "test")
    raise FileNotFoundError(f"train/ and test/ folders not found below {base}")


def output_dir(smoke=False):
    """/kaggle/working on Kaggle, otherwise a local folder next to this file. Smoke tests
    write to a separate SMOKE subfolder so their files are never mistaken for results."""
    if os.path.isdir("/kaggle/working"):
        d = "/kaggle/working"
    else:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corrected_legacy_out")
    if smoke:
        d = os.path.join(d, "SMOKE")
    os.makedirs(d, exist_ok=True)
    return d


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit():
    """Commit hash of this code. Kaggle has no git, so a CODE_VERSION file is used there."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        p = find_file("CODE_VERSION", [here])
        return open(p).read().strip() if p else "unknown"


def save_json(path, obj):
    obj = dict(obj)
    obj.setdefault("git_commit", git_commit())
    obj.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# SMAP labels and data
# ---------------------------------------------------------------------------
def load_labels(csv_path):
    """One row per channel. The CSV lists P-2 twice (two different anomaly ranges);
    the rows are merged here. The original code kept only the first P-2 row."""
    df = pd.read_csv(csv_path)
    df["seqs"] = df["anomaly_sequences"].apply(ast.literal_eval)   # no eval()
    rows = []
    for ch, g in df.groupby("chan_id", sort=False):
        seqs = [list(map(int, s)) for ss in g["seqs"] for s in ss]   # CSV order kept
        rows.append({"chan_id": ch, "spacecraft": g["spacecraft"].iloc[0],
                     "sequences": seqs, "n_csv_rows": len(g),
                     "num_values": int(g["num_values"].iloc[0])})
    return pd.DataFrame(rows).set_index("chan_id")


def point_labels(sequences, n):
    """Per-timestep labels. Sequence ends in labeled_anomalies.csv are inclusive."""
    y = np.zeros(n, dtype=np.int8)
    for s, e in sequences:
        y[max(0, s):min(n, e + 1)] = 1
    return y


def create_windows(data, window=WINDOW, stride=1):
    n = (len(data) - window) // stride + 1
    if n <= 0:
        return np.zeros((0, window, data.shape[1]), dtype=np.float32)
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return data[idx].astype(np.float32)


def prepare_channel(train_raw, test_raw, sequences, clip=True):
    """MinMax scaling fitted on the training file only, then windows and labels."""
    scaler = MinMaxScaler().fit(train_raw)
    tr = scaler.transform(train_raw)
    te = scaler.transform(test_raw)
    if clip:
        tr, te = np.clip(tr, 0, 1), np.clip(te, 0, 1)
    tr, te = tr.astype(np.float32), te.astype(np.float32)
    y = point_labels(sequences, len(te))
    normal_windows = create_windows(tr, WINDOW, STRIDE_NORMAL)
    normal_starts = np.arange(len(normal_windows)) * STRIDE_NORMAL
    return {"train": tr, "test": te, "labels": y,
            "normal_windows": normal_windows, "normal_starts": normal_starts,
            "legacy_anomaly_windows": legacy_anomaly_windows(te, sequences),
            "scaler_min": scaler.data_min_.astype(np.float32),
            "scaler_max": scaler.data_max_.astype(np.float32)}


def build_smap_dataset(channels=None, extra_roots=()):
    """Load and prepare SMAP channels straight from the raw release (a few seconds).
    Returns ({channel: prepared dict}, labels table). Only SMAP channels (25 features)."""
    csv, train_dir, test_dir = find_smap_raw(extra_roots)
    labels = load_labels(csv)
    if channels is None:
        channels = [c for c in labels.index if labels.loc[c, "spacecraft"] == "SMAP"]
    data = {}
    for ch in channels:
        data[ch] = prepare_channel(np.load(os.path.join(train_dir, f"{ch}.npy")),
                                   np.load(os.path.join(test_dir, f"{ch}.npy")),
                                   labels.loc[ch, "sequences"])
    return data, labels


def legacy_anomaly_windows(test_scaled, sequences):
    """Anomaly windows exactly as the original 02 built them: for each labelled segment,
    stride-5 windows over test[start : end + 30]."""
    out = []
    for s, e in sequences:
        seg = test_scaled[max(0, s):min(e + WINDOW, len(test_scaled))]
        if len(seg) >= WINDOW:
            out.append(create_windows(seg, WINDOW, STRIDE_ANOMALY))
    return np.concatenate(out) if out else np.zeros((0, WINDOW, test_scaled.shape[1]), np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_size=25, hidden1=64, hidden2=32, latent=16):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, latent)

    def forward(self, x):
        out, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(out)       # keep only the last hidden state
        return self.fc(h_n[-1])


class LSTMDecoder(nn.Module):
    def __init__(self, latent=16, hidden1=32, hidden2=64, output_size=25, seq_len=WINDOW):
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(latent, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, output_size)

    def forward(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(z_rep)
        out, _ = self.lstm2(out)
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    """Encoder 64 -> 32 -> code 16; decoder 16 -> 32 -> 64 -> d. 69,481 parameters at d=25."""

    def __init__(self, input_size=25, seq_len=WINDOW, hidden1=64, hidden2=32, latent=16):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden1, hidden2, latent)
        self.decoder = LSTMDecoder(latent, hidden2, hidden1, input_size, seq_len)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=(1, 2))


class MLPAutoencoder(nn.Module):
    def __init__(self, input_size=25, seq_len=WINDOW, latent=16):
        super().__init__()
        self.seq_len, self.input_size = seq_len, input_size
        flat = seq_len * input_size
        self.encoder = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, 64),
                                     nn.ReLU(), nn.Linear(64, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 256),
                                     nn.ReLU(), nn.Linear(256, flat))

    def forward(self, x):
        b = x.shape[0]
        return self.decoder(self.encoder(x.reshape(b, -1))).reshape(b, self.seq_len, self.input_size)

    def reconstruction_error(self, x):
        return torch.mean((x - self.forward(x)) ** 2, dim=(1, 2))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# First-order MAML (FOMAML)
# ---------------------------------------------------------------------------
_mse = nn.MSELoss()


def inner_adapt(model, support, lr=0.01, steps=10):
    """Deep-copy the model and take `steps` SGD steps on the support windows.
    steps=0 returns an unchanged copy (the zero-step control)."""
    learner = copy.deepcopy(model)
    learner.train()
    opt = torch.optim.SGD(learner.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _mse(learner(support), support).backward()
        opt.step()
    return learner


def fomaml_outer_step(model, outer_opt, episodes, inner_lr=0.01, inner_steps=10,
                      clip=1.0, train=True):
    """One outer step. `episodes` is a list of (support, query) tensors.

    The query-loss gradient is taken with respect to the ADAPTED copy's parameters,
    averaged over tasks, and written onto the meta-parameters' .grad before the
    optimizer step. Calling .backward() on the copy instead leaves the meta-model
    without gradients; check_outer_step() catches that."""
    params = list(model.parameters())
    accum = [torch.zeros_like(p) for p in params]
    total = 0.0
    for support, query in episodes:
        learner = inner_adapt(model, support, inner_lr, inner_steps)
        qloss = _mse(learner(query), query)
        if train:
            grads = torch.autograd.grad(qloss, list(learner.parameters()))
            for a, g in zip(accum, grads):
                a.add_(g.detach())
        total += qloss.item()
    if train:
        outer_opt.zero_grad()
        for p, a in zip(params, accum):
            p.grad = a / len(episodes)
        torch.nn.utils.clip_grad_norm_(params, clip)
        outer_opt.step()
    return total / len(episodes)


def check_outer_step(model, before):
    """Runtime checks 1 and 2: every meta-parameter has a non-zero gradient, and the
    meta-parameters changed. `before` is a list of parameter copies taken before the step."""
    grads = [p.grad for p in model.parameters()]
    missing = [i for i, g in enumerate(grads) if g is None or float(g.abs().sum()) == 0.0]
    moved = sum(float((p.detach() - b).norm() ** 2) for p, b in zip(model.parameters(), before)) ** 0.5
    assert not missing, f"meta-parameters without gradient: {missing}"
    assert moved > 0, "meta-parameters did not change after the outer step"
    return {"n_params_with_grad": len(grads) - len(missing), "meta_param_change_l2": moved}


def adaptation_report(model, adapted, support):
    """Check 4: how far adaptation moved the parameters, and whether support loss fell."""
    num = sum(float((a.detach() - p.detach()).norm() ** 2)
              for a, p in zip(adapted.parameters(), model.parameters())) ** 0.5
    den = sum(float(p.detach().norm() ** 2) for p in model.parameters()) ** 0.5
    with torch.no_grad():
        before = _mse(model(support), support).item()
        after = _mse(adapted(support), support).item()
    return {"relative_param_change": num / den, "support_loss_before": before,
            "support_loss_after": after}


def sample_episode(windows, rng, support_size=20, query_size=20):
    idx = rng.permutation(len(windows))
    need = support_size + query_size
    if len(windows) < need:
        s = windows[rng.choice(len(windows), support_size, replace=True)]
        q = windows[rng.choice(len(windows), query_size, replace=True)]
    else:
        s, q = windows[idx[:support_size]], windows[idx[support_size:need]]
    return s, q


def fixed_episodes(windows_by_task, n_per_task, seed, support_size=20, query_size=20):
    """Validation episodes drawn once with their own seed and reused at every check.
    The original drew one fresh random episode per check from the training RNG, which
    made the validation curve noisy and coupled it to the training sample stream."""
    rng = np.random.RandomState(seed)
    return [(t, *sample_episode(w, rng, support_size, query_size))
            for t, w in windows_by_task.items() for _ in range(n_per_task)]


def to_tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def meta_validation_loss(model, val_episodes, device, inner_lr, inner_steps):
    eps = [(to_tensor(s, device), to_tensor(q, device)) for _, s, q in val_episodes]
    return fomaml_outer_step(model, None, eps, inner_lr, inner_steps, train=False)


def train_maml(model, train_windows, val_episodes, device, *, seed, n_outer, val_every,
               inner_lr=0.01, inner_steps=10, outer_lr=1e-3, tasks_per_batch=4,
               support_size=20, query_size=20, use_scheduler=True, ckpt_path=None,
               resume_from=None, log=print):
    """Meta-train with FOMAML. Training episodes come from their own RNG (seeded by
    `seed`); validation uses the fixed `val_episodes`, so validating does not change the
    training sample stream. The best-validation state is returned. Checkpoints hold the
    RNG state too, so a resumed run continues the same sample stream."""
    tasks = list(train_windows)
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=outer_lr)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5,
                                                        min_lr=1e-5) if use_scheduler else None)
    state = {"step": 0, "best_val": float("inf"), "best_step": 0, "history": [],
             "best_state": copy.deepcopy(model.state_dict()), "checks": {}}
    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        rng.set_state(ck["rng"])
        state.update({k: ck[k] for k in ["step", "best_val", "best_step", "history",
                                         "best_state", "checks"]})
        log(f"resumed from {resume_from} at step {state['step']}")
    t0 = time.time()
    first_val = None
    for step in range(state["step"] + 1, n_outer + 1):
        batch = rng.choice(tasks, size=min(tasks_per_batch, len(tasks)), replace=False)
        eps = []
        for t in batch:
            s, q = sample_episode(train_windows[t], rng, support_size, query_size)
            eps.append((to_tensor(s, device), to_tensor(q, device)))
        before = [p.detach().clone() for p in model.parameters()] if step == 1 else None
        model.train()
        tl = fomaml_outer_step(model, opt, eps, inner_lr, inner_steps)
        if step == 1:        # runtime checks 1, 2 and 4
            state["checks"]["outer_step"] = check_outer_step(model, before)
            s, _ = eps[0]
            rep = adaptation_report(model, inner_adapt(model, s, inner_lr, inner_steps), s)
            state["checks"]["adaptation_at_step_1"] = rep
            assert rep["relative_param_change"] > 0, "inner loop did not move the parameters"
            if rep["support_loss_after"] >= rep["support_loss_before"]:
                log("WARNING: adaptation did not lower the support loss at step 1")
        if step % val_every == 0 or step == n_outer:
            vl = meta_validation_loss(model, val_episodes, device, inner_lr, inner_steps)
            if sched is not None:
                sched.step(vl)
            state["history"].append({"step": step, "train_loss": tl, "val_loss": vl,
                                     "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
            first_val = state["history"][0]["val_loss"]
            if vl < state["best_val"]:
                state.update(best_val=vl, best_step=step,
                             best_state=copy.deepcopy(model.state_dict()))
            log(f"step {step:6d} | train {tl:.6f} | val {vl:.6f} | best {state['best_val']:.6f} "
                f"@ {state['best_step']} | {time.time() - t0:.0f}s")
            state["step"] = step
            if ckpt_path:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict() if sched else None,
                            "rng": rng.get_state(), **state}, ckpt_path)
    vals = [h["val_loss"] for h in state["history"]]
    flat = len(vals) > 1 and max(vals) - min(vals) <= 1e-9 * max(1.0, abs(first_val or 0))
    if flat:   # runtime check 3
        log("WARNING: meta-validation loss is flat. The meta-model may not be training.")
    state["checks"]["val_loss_flat"] = bool(flat)
    model.load_state_dict(state["best_state"])
    info = {k: state[k] for k in ["best_val", "best_step", "history", "checks"]}
    info.update(steps_reached=state["step"], max_outer_steps=n_outer)
    return model, info


def train_conventional(model, pool, device, *, seed, max_epochs=150, patience=15, batch=128,
                       lr=1e-3, log=print):
    """Conventional training on pooled normal windows (the original static-AE recipe:
    random 90/10 split, Adam 1e-3, batch 128, early stopping on the 10% split)."""
    seed_everything(seed)
    p = np.random.RandomState(seed).permutation(len(pool))
    cut = int(0.9 * len(pool))
    tr = torch.as_tensor(pool[p[:cut]], dtype=torch.float32)
    va = to_tensor(pool[p[cut:]], device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, bad, best_state, hist = float("inf"), 0, None, []
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        for st in range(0, len(tr), batch):
            b = tr[perm[st:st + batch]].to(device)
            opt.zero_grad(); _mse(model(b), b).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = _mse(model(va), va).item()
        hist.append(vl)
        if vl < best - 1e-6:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log(f"conventional training: best val {best:.6f} after {len(hist)} epochs")
    return model, {"best_val": best, "epochs_run": len(hist), "val_history": hist}


# ---------------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def window_errors(model, windows, device, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(windows), batch):
        out.append(model.reconstruction_error(to_tensor(windows[i:i + batch], device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def pointwise_from_window_errors(errors, n, window=WINDOW):
    """Stride-1 windows w = 0..n-window. A timestep's score is the mean error of all
    windows that cover it (between 1 and `window` windows)."""
    assert len(errors) == n - window + 1
    diff = np.zeros(n + 1)
    np.add.at(diff, np.arange(len(errors)), errors)
    np.add.at(diff, np.arange(len(errors)) + window, -errors)
    sums = np.cumsum(diff)[:n]
    t = np.arange(n)
    cover = np.minimum(t, n - window) - np.maximum(0, t - window + 1) + 1
    return sums / cover


def pointwise_scores(model, series, device, window=WINDOW):
    errs = window_errors(model, create_windows(series, window, 1), device)
    return pointwise_from_window_errors(errs, len(series), window), errs


def label_free_threshold(support_errors):
    """tau = mean + 2 std of the support-window errors when K > 1, 1.20 x mean when K = 1.
    std is the sample standard deviation (n - 1), as torch's .std() in the original code."""
    e = np.asarray(support_errors, dtype=np.float64)
    return float(e.mean() * 1.20) if len(e) == 1 else float(e.mean() + 2.0 * e.std(ddof=1))


def detection_metrics(scores, labels, tau=None):
    """All metrics for one scored series. F1 at tau is the deployable number; best_f1 uses
    the labels to pick the threshold and is therefore labelled 'oracle'.
    Unlike the original compute_metrics, an all-positive prediction is scored as it is
    (the original set F1 to 0, which hid how much F1 depends on the anomaly fraction)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    out = {"n": int(len(labels)), "prevalence": float(labels.mean()),
           "trivial_all_positive_f1": float(2 * labels.mean() / (1 + labels.mean()))}
    both = len(np.unique(labels)) > 1
    out["roc_auc"] = float(roc_auc_score(labels, scores)) if both else None
    out["pr_auc"] = float(average_precision_score(labels, scores)) if both else None
    if both:
        p, r, _ = precision_recall_curve(labels, scores)
        f = 2 * p * r / np.maximum(p + r, 1e-12)
        i = int(np.nanargmax(f))
        out.update(oracle_best_f1=float(f[i]), oracle_best_f1_precision=float(p[i]),
                   oracle_best_f1_recall=float(r[i]))
        out["separation_ratio"] = float(scores[labels == 1].mean() / max(scores[labels == 0].mean(), 1e-12))
    if tau is not None:
        pred = (scores > tau).astype(int)
        out.update(tau=float(tau), flagged_fraction=float(pred.mean()),
                   f1_at_tau=float(f1_score(labels, pred, zero_division=0)),
                   precision_at_tau=float(precision_score(labels, pred, zero_division=0)),
                   recall_at_tau=float(recall_score(labels, pred, zero_division=0)))
    return out


# ---------------------------------------------------------------------------
# Legacy (window-level) query, kept only as a bridge to the old numbers
# ---------------------------------------------------------------------------
def legacy_eval_query(normal_windows, anomaly_windows, k_shot, support_seed, max_normal_query=100):
    """Exactly the original build_eval_query: K normal support windows from the TRAINING
    file; query = ALL anomaly windows from the TEST file + up to 100 other normal windows
    from the training file. Also returns the support/query window indices so overlap
    between support and query windows can be measured."""
    rng = np.random.RandomState(support_seed)
    n_idx = np.arange(len(normal_windows))
    a_idx = np.arange(len(anomaly_windows))
    rng.shuffle(n_idx)
    rng.shuffle(a_idx)
    sup_i = n_idx[:k_shot]
    rem_i = n_idx[k_shot:k_shot + max_normal_query]
    q = np.concatenate([anomaly_windows[a_idx], normal_windows[rem_i]], 0)
    y = np.concatenate([np.ones(len(a_idx)), np.zeros(len(rem_i))])
    perm = rng.permutation(len(q))
    return normal_windows[sup_i], q[perm], y[perm], sup_i, rem_i


def overlapping_fraction(sup_idx, query_idx, starts, window=WINDOW):
    """Fraction of normal query windows that share at least one timestep with a support window."""
    if len(query_idx) == 0 or len(sup_idx) == 0:
        return 0.0
    s = starts[np.asarray(sup_idx)][None, :]
    q = starts[np.asarray(query_idx)][:, None]
    return float((np.abs(q - s) < window).any(axis=1).mean())
