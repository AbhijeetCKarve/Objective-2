"""Audit of the ORIGINAL SMAP evaluation in 06_smap_maml_clean.ipynb.

What it does, on CPU, with the real SMAP files and the original saved checkpoints:
  1. Rebuilds the old window data and re-runs the old evaluation code verbatim with the
     old checkpoints (smap_maml_best.pt, smap_static_ae.pt). If the data code is right,
     this reproduces the manuscript's Version B table.
  2. Measures properties of the old query sets: anomaly fraction, the F1 a trivial
     "flag everything" rule would get, and how many normal query windows share
     timesteps with the support windows.
  3. Scores the same channels with an UNTRAINED random LSTM-AE and with a model-free
     score, under the old window protocol and under point-wise scoring of the test file.

Usage:  python corrected_legacy/audit_old_smap_protocol.py <folder with old .pt files>
Writes: corrected_legacy/audit/audit_old_smap_protocol.json
"""
import copy
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smap_common as sc  # noqa: E402

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/user/abhijeetckarve/maml_ae"
DEVICE = torch.device("cpu")
K_SHOTS, SEEDS = [1, 5, 10], [42, 123, 456, 789, 1024]
CHANNELS = sc.EVAL_CHANNELS + sc.SENSITIVITY_CHANNELS
torch.set_num_threads(4)

# --------------------------------------------------------------- data
csv, train_dir, test_dir = sc.find_smap_raw()
labels = sc.load_labels(csv)
data = {}
for ch in CHANNELS:
    d = sc.prepare_channel(np.load(f"{train_dir}/{ch}.npy"), np.load(f"{test_dir}/{ch}.npy"),
                           labels.loc[ch, "sequences"])
    data[ch] = d

# --------------------------------------------------------------- old code, verbatim
crit = torch.nn.MSELoss()


def adapt_model(model, support, n_steps=5, lr=0.01):
    learner = copy.deepcopy(model); learner.train()
    o = torch.optim.SGD(learner.parameters(), lr=lr)
    for _ in range(n_steps):
        o.zero_grad(); l = crit(learner(support), support); l.backward(); o.step()
    return learner


def compute_threshold(model, support, multiplier=2.0):
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(support)
    m = errors.mean().item()
    return m + multiplier * errors.std().item() if len(errors) > 1 else m * 1.20


def compute_metrics(scores, labels_, tau):
    preds = (scores > tau).astype(int)
    if 0 < preds.sum() < len(preds):
        f1 = f1_score(labels_, preds, zero_division=0)
    else:
        f1 = 0.0
    au = roc_auc_score(labels_, scores) if len(np.unique(labels_)) > 1 else 0.5
    return {"f1": round(float(f1), 4), "auroc": round(float(au), 4)}


def load(name):
    m = sc.LSTMAutoencoder(25)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, name), map_location="cpu",
                                 weights_only=False)["model_state_dict"])
    return m


maml_model, static_model = load("smap_maml_best.pt"), load("smap_static_ae.pt")


def evaluate_channel(ch, k, seed):
    d = data[ch]
    sup, q, y, _, _ = sc.legacy_eval_query(d["normal_windows"], d["legacy_anomaly_windows"], k, seed)
    sup, q = sc.to_tensor(sup, DEVICE), sc.to_tensor(q, DEVICE)
    out = {}
    for name, base, steps in [("MAML-AE", maml_model, 5), ("Static-AE", static_model, 5)]:
        a = adapt_model(base, sup, steps, 0.01); a.eval()
        with torch.no_grad():
            s = a.reconstruction_error(q).numpy()
        out[name] = compute_metrics(s, y, compute_threshold(a, sup))
    a = adapt_model(sc.MLPAutoencoder(25), sup, 50, 0.01); a.eval()
    with torch.no_grad():
        s = a.reconstruction_error(q).numpy()
    out["MLP-AE"] = compute_metrics(s, y, compute_threshold(a, sup))
    sf, qf = sup.numpy().reshape(len(sup), -1), q.numpy().reshape(len(q), -1)
    iso = IsolationForest(n_estimators=100, contamination="auto", random_state=seed).fit(sf)
    isc = -iso.score_samples(qf); ip = (iso.predict(qf) == -1).astype(int)
    f1 = f1_score(y, ip, zero_division=0) if 0 < ip.sum() < len(ip) else 0.0
    out["Isolation-Forest"] = {"f1": round(float(f1), 4),
                               "auroc": round(float(roc_auc_score(y, isc)), 4)}
    return out


print("1) Re-running the old evaluation with the old checkpoints ...")
repro = {}
for ch in sc.EVAL_CHANNELS:
    repro[ch] = {}
    for k in K_SHOTS:
        runs = []
        for seed in SEEDS:
            np.random.seed(seed); torch.manual_seed(seed)
            runs.append(evaluate_channel(ch, k, seed))
        repro[ch][k] = {m: {mt: float(np.mean([r[m][mt] for r in runs])) for mt in ["f1", "auroc"]}
                        for m in runs[0]}
macro = {k: {m: {mt: float(np.mean([repro[c][k][m][mt] for c in sc.EVAL_CHANNELS]))
                 for mt in ["auroc", "f1"]} for m in repro[sc.EVAL_CHANNELS[0]][k]} for k in K_SHOTS}
VERSION_B = {1: {"MAML-AE": (0.487, 0.316), "Static-AE": (0.450, 0.279), "MLP-AE": (0.586, 0.391), "Isolation-Forest": (0.500, 0.000)},
             5: {"MAML-AE": (0.475, 0.152), "Static-AE": (0.453, 0.134), "MLP-AE": (0.586, 0.172), "Isolation-Forest": (0.590, 0.161)},
             10: {"MAML-AE": (0.480, 0.107), "Static-AE": (0.457, 0.102), "MLP-AE": (0.589, 0.145), "Isolation-Forest": (0.607, 0.265)}}
for k in K_SHOTS:
    for m, (au, f1) in VERSION_B[k].items():
        r = macro[k][m]
        print(f"  {k:2d}-shot {m:17s} AUROC {r['auroc']:.3f} (Version B {au:.3f}) | "
              f"F1 {r['f1']:.3f} (Version B {f1:.3f})")

print("\n2) Properties of the old query sets ...")
props = {}
for ch in CHANNELS:
    d = data[ch]
    props[ch] = {"n_normal_windows_train_file": len(d["normal_windows"]),
                 "n_legacy_anomaly_windows": len(d["legacy_anomaly_windows"]),
                 "test_point_prevalence": float(d["labels"].mean())}
    for k in K_SHOTS:
        prev, ov = [], []
        for seed in SEEDS:
            _, _, y, si, qi = sc.legacy_eval_query(d["normal_windows"], d["legacy_anomaly_windows"], k, seed)
            prev.append(y.mean()); ov.append(sc.overlapping_fraction(si, qi, d["normal_starts"]))
        p = float(np.mean(prev))
        props[ch][f"k{k}"] = {"query_anomaly_fraction": p, "trivial_flag_all_f1": 2 * p / (1 + p),
                              "normal_query_windows_overlapping_support": float(np.mean(ov))}
    print(f"  {ch}: normal windows {props[ch]['n_normal_windows_train_file']:3d}, anomaly windows "
          f"{props[ch]['n_legacy_anomaly_windows']:3d}, query anomaly fraction "
          f"{props[ch]['k10']['query_anomaly_fraction']:.2f}, flag-all F1 "
          f"{props[ch]['k10']['trivial_flag_all_f1']:.2f}, overlap (K=10) "
          f"{props[ch]['k10']['normal_query_windows_overlapping_support']:.2f}")

print("\n3) Untrained random LSTM-AE and a model-free score ...")
torch.manual_seed(42)
random_model = sc.LSTMAutoencoder(25)
confound = {}
for ch in CHANNELS:
    d = data[ch]
    _, q, y, _, _ = sc.legacy_eval_query(d["normal_windows"], d["legacy_anomaly_windows"], 10, 42)
    legacy_rand = roc_auc_score(y, sc.window_errors(random_model, q, DEVICE))
    legacy_mean = roc_auc_score(y, q.mean(axis=(1, 2)))
    pw, _ = sc.pointwise_scores(random_model, d["test"], DEVICE)
    point_rand = roc_auc_score(d["labels"], pw)
    confound[ch] = {"legacy_window_auroc_random_untrained_lstm_ae": float(legacy_rand),
                    "legacy_window_auroc_window_mean_value": float(legacy_mean),
                    "pointwise_auroc_random_untrained_lstm_ae": float(point_rand)}
    print(f"  {ch}: old window protocol, random model {legacy_rand:.3f}, window mean value "
          f"{legacy_mean:.3f} | point-wise on test file, random model {point_rand:.3f}")
for key in ["legacy_window_auroc_random_untrained_lstm_ae", "legacy_window_auroc_window_mean_value",
            "pointwise_auroc_random_untrained_lstm_ae"]:
    print(f"  macro over 7 eval channels, {key}: "
          f"{np.mean([confound[c][key] for c in sc.EVAL_CHANNELS]):.3f}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit")
os.makedirs(out_dir, exist_ok=True)
sc.save_json(os.path.join(out_dir, "audit_old_smap_protocol.json"),
             {"description": "Audit of the original SMAP evaluation (06_smap_maml_clean). CPU run.",
              "checkpoints": {"maml": "smap_maml_best.pt (step 18000)", "static": "smap_static_ae.pt"},
              "reproduction_macro": macro, "reproduction_per_channel": repro,
              "manuscript_version_b": {str(k): v for k, v in VERSION_B.items()},
              "query_set_properties": props, "random_model_and_model_free": confound,
              "torch": torch.__version__, "device": "cpu"})
print("\nsaved", os.path.join(out_dir, "audit_old_smap_protocol.json"))
