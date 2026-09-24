"""Unit tests for smap_common. Run with:  python -m pytest corrected_legacy/tests -q"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import smap_common as sc  # noqa: E402


def _episodes(n=4, d=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.rand(20, 30, d, generator=g), torch.rand(20, 30, d, generator=g)) for _ in range(n)]


def test_lstm_ae_parameter_count():
    assert sc.count_params(sc.LSTMAutoencoder(25)) == 69481


def test_mlp_ae_parameter_count_matches_old_profile():
    assert sc.count_params(sc.MLPAutoencoder(25)) == 420222


def test_state_dict_keys_match_original_checkpoints():
    keys = set(sc.LSTMAutoencoder(25).state_dict())
    assert "encoder.lstm1.weight_ih_l0" in keys and "decoder.fc.bias" in keys


def test_fomaml_step_gives_gradients_and_moves_parameters():
    torch.manual_seed(0)
    m = sc.LSTMAutoencoder(25)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in m.parameters()]
    sc.fomaml_outer_step(m, opt, _episodes(), inner_steps=3)
    info = sc.check_outer_step(m, before)
    assert info["n_params_with_grad"] == len(before)
    assert info["meta_param_change_l2"] > 0


def _broken_outer_step(model, opt, episodes, inner_lr=0.01, inner_steps=3):
    """The loop from the Kaggle notebook '04-maml-training (3)': backward() is called on
    the adapted deep copy, so the meta-model never receives a gradient."""
    meta = torch.tensor(0.0)
    for s, q in episodes:
        learner = sc.inner_adapt(model, s, inner_lr, inner_steps)
        meta = meta + sc._mse(learner(q), q)
    meta = meta / len(episodes)
    opt.zero_grad()
    meta.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def test_broken_loop_is_detected():
    torch.manual_seed(0)
    m = sc.LSTMAutoencoder(25)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in m.parameters()]
    _broken_outer_step(m, opt, _episodes())
    with pytest.raises(AssertionError):
        sc.check_outer_step(m, before)


def test_zero_step_adaptation_is_identity_and_adaptation_lowers_support_loss():
    torch.manual_seed(0)
    m = sc.LSTMAutoencoder(25)
    s, _ = _episodes(1)[0]
    same = sc.inner_adapt(m, s, steps=0)
    assert all(torch.equal(a, b) for a, b in zip(same.parameters(), m.parameters()))
    rep = sc.adaptation_report(m, sc.inner_adapt(m, s, lr=0.05, steps=20), s)
    assert rep["relative_param_change"] > 0
    assert rep["support_loss_after"] < rep["support_loss_before"]


def test_pointwise_mapping_is_mean_of_covering_windows():
    rng = np.random.RandomState(0)
    n, L = 50, 7
    errs = rng.rand(n - L + 1)
    got = sc.pointwise_from_window_errors(errs, n, L)
    ref = np.array([errs[max(0, t - L + 1):min(t, n - L) + 1].mean() for t in range(n)])
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_point_labels_are_inclusive():
    y = sc.point_labels([[2, 4]], 8)
    assert y.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


def test_label_free_threshold():
    assert sc.label_free_threshold([2.0]) == pytest.approx(2.4)
    e = np.array([1.0, 2.0, 3.0])
    assert sc.label_free_threshold(e) == pytest.approx(e.mean() + 2 * e.std(ddof=1))
    assert sc.label_free_threshold(e) == pytest.approx(float(torch.tensor(e).mean() + 2 * torch.tensor(e).std()))


def test_all_positive_prediction_is_scored_not_zeroed():
    y = np.array([1, 1, 1, 0])
    m = sc.detection_metrics(np.array([5.0, 5, 5, 5]), y, tau=0.0)
    assert m["f1_at_tau"] == pytest.approx(2 * 0.75 / 1.75)
    assert m["trivial_all_positive_f1"] == pytest.approx(2 * 0.75 / 1.75)


def test_overlap_fraction():
    starts = np.arange(10) * 10
    assert sc.overlapping_fraction([3], [2, 4, 7], starts) == pytest.approx(2 / 3)


def test_fixed_episodes_are_reproducible():
    w = {"a": np.random.RandomState(1).rand(60, 30, 2).astype(np.float32)}
    e1 = sc.fixed_episodes(w, 3, seed=7)
    e2 = sc.fixed_episodes(w, 3, seed=7)
    assert all(np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) for a, b in zip(e1, e2))


def test_split_is_disjoint_and_has_54_channels():
    a, b, c = set(sc.META_TRAIN), set(sc.META_VAL), set(sc.META_TEST)
    assert not (a & b or a & c or b & c)
    assert len(a | b | c) == 54
    assert sc.EVAL_CHANNELS == ["E-3", "D-7", "E-6", "D-6", "T-2", "A-6", "D-3"]


def _toy_tasks(seed=0, d=3):
    r = np.random.RandomState(seed)
    return {f"t{i}": r.rand(50, 30, d).astype(np.float32) for i in range(5)}


def test_train_maml_runs_checks_and_resume_matches_uninterrupted(tmp_path):
    tasks = _toy_tasks()
    val = sc.fixed_episodes({"v": _toy_tasks(1)["t0"]}, 2, seed=3)
    kw = dict(seed=5, val_every=2, inner_steps=2, support_size=5, query_size=5,
              tasks_per_batch=2, log=lambda *_: None)
    torch.manual_seed(0); a = sc.LSTMAutoencoder(3)
    torch.manual_seed(0); b = sc.LSTMAutoencoder(3)
    a, info = sc.train_maml(a, tasks, val, "cpu", n_outer=4, **kw)
    assert info["checks"]["outer_step"]["meta_param_change_l2"] > 0
    ck = str(tmp_path / "ck.pt")
    sc.train_maml(b, tasks, val, "cpu", n_outer=2, ckpt_path=ck, **kw)
    torch.manual_seed(0); c = sc.LSTMAutoencoder(3)
    c, info_c = sc.train_maml(c, tasks, val, "cpu", n_outer=4, resume_from=ck, **kw)
    assert info_c["steps_reached"] == 4
    assert all(torch.allclose(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))


def test_train_conventional_lowers_loss():
    pool = _toy_tasks()["t0"]
    torch.manual_seed(0)
    m, info = sc.train_conventional(sc.LSTMAutoencoder(3), pool, "cpu", seed=1, max_epochs=5,
                                    patience=5, log=lambda *_: None)
    assert info["val_history"][-1] <= info["val_history"][0]
