import pytest

from trade_overfit.demo import demo_true_edge
from trade_overfit.regimes import regime_report, trailing_vol, vol_regime_labels


def _two_regime_series():
    # calm first half, wild second half
    import random
    rng = random.Random(11)
    calm = [rng.gauss(0.0008, 0.005) for _ in range(400)]
    stress = [rng.gauss(-0.0005, 0.03) for _ in range(400)]
    return calm + stress


def test_labels_cover_every_bar():
    rets = demo_true_edge(n=500, seed=7)
    labels = vol_regime_labels(rets)
    assert len(labels) == 500
    assert set(labels) <= {"calm", "stress"}


def test_vol_regime_splits_mixed_series():
    labels = vol_regime_labels(_two_regime_series())
    assert "calm" in labels and "stress" in labels
    # the wild second half should be mostly stress
    assert labels[600:].count("stress") > labels[600:].count("calm")


def test_user_labels_accepted():
    rets = demo_true_edge(n=200, seed=7)
    labels = ["bull"] * 100 + ["bear"] * 100
    rep = regime_report(rets, labels=labels)
    assert rep["label_source"] == "user"
    assert set(rep["regimes"]) == {"bull", "bear"}
    assert rep["worst_regime"] in ("bull", "bear")
    assert rep["worst_regime_sharpe"] is not None


def test_user_labels_length_mismatch_raises():
    with pytest.raises(ValueError):
        regime_report([0.01] * 10, labels=["a"] * 5)


def test_worst_regime_is_minimum_sharpe():
    rep = regime_report(_two_regime_series())
    sharpes = {k: v["sharpe"] for k, v in rep["regimes"].items()}
    assert rep["worst_regime_sharpe"] == min(s for s in sharpes.values() if s is not None)


def test_trailing_vol_warmup():
    vols = trailing_vol([0.01] * 10, lookback=63)
    assert vols[0] is None
    assert vols[-1] is not None
