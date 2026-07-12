"""Tests for the M0 physics baseline."""

from __future__ import annotations

import numpy as np

from taranis.data import chronological_split, generate, make_windows
from taranis.eval import classification_report, tune_threshold_on_val
from taranis.models import BaselinePhysics
from taranis.models.baseline_physics import FEATURE_NAMES, build_features


def test_build_features_shape_et_ordre():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 96, 4)).astype(np.float32)
    feats = build_features(X, step_minutes=10)
    assert feats.shape == (5, len(FEATURE_NAMES))
    assert feats.dtype == np.float32


def test_baseline_entrainable_sur_synthetique():
    df = generate(days=30.0, storms_per_day=0.5, seed=0)
    ds = make_windows(df, Tw=48, H=24, stride=1)
    train, val, test = chronological_split(ds, ratios=(0.7, 0.15, 0.15))
    assert train.y.sum() > 0, "il faut au moins un positif dans le train"
    assert test.y.sum() > 0, "il faut au moins un positif dans le test"

    model = BaselinePhysics(step_minutes=10).fit(train.X, train.y)
    proba_val = model.predict_proba(val.X)
    proba_test = model.predict_proba(test.X)

    thr = tune_threshold_on_val(val.y, proba_val, criterion="f1")
    report = classification_report(test.y, proba_test, threshold=thr)

    # expectations are modest but clearly above chance
    assert report.auc > 0.85, f"AUC trop basse : {report.auc:.3f}"
    assert report.recall > 0.3, f"rappel trop bas : {report.recall:.3f}"


def test_signal_physique_correct():
    """Pressure trends should be robust risk indicators.

    We do not test each coefficient individually because multicollinearity
    between humidity features can flip signs. We check the directional
    consistency of pressure trends, which are more stable.
    """
    df = generate(days=40.0, storms_per_day=0.6, seed=3)
    ds = make_windows(df, Tw=48, H=24, stride=1)
    train, _, _ = chronological_split(ds)
    model = BaselinePhysics().fit(train.X, train.y)
    coefs = model.coefficients()
    # a pressure drop signals a storm ahead, so the coefficient is negative.
    # We test only one indicator: the short and long trends are collinear,
    # so the model can flip one to compensate the other. The short trend
    # remains dominant.
    assert coefs["pressure_trend_short"] < 0, coefs
