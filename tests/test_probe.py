"""Tests for the downstream linear probe."""

from __future__ import annotations

import numpy as np
import torch

from taranis.eval import LinearProbe
from taranis.models import TSJEPA, TSJEPAConfig


def _fake_labelled_dataset(n=512, Tw=32, V=4, seed=0):
    """Small structured dataset: positives have a simple signature."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(0, 1, size=(n, Tw, V)).astype(np.float32)
    # positive signature: pressure trend down + humidity trend up
    for i in np.where(y == 1)[0]:
        x[i, :, 0] += np.linspace(0.5, -0.5, Tw)   # pressure falls
        x[i, :, 2] += np.linspace(-0.3, 0.3, Tw)   # humidity rises
    return x, y


def test_sonde_apprend_signal_facile():
    torch.manual_seed(0)
    cfg = TSJEPAConfig(
        Tw=32,
        n_canaux=4,
        patch_len=4,
        d_model=16,
        n_layers_encoder=2,
        n_layers_predictor=1,
        n_heads=4,
    )
    model = TSJEPA(cfg)
    X_train, y_train = _fake_labelled_dataset(n=512, Tw=32, seed=0)
    X_test, y_test = _fake_labelled_dataset(n=256, Tw=32, seed=1)

    probe = LinearProbe(model).fit(X_train, y_train)
    proba = probe.predict_proba(X_test)

    # without pre-training, the encoder produces essentially random
    # embeddings. The probe cannot work miracles. We only check that it
    # ends up a hair above chance, which is enough to validate that the
    # pipeline (extraction, scaler, regression) works.
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_test, proba)
    assert auc > 0.53, f"la sonde doit dépasser le hasard, AUC={auc:.3f}"


def test_encodeur_reste_gele_apres_fit():
    torch.manual_seed(0)
    cfg = TSJEPAConfig(
        Tw=16,
        n_canaux=4,
        patch_len=4,
        d_model=16,
        n_layers_encoder=1,
        n_layers_predictor=1,
        n_heads=4,
    )
    model = TSJEPA(cfg)
    # save encoder weights before fit
    before = {n: p.clone() for n, p in model.encoder.named_parameters()}

    X, y = _fake_labelled_dataset(n=128, Tw=16)
    LinearProbe(model).fit(X, y)

    # encoder weights must not have moved
    for n, p in model.encoder.named_parameters():
        assert torch.allclose(before[n], p), f"encodeur modifié à {n}"
        assert not p.requires_grad, f"{n} devrait être gelé"
