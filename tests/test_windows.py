"""Tests du fenêtrage et du split chronologique."""

from __future__ import annotations

import numpy as np

from taranis.data import (
    channel_stats,
    chronological_split,
    generate,
    make_windows,
    normalize,
)


def test_formes_et_types():
    df = generate(days=5.0, seed=0)
    ds = make_windows(df, Tw=48, H=24, stride=1)
    assert ds.X.dtype == np.float32
    assert ds.y.dtype == np.int64
    assert ds.X.shape[1:] == (48, 4)
    assert len(ds) == ds.X.shape[0] == ds.y.shape[0]


def test_labels_positifs_precedent_les_onsets():
    """Si un onset est à l'indice t*, les fenêtres qui finissent en [t*-H, t*-1] doivent être positives."""
    df = generate(days=15.0, storms_per_day=1.0, seed=5)
    Tw, H = 48, 24
    ds = make_windows(df, Tw=Tw, H=H, stride=1)

    onsets = np.flatnonzero(df["storm_onset"].to_numpy())
    assert len(onsets) > 0

    # pour chaque onset, on identifie les indices de fin de fenêtre qui devraient être positifs
    for onset in onsets:
        # une fenêtre finissant à `t` a un horizon [t+1, t+H]
        # on veut donc t in [onset - H, onset - 1]
        candidats_end = range(max(Tw - 1, onset - H), onset)
        for end_idx in candidats_end:
            i = end_idx - (Tw - 1)  # position dans le tableau ds
            if 0 <= i < len(ds):
                assert ds.y[i] == 1, (
                    f"fenêtre finissant en {end_idx} devrait être positive "
                    f"pour l'onset {onset}"
                )


def test_split_chronologique_disjoint_et_ordonne():
    df = generate(days=5.0, seed=1)
    ds = make_windows(df, Tw=48, H=24, stride=1)
    train, val, test = chronological_split(ds, ratios=(0.7, 0.15, 0.15))
    assert len(train) + len(val) + len(test) == len(ds)
    # ordre strict
    assert train.timestamps[-1] < val.timestamps[0]
    assert val.timestamps[-1] < test.timestamps[0]


def test_normalisation_bien_calee_sur_le_train():
    df = generate(days=10.0, seed=2)
    ds = make_windows(df, Tw=48, H=24, stride=4)
    train, val, test = chronological_split(ds)
    mean, std = channel_stats(train.X)

    Xn = normalize(train.X, mean, std)
    # moyennes et stds cibles à peu près 0 et 1 sur le train
    # tolérance légère : float32 + variance intra-fenêtre
    assert np.allclose(Xn.reshape(-1, 4).mean(axis=0), 0, atol=5e-3)
    assert np.allclose(Xn.reshape(-1, 4).std(axis=0), 1, atol=1e-2)

    # val et test conservent des ordres de grandeur raisonnables (pas de fuite d'échelle)
    Xn_val = normalize(val.X, mean, std)
    assert np.abs(Xn_val).max() < 20, "val doit rester dans une plage raisonnable"


def test_dataset_trop_court_leve_erreur():
    df = generate(days=0.1, step_minutes=10, seed=0)
    try:
        make_windows(df, Tw=96, H=48)
    except ValueError as e:
        assert "trop courte" in str(e)
    else:
        raise AssertionError("attendait un ValueError")
