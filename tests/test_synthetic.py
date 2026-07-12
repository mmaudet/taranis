"""Tests du générateur météo synthétique."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taranis.data.synthetic import generate


def test_schema_et_pas_de_temps_regulier():
    df = generate(days=2.0, step_minutes=10, seed=0)
    attendues = {
        "timestamp",
        "pressure",
        "temp",
        "humidity",
        "wind",
        "storm_active",
        "storm_onset",
    }
    assert set(df.columns) == attendues
    # pas de temps régulier
    diffs = df["timestamp"].diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(minutes=10)


def test_reproductibilite_avec_meme_seed():
    a = generate(days=1.0, seed=42)
    b = generate(days=1.0, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_seeds_differents_donnent_signaux_differents():
    a = generate(days=1.0, seed=1)
    b = generate(days=1.0, seed=2)
    assert not np.allclose(a["pressure"].values, b["pressure"].values)


def test_plages_physiques_plausibles():
    df = generate(days=5.0, storms_per_day=0.5, seed=7)
    # bornes larges mais quand même physiques
    assert df["pressure"].between(970, 1050).all()
    assert df["humidity"].between(0, 100).all()
    assert (df["wind"] >= 0).all()
    assert df["temp"].between(-20, 45).all()


def test_labels_coherents():
    df = generate(days=10.0, storms_per_day=1.0, seed=3)
    # chaque onset doit être compris dans une fenêtre active
    onsets = df.index[df["storm_onset"]].tolist()
    assert onsets, "attendu au moins un onset avec ce taux"
    for idx in onsets:
        assert df.loc[idx, "storm_active"], "l'onset doit tomber dans un intervalle actif"


def test_signature_orage_visible():
    """Un orage doit se traduire par une chute de pression et une hausse d'humidité."""
    df = generate(days=15.0, storms_per_day=1.0, seed=11)
    onsets = df.index[df["storm_onset"]].tolist()
    assert onsets

    # on prend le premier onset qui a assez de contexte avant et après
    step = 10  # minutes, cohérent avec le défaut du générateur
    n_before = 60 // step  # 1h avant
    n_after = 30 // step   # 30 min après

    verifications = 0
    for idx in onsets:
        if idx < n_before or idx + n_after >= len(df):
            continue
        p_before = df["pressure"].iloc[idx - n_before]
        p_after = df["pressure"].iloc[idx + n_after]
        h_before = df["humidity"].iloc[idx - n_before]
        h_after = df["humidity"].iloc[idx + n_after]
        # pression baisse et humidité monte, sans exiger un seuil dur
        if p_after < p_before and h_after > h_before:
            verifications += 1

    assert verifications >= 1, "au moins un orage doit montrer la signature attendue"
