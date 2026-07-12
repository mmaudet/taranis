"""Regression test guarding against representation collapse.

Run a mini training (200 steps, compact model) on structured random data
and check that after training:

- the mean embedding std remains above a threshold,
- the effective covariance rank remains above a threshold.

These two guardrails are the only reliable ones to detect collapse.
"""

from __future__ import annotations

import numpy as np
import torch

from taranis.models import TSJEPAConfig
from taranis.train import TrainerConfig, TSJEPATrainer


def _fake_dataset(n=1024, Tw=32, V=4, seed=0):
    rng = np.random.default_rng(seed)
    # structured signal: slow sinusoid + trend + noise
    t = np.arange(Tw)[None, :, None]
    base = np.sin(2 * np.pi * t / Tw) + 0.5 * np.cos(2 * np.pi * t / (Tw / 3))
    x = np.tile(base, (n, 1, V)).astype(np.float32)
    # per-sample perturbations
    x += rng.normal(0.0, 0.3, size=x.shape).astype(np.float32)
    return x


def test_pas_de_collapse_apres_petit_entrainement():
    torch.manual_seed(0)
    X_train = _fake_dataset(n=1024, Tw=32, V=4, seed=0)
    X_val = _fake_dataset(n=256, Tw=32, V=4, seed=1)

    mcfg = TSJEPAConfig(
        Tw=32,
        n_canaux=4,
        patch_len=4,
        d_model=32,
        n_layers_encoder=2,
        n_layers_predictor=1,
        n_heads=4,
    )
    tcfg = TrainerConfig(
        n_steps=200,
        batch_size=64,
        val_every=100,
        val_batches=2,
        log_every=100,
        warmup_steps=20,
        n_blocks=2,
        block_size=2,
        seed=0,
    )
    trainer = TSJEPATrainer(mcfg, tcfg, X_train, X_val)
    state = trainer.fit(verbose=False)

    last = state.val_stats[-1]
    # embeddings must keep a clear dispersion
    assert last["z_ctx_std"] > 0.1, f"z_ctx_std trop bas : {last['z_ctx_std']}"
    assert last["z_tgt_std"] > 0.1, f"z_tgt_std trop bas : {last['z_tgt_std']}"
    # effective rank >= 2, well above 1 (full collapse)
    assert last["z_ctx_rank"] >= 2.0, f"z_ctx_rank trop bas : {last['z_ctx_rank']}"
    assert last["z_tgt_rank"] >= 2.0, f"z_tgt_rank trop bas : {last['z_tgt_rank']}"


def test_perte_descend_dans_les_100_premiers_pas():
    torch.manual_seed(0)
    X_train = _fake_dataset(n=1024, Tw=32, V=4, seed=0)
    X_val = _fake_dataset(n=256, Tw=32, V=4, seed=1)

    mcfg = TSJEPAConfig(Tw=32, n_canaux=4, patch_len=4, d_model=32,
                       n_layers_encoder=2, n_layers_predictor=1, n_heads=4)
    tcfg = TrainerConfig(n_steps=100, batch_size=64, val_every=100,
                        log_every=25, warmup_steps=10, n_blocks=2,
                        block_size=2, seed=0)
    trainer = TSJEPATrainer(mcfg, tcfg, X_train, X_val)
    state = trainer.fit(verbose=False)

    losses = [entry["loss"] for entry in state.train_losses]
    assert losses[-1] < losses[0], (
        f"la perte doit descendre : {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
