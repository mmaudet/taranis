"""Boucle d'entraînement TS-JEPA, avec surveillance du collapse.

Le trainer suit un motif simple, lisible étape par étape :

1. À chaque batch, on tire un masque par blocs (partagé sur le batch).
2. On calcule la perte SmoothL1 en espace latent, on rétropropage.
3. On clippe les gradients pour la stabilité.
4. On met à jour l'encodeur cible par EMA, avec un `tau` qui monte
   progressivement de sa valeur initiale vers 0.999 (planning cosine).
5. Le taux d'apprentissage suit un **warmup** linéaire, puis une
   **décroissance cosine** vers zéro.
6. À intervalle régulier, on évalue sur un batch de validation et on mesure :
    - la perte de validation,
    - l'écart-type moyen des embeddings,
    - le rang effectif de leur covariance.

Ces trois indicateurs sont les seuls garde-fous fiables contre le collapse.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from taranis.models import (
    TSJEPA,
    TSJEPAConfig,
    embedding_stats,
    jepa_loss,
    sample_block_mask,
)

# --------------------------------------------------------------------------- #
# Utilitaires de planning
# --------------------------------------------------------------------------- #


def cosine_schedule(step: int, total: int, v_start: float, v_end: float) -> float:
    """Décroissance / croissance cosine entre `v_start` et `v_end`."""
    if total <= 0:
        return v_end
    t = min(step, total) / total
    return v_end + 0.5 * (v_start - v_end) * (1 + math.cos(math.pi * t))


def warmup_cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float = 0.0) -> float:
    """LR linéairement de 0 à `lr_max` sur `warmup` pas, puis cosine vers `lr_min`."""
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    return cosine_schedule(step - warmup, max(1, total - warmup), lr_max, lr_min)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class TrainerConfig:
    # boucle
    n_steps: int = 2000
    batch_size: int = 128
    val_every: int = 100
    val_batches: int = 4
    log_every: int = 50

    # optimisation
    lr_max: float = 1e-3
    lr_min: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # EMA de la cible
    tau_start: float = 0.996
    tau_end: float = 0.999

    # masquage
    n_blocks: int = 2
    block_size: int = 3

    # divers
    seed: int = 0
    device: str = "cpu"
    log_dir: str | None = None  # écriture JSON si fourni


@dataclass
class TrainerState:
    step: int = 0
    train_losses: list[dict] = field(default_factory=list)
    val_stats: list[dict] = field(default_factory=list)
    wall_time_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class TSJEPATrainer:
    """Entraîneur TS-JEPA sur un tenseur `(N, Tw, V)` fourni en mémoire.

    Interface d'usage typique :

        trainer = TSJEPATrainer(model_config, train_config, X_train, X_val)
        state = trainer.fit()
        trainer.save('runs/tsjepa_synth')
    """

    def __init__(
        self,
        model_config: TSJEPAConfig,
        trainer_config: TrainerConfig,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ):
        self.mcfg = model_config
        self.tcfg = trainer_config
        torch.manual_seed(trainer_config.seed)
        np.random.seed(trainer_config.seed)

        self.device = torch.device(trainer_config.device)
        self.model = TSJEPA(model_config).to(self.device)

        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_config.lr_max,
            weight_decay=trainer_config.weight_decay,
        )

        self.train_loader = self._make_loader(X_train, shuffle=True)
        self.val_loader = self._make_loader(X_val, shuffle=False)
        # itérateur infini sur le val, pour piocher `val_batches` à chaque évaluation
        self._val_iter = None

        self.state = TrainerState()

    # ---- data loading ---- #

    def _make_loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        tensor = torch.from_numpy(X).float()
        ds = TensorDataset(tensor)
        return DataLoader(
            ds,
            batch_size=self.tcfg.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0,
        )

    def _next_val_batch(self) -> torch.Tensor:
        while True:
            if self._val_iter is None:
                self._val_iter = iter(self.val_loader)
            try:
                (x,) = next(self._val_iter)
                return x.to(self.device)
            except StopIteration:
                self._val_iter = None

    # ---- schedules ---- #

    def _current_lr(self) -> float:
        return warmup_cosine_lr(
            self.state.step,
            self.tcfg.warmup_steps,
            self.tcfg.n_steps,
            self.tcfg.lr_max,
            self.tcfg.lr_min,
        )

    def _current_tau(self) -> float:
        return cosine_schedule(
            self.state.step,
            self.tcfg.n_steps,
            self.tcfg.tau_start,
            self.tcfg.tau_end,
        )

    def _apply_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    # ---- steps ---- #

    def train_step(self, x: torch.Tensor) -> float:
        self.model.train()
        # masque par batch, partagé
        ctx, tgt = sample_block_mask(
            self.mcfg.n_patches, self.tcfg.n_blocks, self.tcfg.block_size
        )
        ctx = ctx.to(self.device)
        tgt = tgt.to(self.device)

        lr = self._current_lr()
        self._apply_lr(lr)

        pred, z_tgt = self.model(x, ctx, tgt)
        loss = jepa_loss(pred, z_tgt)

        self.opt.zero_grad()
        loss.backward()
        # clip pour éviter les explosions au démarrage
        nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
        self.opt.step()

        tau = self._current_tau()
        self.model.update_target(tau=tau)

        return loss.item()

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        losses = []
        stats_ctx = []
        stats_tgt = []

        for _ in range(self.tcfg.val_batches):
            x = self._next_val_batch()
            ctx, tgt = sample_block_mask(
                self.mcfg.n_patches, self.tcfg.n_blocks, self.tcfg.block_size
            )
            ctx = ctx.to(self.device)
            tgt = tgt.to(self.device)

            pred, z_tgt = self.model(x, ctx, tgt)
            losses.append(jepa_loss(pred, z_tgt).item())

            z_ctx = self.model.encode_context(x, ctx)
            stats_ctx.append(embedding_stats(z_ctx))
            stats_tgt.append(embedding_stats(z_tgt))

        def _mean(key, lst):
            return float(np.mean([s[key] for s in lst]))

        return {
            "val_loss": float(np.mean(losses)),
            "z_ctx_std": _mean("std_moy", stats_ctx),
            "z_ctx_rank": _mean("eff_rank", stats_ctx),
            "z_tgt_std": _mean("std_moy", stats_tgt),
            "z_tgt_rank": _mean("eff_rank", stats_tgt),
        }

    # ---- fit ---- #

    def fit(self, verbose: bool = True) -> TrainerState:
        start = time.time()
        recent = []
        train_iter = iter(self.train_loader)

        while self.state.step < self.tcfg.n_steps:
            try:
                (x,) = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                (x,) = next(train_iter)
            x = x.to(self.device)

            loss = self.train_step(x)
            recent.append(loss)
            self.state.step += 1

            if self.state.step % self.tcfg.log_every == 0:
                avg = float(np.mean(recent[-self.tcfg.log_every :]))
                self.state.train_losses.append(
                    {
                        "step": self.state.step,
                        "loss": avg,
                        "lr": self._current_lr(),
                        "tau": self._current_tau(),
                    }
                )
                if verbose:
                    print(
                        f"[step {self.state.step:4d}/{self.tcfg.n_steps}]  "
                        f"loss={avg:.5f}  lr={self._current_lr():.2e}  "
                        f"tau={self._current_tau():.4f}"
                    )

            if self.state.step % self.tcfg.val_every == 0:
                stats = self.validate()
                stats["step"] = self.state.step
                self.state.val_stats.append(stats)
                if verbose:
                    print(
                        f"       val_loss={stats['val_loss']:.5f}  "
                        f"z_ctx_std={stats['z_ctx_std']:.3f}  "
                        f"z_ctx_rank={stats['z_ctx_rank']:.1f}  "
                        f"z_tgt_std={stats['z_tgt_std']:.3f}  "
                        f"z_tgt_rank={stats['z_tgt_rank']:.1f}"
                    )

        self.state.wall_time_seconds = time.time() - start
        if verbose:
            print(f"\nFini en {self.state.wall_time_seconds:.1f} s")
        return self.state

    # ---- checkpoint ---- #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": self.mcfg.__dict__,
            },
            path / "model.pt",
        )

        log = {
            "model_config": {k: v for k, v in self.mcfg.__dict__.items() if k != "n_patches"},
            "trainer_config": self.tcfg.__dict__,
            "train_losses": self.state.train_losses,
            "val_stats": self.state.val_stats,
            "wall_time_seconds": self.state.wall_time_seconds,
        }
        (path / "log.json").write_text(json.dumps(log, indent=2, default=str))
