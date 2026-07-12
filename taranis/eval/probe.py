"""Downstream linear probe on a frozen TS-JEPA encoder.

Simple idea. Once TS-JEPA has been pre-trained in a self-supervised way,
we freeze its encoder and train a tiny linear classifier on the
embeddings for the downstream task ("is a storm about to happen in the
horizon?").

Only three things happen in the probe:

1. Each `(Tw, V)` window passes through the frozen online encoder,
   yielding `(N, D)` tokens.
2. We aggregate with **mean pooling** over patches, yielding a single
   `(D,)` vector.
3. A **logistic regression** learns to separate the classes on those
   vectors.

This is the standard protocol for evaluating a learned representation.
If the encoder has captured something useful, linear regression is
enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from taranis.models import TSJEPA, TSJEPAConfig


def load_encoder_from_checkpoint(path: str | Path) -> TSJEPA:
    """Reload a TSJEPA model from its `model.pt` checkpoint."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = payload["model_config"]
    # `n_patches` is a computed property, so do not pass it to the constructor
    cfg_dict = {k: v for k, v in cfg_dict.items() if k != "n_patches"}
    cfg = TSJEPAConfig(**cfg_dict)
    model = TSJEPA(cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _mean_pooled_embeddings(
    model: TSJEPA,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Push windows through the online encoder, average over patches.

    Returns: `(N, D)`.
    """
    model.eval()
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            p = model.patch_embed(batch)
            positions = torch.arange(p.size(1), device=device)
            p = p + model.pos_embed(positions).unsqueeze(0)
            z = model.encoder(p)  # (B, N, D)
            z_pooled = z.mean(dim=1)  # (B, D)
            outs.append(z_pooled.cpu().numpy())
    return np.concatenate(outs, axis=0)


@dataclass
class LinearProbe:
    """Downstream linear probe on a frozen TS-JEPA encoder."""

    encoder: TSJEPA
    C: float = 1.0
    class_weight: str | None = "balanced"
    scaler: StandardScaler = field(default_factory=StandardScaler)
    clf: LogisticRegression = field(init=False)

    def __post_init__(self):
        # freeze parameters and switch to eval mode
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.clf = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=2000,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearProbe:
        Z = _mean_pooled_embeddings(self.encoder, X)
        Z = self.scaler.fit_transform(Z)
        self.clf.fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = _mean_pooled_embeddings(self.encoder, X)
        Z = self.scaler.transform(Z)
        return self.clf.predict_proba(Z)[:, 1]
