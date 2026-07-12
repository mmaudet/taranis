"""Sonde linéaire aval sur un encodeur TS-JEPA gelé.

L'idée est simple. Une fois que TS-JEPA a été pré-entraîné en auto-supervisé,
on gèle son encodeur et on entraîne un tout petit classifieur linéaire sur
les embeddings pour la tâche aval (« un orage arrive-t-il dans l'horizon »).

Trois choses seulement se passent dans la sonde :

1. Chaque fenêtre `(Tw, V)` passe dans l'encodeur online gelé → `(N, D)` tokens.
2. On agrège par **mean pooling** sur les patches → un unique vecteur `(D,)`.
3. Une **régression logistique** apprend à séparer les classes sur ces vecteurs.

C'est le protocole classique d'évaluation d'une représentation apprise. Si
l'encodeur a capté quelque chose d'utile, la régression linéaire suffit.
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
    """Recharge un modèle TSJEPA à partir de son checkpoint `model.pt`."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = payload["model_config"]
    # `n_patches` est une propriété calculée, on ne la passe pas au constructeur
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
    """Passe les fenêtres dans l'encodeur online, moyenne sur les patches.

    Retour : `(N, D)`.
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
    """Sonde linéaire aval sur un encodeur TS-JEPA gelé."""

    encoder: TSJEPA
    C: float = 1.0
    class_weight: str | None = "balanced"
    scaler: StandardScaler = field(default_factory=StandardScaler)
    clf: LogisticRegression = field(init=False)

    def __post_init__(self):
        # gel des paramètres et mode eval
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
