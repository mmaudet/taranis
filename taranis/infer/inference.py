"""Cœur d'inférence Taranis.

Charge un encodeur TS-JEPA gelé, sa sonde linéaire aval, et expose une
fonction `predict(window)` qui renvoie :

- la probabilité brute d'orage,
- un niveau d'alerte discret (`AlertLevel.VERT`, `ORANGE`, `ROUGE`),
- un disclaimer de sécurité,
- des métadonnées utiles au front (canaux, période, timestamps).

Les seuils d'alerte s'appuient sur le **seuil de val** de la sonde
(`threshold` sauvegardé par `scripts/save_probe.py`), noté `T`. On
construit trois zones :

- `< T * low_ratio` → VERT (aucun signal notable).
- `[T * low_ratio, T]` → ORANGE (vigilance, signal modéré).
- `> T` → ROUGE (alerte forte, dépasse le seuil optimisé F1).

Par défaut `low_ratio = 0.5`. Ce paramétrage privilégie **la sécurité en
montagne** : on préfère alerter un peu trop (faux positifs) qu'oublier
un vrai orage (faux négatif). Il est configurable au chargement pour
permettre le tuning côté produit.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

from taranis.models import TSJEPA, TSJEPAConfig

DISCLAIMER = (
    "Taranis est une aide à la décision, pas une garantie. "
    "Ne remplace pas les bulletins officiels ni le jugement de l'utilisateur. "
    "En montagne, un faux négatif peut être mortel."
)


class AlertLevel(StrEnum):
    VERT = "vert"
    ORANGE = "orange"
    ROUGE = "rouge"

    @property
    def color(self) -> str:
        return {"vert": "#137333", "orange": "#E37400", "rouge": "#B00020"}[self.value]

    @property
    def french(self) -> str:
        return {
            "vert": "Aucune alerte",
            "orange": "Vigilance",
            "rouge": "Alerte orage",
        }[self.value]


def alert_from_proba(
    proba: float, threshold: float, low_ratio: float = 0.5
) -> AlertLevel:
    """Convertit une probabilité brute en niveau d'alerte.

    - ROUGE si `proba > threshold` (au-dessus du seuil optimisé F1 sur val).
    - ORANGE si `proba > threshold * low_ratio` (moitié du seuil par défaut).
    - VERT sinon.
    """
    if proba > threshold:
        return AlertLevel.ROUGE
    if proba > threshold * low_ratio:
        return AlertLevel.ORANGE
    return AlertLevel.VERT


@dataclass
class Prediction:
    proba: float
    alert: AlertLevel
    threshold: float
    canaux: list[str]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "proba_orage": float(self.proba),
            "alerte": self.alert.value,
            "alerte_libelle": self.alert.french,
            "alerte_couleur": self.alert.color,
            "seuil_reference": float(self.threshold),
            "canaux": list(self.canaux),
            "disclaimer": self.disclaimer,
        }


def load_probe(path: str | Path) -> dict:
    """Charge le pickle produit par `scripts/save_probe.py`."""
    path = Path(path)
    if path.is_dir():
        path = path / "probe.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


class Predictor:
    """Wrapper d'inférence, prêt à être appelé depuis une API."""

    def __init__(
        self,
        probe_path: str | Path,
        low_ratio: float = 0.5,
    ):
        payload = load_probe(probe_path)
        cfg_dict = {k: v for k, v in payload["encoder_config"].items() if k != "n_patches"}
        self.config = TSJEPAConfig(**cfg_dict)
        self.canaux: list[str] = list(payload["canaux"])
        self.mean = np.asarray(payload["mean"], dtype=np.float32)
        self.std = np.asarray(payload["std"], dtype=np.float32)
        self.threshold: float = float(payload["threshold"])
        self.low_ratio = low_ratio

        # coefficients de la sonde logistique
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
        self.clf_coef = np.asarray(payload["clf_coef"], dtype=np.float32).squeeze()
        self.clf_intercept = float(np.asarray(payload["clf_intercept"]).squeeze())

        # encodeur gelé
        encoder_ckpt = Path(payload["encoder_path"]) / "model.pt"
        self.encoder = TSJEPA(self.config)
        state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state["model_state"])
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    # ---- inférence ---- #

    def _embed(self, X_norm: np.ndarray) -> np.ndarray:
        """Encode + mean-pool sur les patches. Entrée : (B, Tw, V) normalisée."""
        x = torch.from_numpy(X_norm).float()
        with torch.no_grad():
            p = self.encoder.patch_embed(x)
            positions = torch.arange(p.size(1))
            p = p + self.encoder.pos_embed(positions).unsqueeze(0)
            z = self.encoder.encoder(p)
            return z.mean(dim=1).numpy()

    def _logistic(self, z: np.ndarray) -> np.ndarray:
        z_scaled = (z - self.scaler_mean) / self.scaler_scale
        logits = z_scaled @ self.clf_coef + self.clf_intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_from_raw(self, X_raw: np.ndarray) -> Prediction:
        """Prédit à partir d'une fenêtre en **unités physiques**.

        Attend `X_raw` de forme `(Tw, V)` avec les canaux dans l'ordre de
        `self.canaux`. On normalise avec les stats du train.
        """
        if X_raw.ndim != 2 or X_raw.shape[1] != len(self.canaux):
            raise ValueError(
                f"attendu (Tw, {len(self.canaux)}), obtenu {X_raw.shape}"
            )
        if X_raw.shape[0] != self.config.Tw:
            raise ValueError(
                f"attendu Tw={self.config.Tw} pas de temps, obtenu {X_raw.shape[0]}"
            )
        X_norm = ((X_raw - self.mean) / self.std).astype(np.float32)[None, :, :]
        z = self._embed(X_norm)
        proba = float(self._logistic(z)[0])
        alert = alert_from_proba(proba, self.threshold, self.low_ratio)
        return Prediction(
            proba=proba, alert=alert, threshold=self.threshold, canaux=self.canaux
        )

    def predict_from_norm(self, X_norm: np.ndarray) -> Prediction:
        """Variante quand la fenêtre est déjà normalisée (test set, benchmark)."""
        if X_norm.shape != (self.config.Tw, len(self.canaux)):
            raise ValueError(
                f"attendu ({self.config.Tw}, {len(self.canaux)}), obtenu {X_norm.shape}"
            )
        z = self._embed(X_norm.astype(np.float32)[None, :, :])
        proba = float(self._logistic(z)[0])
        alert = alert_from_proba(proba, self.threshold, self.low_ratio)
        return Prediction(
            proba=proba, alert=alert, threshold=self.threshold, canaux=self.canaux
        )
