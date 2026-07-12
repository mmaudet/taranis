"""Taranis inference core.

Loads a frozen TS-JEPA encoder and its downstream linear probe, then
exposes a `predict(window)` function that returns:

- the raw storm probability,
- a discrete alert level (`AlertLevel.VERT`, `ORANGE`, `ROUGE`),
- a safety disclaimer,
- metadata useful for the front-end (channels, period, timestamps).

Alert thresholds are built from the probe's **validation threshold**
(`threshold` saved by `scripts/save_probe.py`), noted `T`. Three zones:

- `< T * low_ratio`   : VERT (no notable signal).
- `[T * low_ratio, T]`: ORANGE (vigilance, moderate signal).
- `> T`               : ROUGE (strong alert, above the F1-optimised threshold).

Default `low_ratio = 0.5`. This favours **mountain safety**: better to
alert a bit too often (false positives) than to miss a real storm (false
negative). It is configurable at load time to allow product-side tuning.
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
    proba: float,
    orange_threshold: float,
    rouge_threshold: float,
) -> AlertLevel:
    """Convert a raw probability to an alert level using two thresholds.

    - ROUGE if `proba >= rouge_threshold` (strong alert).
    - ORANGE if `proba >= orange_threshold` (vigilance).
    - VERT otherwise.

    Both thresholds are **recalibrated on validation** by target recall,
    to favour safety (see `calibrate_alert_thresholds`).
    """
    if proba >= rouge_threshold:
        return AlertLevel.ROUGE
    if proba >= orange_threshold:
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
    """Load the pickle produced by `scripts/save_probe.py`."""
    path = Path(path)
    if path.is_dir():
        path = path / "probe.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


class Predictor:
    """Inference wrapper, ready to be called from an API."""

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
        # alert thresholds recalibrated by target recall on validation, when
        # present in the pickle. Otherwise fall back on the F1-max threshold
        # with low_ratio as the ORANGE cue (backward compatibility).
        if "orange_threshold" in payload and "rouge_threshold" in payload:
            self.orange_threshold = float(payload["orange_threshold"])
            self.rouge_threshold = float(payload["rouge_threshold"])
            self.calibration = payload.get("calibration", {})
        else:
            self.orange_threshold = self.threshold * low_ratio
            self.rouge_threshold = self.threshold
            self.calibration = {}

        # logistic probe coefficients
        self.scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
        self.clf_coef = np.asarray(payload["clf_coef"], dtype=np.float32).squeeze()
        self.clf_intercept = float(np.asarray(payload["clf_intercept"]).squeeze())

        # frozen encoder
        encoder_ckpt = Path(payload["encoder_path"]) / "model.pt"
        self.encoder = TSJEPA(self.config)
        state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state["model_state"])
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    # ---- inference ---- #

    def _embed(self, X_norm: np.ndarray) -> np.ndarray:
        """Encode + mean-pool over patches. Input: (B, Tw, V), normalised."""
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
        """Predict from a window in **physical units**.

        Expects `X_raw` of shape `(Tw, V)` with channels in the order of
        `self.canaux`. Normalisation uses training statistics.
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
        alert = alert_from_proba(proba, self.orange_threshold, self.rouge_threshold)
        return Prediction(
            proba=proba, alert=alert, threshold=self.threshold, canaux=self.canaux
        )

    def predict_from_norm(self, X_norm: np.ndarray) -> Prediction:
        """Variant when the window is already normalised (test set, benchmark)."""
        if X_norm.shape != (self.config.Tw, len(self.canaux)):
            raise ValueError(
                f"attendu ({self.config.Tw}, {len(self.canaux)}), obtenu {X_norm.shape}"
            )
        z = self._embed(X_norm.astype(np.float32)[None, :, :])
        proba = float(self._logistic(z)[0])
        alert = alert_from_proba(proba, self.orange_threshold, self.rouge_threshold)
        return Prediction(
            proba=proba, alert=alert, threshold=self.threshold, canaux=self.canaux
        )
