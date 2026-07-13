"""Sensor-focused baseline: 15 features on 3 channels (P, T, HR).

For chapter 18's sensor constraint experiment. A RuuviTag Pro (or any
comparable BLE portable sensor) provides only pressure, temperature,
and humidity. We restrict the enriched feature set to what those three
channels can produce:

- 6 pressure features (current, trends 3/6/12h, min 24h, std 24h)
- 4 temperature (current, amplitude, mean 12h, dropoff 3h)
- 4 humidity (current, mean 6h, delta 6h, max 12h)
- 2 dew-point-adjacent (Magnus-Tetens estimate, T-Td spread)
- 1 interaction (pressure_trend x humidity)

Wind and gust features from chapter 17 are dropped, and interactions
involving wind are dropped too. Total: 17. Slightly fewer than the
25 of the enriched baseline, but honest with the sensor contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES_3CH = (
    # pressure
    "p_last", "p_trend_3h", "p_trend_6h", "p_trend_12h", "p_min_24h", "p_std_24h",
    # temperature
    "t_last", "t_amplitude", "t_mean_12h", "t_dropoff_3h",
    # humidity
    "h_last", "h_mean_6h", "h_delta_6h", "h_max_12h",
    # dew-point-adjacent
    "td_estimate", "td_spread",
    # interaction (single one, physically meaningful)
    "px_p_trend_x_h",
)


def _dew_point_c(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Approximate dew point via Magnus-Tetens (deg C)."""
    a, b = 17.625, 243.04
    rh = np.clip(rh_pct, 0.1, 100.0)
    lam = np.log(rh / 100.0) + a * temp_c / (b + temp_c)
    return b * lam / (a - lam)


def build_features_3ch(
    X: np.ndarray,
    step_minutes: int = 180,
) -> np.ndarray:
    """Compute 17 features from (N, Tw, 3) windows.

    Channels order: pressure, temp, humidity.
    """
    if X.ndim != 3 or X.shape[-1] < 3:
        raise ValueError(f"expected (N, Tw, >=3), got {X.shape}")

    pressure = X[:, :, 0]
    temp = X[:, :, 1]
    humidity = X[:, :, 2]

    Tw = X.shape[1]

    def lag(hours):
        return max(1, min(hours * 60 // step_minutes, Tw - 1))

    lag_3h = lag(3)
    lag_6h = lag(6)
    lag_12h = lag(12)
    lag_24h = lag(24)

    # --- pressure ---
    p_last = pressure[:, -1]
    p_trend_3h = p_last - pressure[:, -1 - lag_3h]
    p_trend_6h = p_last - pressure[:, -1 - lag_6h]
    p_trend_12h = p_last - pressure[:, -1 - lag_12h]
    p_min_24h = pressure[:, -1 - lag_24h :].min(axis=1)
    p_std_24h = pressure[:, -1 - lag_24h :].std(axis=1)

    # --- temperature ---
    t_last = temp[:, -1]
    t_amplitude = temp.max(axis=1) - temp.min(axis=1)
    t_mean_12h = temp[:, -1 - lag_12h :].mean(axis=1)
    t_dropoff_3h = t_last - temp[:, -1 - lag_3h]

    # --- humidity ---
    h_last = humidity[:, -1]
    h_mean_6h = humidity[:, -1 - lag_6h :].mean(axis=1)
    h_delta_6h = h_last - humidity[:, -1 - lag_6h]
    h_max_12h = humidity[:, -1 - lag_12h :].max(axis=1)

    # --- dew point ---
    td_estimate = _dew_point_c(t_last, h_last)
    td_spread = t_last - td_estimate

    # --- interaction ---
    px_p_trend_x_h = p_trend_6h * h_last

    feats = np.stack(
        [
            p_last, p_trend_3h, p_trend_6h, p_trend_12h, p_min_24h, p_std_24h,
            t_last, t_amplitude, t_mean_12h, t_dropoff_3h,
            h_last, h_mean_6h, h_delta_6h, h_max_12h,
            td_estimate, td_spread,
            px_p_trend_x_h,
        ],
        axis=1,
    ).astype(np.float32)
    return feats


@dataclass
class Baseline3ch:
    """3-channel baseline: 17 features + LogisticRegression."""

    step_minutes: int = 180
    C: float = 1.0
    class_weight: str | None = "balanced"

    scaler: StandardScaler = field(default_factory=StandardScaler)
    clf: LogisticRegression = field(init=False)

    def __post_init__(self):
        self.clf = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=2000,
            solver="lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Baseline3ch:
        feats = build_features_3ch(X, self.step_minutes)
        feats = self.scaler.fit_transform(feats)
        self.clf.fit(feats, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = build_features_3ch(X, self.step_minutes)
        feats = self.scaler.transform(feats)
        return self.clf.predict_proba(feats)[:, 1]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES_3CH, self.clf.coef_[0].tolist(), strict=True))


@dataclass
class Baseline3chHGB:
    """3-channel baseline: 17 features + HistGradientBoosting."""

    step_minutes: int = 180
    max_iter: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1

    clf: HistGradientBoostingClassifier = field(init=False)

    def __post_init__(self):
        self.clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            class_weight="balanced",
            random_state=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Baseline3chHGB:
        feats = build_features_3ch(X, self.step_minutes)
        self.clf.fit(feats, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = build_features_3ch(X, self.step_minutes)
        return self.clf.predict_proba(feats)[:, 1]
