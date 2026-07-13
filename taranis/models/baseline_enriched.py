"""Enriched physical baseline: 25 hand-crafted features.

Motivated by chapter 17's control experiment: does TS-JEPA really add
value beyond a richer physical baseline? We keep M0 simple (10 features)
as the pedagogical baseline, and add M0plus (25 features + LogReg) and
M0plusHGB (25 features + histogram gradient boosting) as fair
comparisons.

Features derive from the same 5 channels TS-JEPA sees. The set is
organised by physical intuition:

- 6 pressure-related features: current, trends over 3/6/12h, min 24h,
  standard deviation 24h.
- 4 temperature: current, amplitude, mean 12h, dropoff on last 3h.
- 4 humidity: current, mean 6h, delta 6h, max 12h.
- 6 wind: mean current + max 6h, gust current + max 6h + max 24h, delta 3h.
- 2 dew-point-adjacent: approximate dew point, T-Td spread.
- 3 physically motivated interactions: pressure_trend x humidity,
  gust_max x humidity, wind_delta x temperature_amplitude.

Total: 25.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES_ENRICHED = (
    # pressure
    "p_last", "p_trend_3h", "p_trend_6h", "p_trend_12h", "p_min_24h", "p_std_24h",
    # temperature
    "t_last", "t_amplitude", "t_mean_12h", "t_dropoff_3h",
    # humidity
    "h_last", "h_mean_6h", "h_delta_6h", "h_max_12h",
    # wind
    "w_last", "w_max_6h", "g_last", "g_max_6h", "g_max_24h", "w_delta_3h",
    # dew-point-adjacent (Magnus-Tetens on (T, RH))
    "td_estimate", "td_spread",
    # interactions
    "px_p_trend_x_h", "px_g_max_x_h", "px_w_delta_x_t_amp",
)


def _dew_point_c(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Approximate dew point via Magnus-Tetens (deg C)."""
    a, b = 17.625, 243.04
    rh = np.clip(rh_pct, 0.1, 100.0)
    lam = np.log(rh / 100.0) + a * temp_c / (b + temp_c)
    return b * lam / (a - lam)


def build_features_enriched(
    X: np.ndarray,
    step_minutes: int = 180,
) -> np.ndarray:
    """Compute 25 enriched features from (N, Tw, 5) windows.

    Channels order (from CANAUX_MF_RICH): pressure, temp, humidity, wind, wind_gust.
    Handles both short (Tw~8) and long (Tw=32) windows; steps that exceed
    the window duration are clipped to the window length.
    """
    if X.ndim != 3 or X.shape[-1] < 4:
        raise ValueError(f"expected (N, Tw, >=4), got {X.shape}")

    pressure = X[:, :, 0]
    temp = X[:, :, 1]
    humidity = X[:, :, 2]
    wind = X[:, :, 3]
    # gust: last channel if available, else wind
    gust = X[:, :, 4] if X.shape[-1] >= 5 else wind

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
    t_dropoff_3h = t_last - temp[:, -1 - lag_3h]  # negative = cooling

    # --- humidity ---
    h_last = humidity[:, -1]
    h_mean_6h = humidity[:, -1 - lag_6h :].mean(axis=1)
    h_delta_6h = h_last - humidity[:, -1 - lag_6h]
    h_max_12h = humidity[:, -1 - lag_12h :].max(axis=1)

    # --- wind ---
    w_last = wind[:, -1]
    w_max_6h = wind[:, -1 - lag_6h :].max(axis=1)
    g_last = gust[:, -1]
    g_max_6h = gust[:, -1 - lag_6h :].max(axis=1)
    g_max_24h = gust[:, -1 - lag_24h :].max(axis=1)
    w_delta_3h = w_last - wind[:, -1 - lag_3h]

    # --- dew point ---
    td_estimate = _dew_point_c(t_last, h_last)
    td_spread = t_last - td_estimate

    # --- interactions ---
    px_p_trend_x_h = p_trend_6h * h_last
    px_g_max_x_h = g_max_6h * h_last
    px_w_delta_x_t_amp = w_delta_3h * t_amplitude

    feats = np.stack(
        [
            p_last, p_trend_3h, p_trend_6h, p_trend_12h, p_min_24h, p_std_24h,
            t_last, t_amplitude, t_mean_12h, t_dropoff_3h,
            h_last, h_mean_6h, h_delta_6h, h_max_12h,
            w_last, w_max_6h, g_last, g_max_6h, g_max_24h, w_delta_3h,
            td_estimate, td_spread,
            px_p_trend_x_h, px_g_max_x_h, px_w_delta_x_t_amp,
        ],
        axis=1,
    ).astype(np.float32)
    return feats


@dataclass
class BaselineEnriched:
    """Enriched baseline: 25 features + LogisticRegression."""

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineEnriched:
        feats = build_features_enriched(X, self.step_minutes)
        feats = self.scaler.fit_transform(feats)
        self.clf.fit(feats, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = build_features_enriched(X, self.step_minutes)
        feats = self.scaler.transform(feats)
        return self.clf.predict_proba(feats)[:, 1]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES_ENRICHED, self.clf.coef_[0].tolist(), strict=True))


@dataclass
class BaselineEnrichedHGB:
    """Enriched baseline: 25 features + HistGradientBoosting.

    Captures non-linearities and interactions natively, without needing
    manual crosses. This is the strongest classical baseline we can
    reasonably run against TS-JEPA.
    """

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineEnrichedHGB:
        feats = build_features_enriched(X, self.step_minutes)
        self.clf.fit(feats, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = build_features_enriched(X, self.step_minutes)
        return self.clf.predict_proba(feats)[:, 1]
