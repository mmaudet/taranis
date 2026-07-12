"""Physics baseline M0.

Intentionally simple: small intuitive features extracted from the window,
then a logistic regression. This is what a barometer and a bit of common
sense would give you. The goal is not to beat it at all costs but to keep
it as an honest reference point. As long as TS-JEPA does not beat it, we
document that fact rather than pretending otherwise.

Note: this module operates on either **denormalised** or **normalised**
windows. Features are built from within-window differences and statistics,
so they behave correctly in both cases. Feature semantics are documented
clearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "pressure_last",
    "pressure_trend_short",
    "pressure_trend_long",
    "humidity_last",
    "humidity_mean_short",
    "humidity_delta_short",
    "wind_last",
    "wind_max_short",
    "temp_last",
    "temp_amplitude",
)


def build_features(
    X: np.ndarray,
    step_minutes: int = 10,
    short_lag_min: int = 60,
    long_lag_min: int = 180,
) -> np.ndarray:
    """Build the physics features from a batch of windows.

    Two trend scales are used:

    - a short lag (`short_lag_min`, default 1 hour),
    - a long lag (`long_lag_min`, default 3 hours).

    On 10-minute-step data (synthetic), the defaults yield exactly the
    1h/3h trends used by the historical baseline. On 3-hour-step data
    (Meteo-France SYNOP), longer lags are more appropriate, for instance
    3h and 12h.

    Parameters
    ----------
    X : np.ndarray, shape (N, Tw, 4)
        Channel order: pressure, temp, humidity, wind.
    step_minutes : int
        Signal time step.
    short_lag_min : int
        Short-lag duration in minutes.
    long_lag_min : int
        Long-lag duration in minutes.

    Returns
    -------
    np.ndarray, shape (N, len(FEATURE_NAMES))
    """
    if X.ndim != 3 or X.shape[-1] != 4:
        raise ValueError(f"attendu (N, Tw, 4), obtenu {X.shape}")

    pressure = X[:, :, 0]
    temp = X[:, :, 1]
    humidity = X[:, :, 2]
    wind = X[:, :, 3]

    Tw = X.shape[1]
    lag_short = max(1, min(short_lag_min // step_minutes, Tw - 1))
    lag_long = max(1, min(long_lag_min // step_minutes, Tw - 1))

    pressure_last = pressure[:, -1]
    pressure_trend_short = pressure[:, -1] - pressure[:, -1 - lag_short]
    pressure_trend_long = pressure[:, -1] - pressure[:, -1 - lag_long]

    humidity_last = humidity[:, -1]
    humidity_mean_short = humidity[:, -1 - lag_short :].mean(axis=1)
    humidity_delta_short = humidity[:, -1] - humidity[:, -1 - lag_short]

    wind_last = wind[:, -1]
    wind_max_short = wind[:, -1 - lag_short :].max(axis=1)

    temp_last = temp[:, -1]
    temp_amplitude = temp.max(axis=1) - temp.min(axis=1)

    return np.stack(
        [
            pressure_last,
            pressure_trend_short,
            pressure_trend_long,
            humidity_last,
            humidity_mean_short,
            humidity_delta_short,
            wind_last,
            wind_max_short,
            temp_last,
            temp_amplitude,
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class BaselinePhysics:
    """Logistic regression on physics features.

    The pipeline standardises features before the regression. Benign here,
    but helpful for numerical stability and coefficient interpretability.
    """

    step_minutes: int = 10
    short_lag_min: int = 60
    long_lag_min: int = 180
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

    def _feats(self, X: np.ndarray) -> np.ndarray:
        return build_features(
            X,
            step_minutes=self.step_minutes,
            short_lag_min=self.short_lag_min,
            long_lag_min=self.long_lag_min,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselinePhysics:
        feats = self._feats(X)
        feats = self.scaler.fit_transform(feats)
        self.clf.fit(feats, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = self._feats(X)
        feats = self.scaler.transform(feats)
        return self.clf.predict_proba(feats)[:, 1]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.clf.coef_[0].tolist(), strict=True))
