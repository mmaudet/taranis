"""Baseline physique M0.

Elle est délibérément simple : de petites features intuitives extraites de
la fenêtre, puis une régression logistique. C'est ce qu'un baromètre et un
peu de bon sens font. Notre objectif n'est pas de la battre à tout prix,
c'est de la garder comme point de référence honnête. Tant que TS-JEPA ne la
bat pas, on documente le fait, on ne se raconte pas d'histoire.

Attention : ce module travaille sur des fenêtres **dénormalisées** ou
**normalisées**. Les features sont construites à partir de différences et de
statistiques dans la fenêtre, donc elles se comportent correctement dans les
deux cas. On documente clairement la sémantique de chaque feature.
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
    """Construit les features physiques à partir d'un batch de fenêtres.

    Deux échelles de tendance sont utilisées :

    - un lag court (`short_lag_min`, par défaut 1 heure),
    - un lag long (`long_lag_min`, par défaut 3 heures).

    Sur des données à pas de 10 minutes (synthétique), les défauts donnent
    exactement les tendances 1h et 3h de la baseline historique. Sur des
    données à pas de 3 heures (Météo-France SYNOP), on choisira des lags plus
    longs, par exemple 3 h et 12 h.

    Paramètres
    ----------
    X : np.ndarray, forme (N, Tw, 4)
        Ordre des canaux : pressure, temp, humidity, wind.
    step_minutes : int
        Pas de temps du signal.
    short_lag_min : int
        Durée en minutes du lag court.
    long_lag_min : int
        Durée en minutes du lag long.

    Retour
    ------
    np.ndarray, forme (N, len(FEATURE_NAMES))
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
    """Régression logistique sur features physiques.

    Le pipeline standardise les features avant la régression. C'est bénin ici
    mais utile pour la stabilité et l'interprétabilité des coefficients.
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
