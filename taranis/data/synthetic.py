"""Générateur météo synthétique avec régimes pré-orage.

Objectif pédagogique. On ne cherche pas un modèle physique fidèle, on cherche
un signal multivarié réaliste dans lequel l'apparition d'un orage se traduit
par une signature reconnaissable : chute de pression, montée d'humidité,
rafale de vent, léger fléchissement de la température. C'est ce genre de
signature que la baseline physique et TS-JEPA doivent apprendre à repérer.

Convention de sortie : un DataFrame indexé par un pas de temps régulier, avec
les colonnes minimales `timestamp, pressure, temp, humidity, wind` plus deux
étiquettes `storm_active` et `storm_onset`. Ces étiquettes ne servent qu'à
l'évaluation et à la sonde aval, pas au pré-entraînement JEPA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StormProfile:
    """Perturbation appliquée autour d'un onset d'orage.

    Les amplitudes sont relatives à la baseline. Les durées sont en minutes.
    Le profil suit trois temps : approche (avant onset), pic (juste après),
    récupération lente. C'est une caricature volontaire, lisible à l'œil.
    """

    approach_min: int = 240  # durée d'approche avant l'onset (chute barométrique lente)
    active_min: int = 60     # durée pendant laquelle l'orage est actif
    recovery_min: int = 180  # durée de la récupération après le pic

    d_pressure_hpa: float = -6.0   # chute de pression maximale
    d_humidity_pct: float = 30.0   # montée d'humidité
    d_wind_ms: float = 8.0         # rafale de vent
    d_temp_c: float = -4.0         # fléchissement de température


def _storm_kernel(profile: StormProfile, step_min: int) -> np.ndarray:
    """Enveloppe temporelle unique, indexée par pas de temps.

    Retourne un vecteur de coefficients dans [0, 1] qui module l'amplitude
    de chaque canal. Trois segments :
        - montée douce sur `approach_min` minutes,
        - plateau à 1 sur `active_min` minutes,
        - décroissance exponentielle sur `recovery_min` minutes.
    """
    n_app = max(1, profile.approach_min // step_min)
    n_act = max(1, profile.active_min // step_min)
    n_rec = max(1, profile.recovery_min // step_min)

    approach = np.linspace(0.0, 1.0, n_app, endpoint=False)
    active = np.ones(n_act)
    # décroissance exponentielle atteignant environ 5 % en fin de récupération
    tau = n_rec / 3.0
    recovery = np.exp(-np.arange(n_rec) / tau)

    return np.concatenate([approach, active, recovery])


def _apply_storm(
    signal: np.ndarray,
    onset_idx: int,
    kernel: np.ndarray,
    n_approach: int,
    amplitude: float,
) -> None:
    """Superpose la perturbation d'orage à `signal`, centrée sur `onset_idx`.

    Le kernel commence `n_approach` pas avant l'onset (période d'approche),
    atteint son plateau à l'onset, puis décroît. On tronque proprement aux
    deux bords si l'onset est trop proche du début ou de la fin.
    """
    start = onset_idx - n_approach
    k_lo = max(0, -start)          # tronque le début du kernel si onset trop tôt
    s_lo = max(0, start)           # position dans le signal
    end = min(len(signal), start + len(kernel))
    length = end - s_lo
    if length <= 0:
        return
    signal[s_lo:s_lo + length] += amplitude * kernel[k_lo:k_lo + length]


def generate(
    days: float = 30.0,
    step_minutes: int = 10,
    storms_per_day: float = 0.4,
    seed: int = 0,
    start: str = "2024-06-01",
    profile: StormProfile | None = None,
) -> pd.DataFrame:
    """Simule un flux capteur multivarié sur `days` jours.

    Paramètres
    ----------
    days : float
        Durée simulée, en jours.
    step_minutes : int
        Pas d'échantillonnage. Doit diviser 1440 pour rester lisible.
    storms_per_day : float
        Taux moyen d'orages par jour, suivi par une loi de Poisson.
    seed : int
        Graine du générateur, pour la reproductibilité.
    start : str
        Date de départ ISO, sert d'ancrage temporel.
    profile : StormProfile | None
        Paramètres de la signature d'orage. Valeur par défaut : caricature
        lisible.

    Retour
    ------
    pd.DataFrame
        Colonnes : timestamp, pressure, temp, humidity, wind, storm_active,
        storm_onset. Ordre chronologique, pas de temps constant.
    """
    if profile is None:
        profile = StormProfile()

    rng = np.random.default_rng(seed)

    n = int(round(days * 24 * 60 / step_minutes))
    t = np.arange(n)
    # phase diurne, en fraction de jour
    day_frac = (t * step_minutes) / (24 * 60)

    # baseline pression : lente respiration synoptique + petit bruit
    slow_wave = 5.0 * np.sin(2 * np.pi * day_frac / 4.0)  # période environ 4 jours
    pressure = 1013.0 + slow_wave + rng.normal(0.0, 0.3, size=n).cumsum() * 0.05

    # température : cycle diurne franc + petit bruit
    temp = 15.0 + 8.0 * np.sin(2 * np.pi * (day_frac - 0.25)) + rng.normal(0.0, 0.5, n)

    # humidité : anti-corrélée à la température, plus un bruit lent
    humidity = 60.0 - 1.5 * (temp - 15.0) + rng.normal(0.0, 3.0, n)

    # vent : baseline faible + bruit
    wind = 2.0 + np.abs(rng.normal(0.0, 0.8, n))

    # tirage des onsets d'orage : Poisson sur la durée totale
    expected_events = storms_per_day * days
    n_events = rng.poisson(expected_events)
    kernel = _storm_kernel(profile, step_minutes)
    n_approach = max(1, profile.approach_min // step_minutes)
    n_active = max(1, profile.active_min // step_minutes)
    # marge pour que la période d'approche soit visible avant l'onset
    onsets = np.sort(rng.integers(low=n_approach, high=max(n_approach + 1, n), size=n_events))

    storm_onset = np.zeros(n, dtype=bool)
    storm_active = np.zeros(n, dtype=bool)

    for onset_idx in onsets:
        storm_onset[onset_idx] = True
        # fenêtre active : depuis l'onset jusqu'à la fin du plateau
        active_end = min(n, onset_idx + n_active)
        storm_active[onset_idx:active_end] = True

        # perturbations physiques centrées sur l'onset
        _apply_storm(pressure, onset_idx, kernel, n_approach, profile.d_pressure_hpa)
        _apply_storm(humidity, onset_idx, kernel, n_approach, profile.d_humidity_pct)
        _apply_storm(wind, onset_idx, kernel, n_approach, profile.d_wind_ms)
        _apply_storm(temp, onset_idx, kernel, n_approach, profile.d_temp_c)

    # bornes physiques
    humidity = np.clip(humidity, 5.0, 100.0)
    wind = np.clip(wind, 0.0, None)

    ts = pd.date_range(start=start, periods=n, freq=f"{step_minutes}min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "pressure": pressure,
            "temp": temp,
            "humidity": humidity,
            "wind": wind,
            "storm_active": storm_active,
            "storm_onset": storm_onset,
        }
    )
