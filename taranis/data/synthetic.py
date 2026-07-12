"""Synthetic weather generator with pre-storm regimes.

Pedagogical goal. We do not aim at a physically faithful model but at a
realistic multivariate signal in which storm onset shows a recognisable
signature: pressure drop, humidity rise, wind gust, slight temperature dip.
This is the kind of signature the physics baseline and TS-JEPA must learn to
detect.

Output convention: a DataFrame indexed on a regular time step, with the
minimal columns `timestamp, pressure, temp, humidity, wind` plus two labels
`storm_active` and `storm_onset`. These labels are only used for evaluation
and by the downstream probe, not for JEPA pre-training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StormProfile:
    """Perturbation applied around a storm onset.

    Amplitudes are relative to the baseline. Durations are in minutes.
    The profile follows three phases: approach (before onset), peak (right
    after), slow recovery. This is a deliberate caricature, visually readable.
    """

    approach_min: int = 240  # approach duration before onset (slow barometric drop)
    active_min: int = 60     # duration during which the storm is active
    recovery_min: int = 180  # recovery duration after the peak

    d_pressure_hpa: float = -6.0   # maximum pressure drop
    d_humidity_pct: float = 30.0   # humidity rise
    d_wind_ms: float = 8.0         # wind gust
    d_temp_c: float = -4.0         # temperature dip


def _storm_kernel(profile: StormProfile, step_min: int) -> np.ndarray:
    """Single time envelope indexed by time step.

    Returns a vector of coefficients in [0, 1] that modulates the amplitude
    of each channel. Three segments:
        - smooth ramp up over `approach_min` minutes,
        - plateau at 1 over `active_min` minutes,
        - exponential decay over `recovery_min` minutes.
    """
    n_app = max(1, profile.approach_min // step_min)
    n_act = max(1, profile.active_min // step_min)
    n_rec = max(1, profile.recovery_min // step_min)

    approach = np.linspace(0.0, 1.0, n_app, endpoint=False)
    active = np.ones(n_act)
    # exponential decay reaching about 5 % at the end of recovery
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
    """Superimpose the storm perturbation on `signal`, centred on `onset_idx`.

    The kernel starts `n_approach` steps before onset (approach period),
    reaches its plateau at onset, then decays. Truncated cleanly on both
    edges when the onset is too close to the beginning or the end.
    """
    start = onset_idx - n_approach
    k_lo = max(0, -start)          # truncate kernel start if onset is too early
    s_lo = max(0, start)           # position within the signal
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
    """Simulate a multivariate sensor stream over `days` days.

    Parameters
    ----------
    days : float
        Simulated duration, in days.
    step_minutes : int
        Sampling step. Must divide 1440 to stay readable.
    storms_per_day : float
        Average storm rate per day, drawn from a Poisson distribution.
    seed : int
        Generator seed, for reproducibility.
    start : str
        ISO start date, serves as a time anchor.
    profile : StormProfile | None
        Storm signature parameters. Default: readable caricature.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, pressure, temp, humidity, wind, storm_active,
        storm_onset. Chronological order, constant time step.
    """
    if profile is None:
        profile = StormProfile()

    rng = np.random.default_rng(seed)

    n = int(round(days * 24 * 60 / step_minutes))
    t = np.arange(n)
    # daily phase, in fraction of a day
    day_frac = (t * step_minutes) / (24 * 60)

    # pressure baseline: slow synoptic breathing plus small noise
    slow_wave = 5.0 * np.sin(2 * np.pi * day_frac / 4.0)  # period ~4 days
    pressure = 1013.0 + slow_wave + rng.normal(0.0, 0.3, size=n).cumsum() * 0.05

    # temperature: sharp diurnal cycle plus small noise
    temp = 15.0 + 8.0 * np.sin(2 * np.pi * (day_frac - 0.25)) + rng.normal(0.0, 0.5, n)

    # humidity: anti-correlated with temperature, plus slow noise
    humidity = 60.0 - 1.5 * (temp - 15.0) + rng.normal(0.0, 3.0, n)

    # wind: low baseline plus noise
    wind = 2.0 + np.abs(rng.normal(0.0, 0.8, n))

    # sample storm onsets: Poisson over the total duration
    expected_events = storms_per_day * days
    n_events = rng.poisson(expected_events)
    kernel = _storm_kernel(profile, step_minutes)
    n_approach = max(1, profile.approach_min // step_minutes)
    n_active = max(1, profile.active_min // step_minutes)
    # margin so the approach period is visible before the onset
    onsets = np.sort(rng.integers(low=n_approach, high=max(n_approach + 1, n), size=n_events))

    storm_onset = np.zeros(n, dtype=bool)
    storm_active = np.zeros(n, dtype=bool)

    for onset_idx in onsets:
        storm_onset[onset_idx] = True
        # active window: from onset to the end of the plateau
        active_end = min(n, onset_idx + n_active)
        storm_active[onset_idx:active_end] = True

        # physical perturbations centred on onset
        _apply_storm(pressure, onset_idx, kernel, n_approach, profile.d_pressure_hpa)
        _apply_storm(humidity, onset_idx, kernel, n_approach, profile.d_humidity_pct)
        _apply_storm(wind, onset_idx, kernel, n_approach, profile.d_wind_ms)
        _apply_storm(temp, onset_idx, kernel, n_approach, profile.d_temp_c)

    # physical bounds
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
