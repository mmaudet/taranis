"""Windowing and chronological splitting.

Two ideas to remember:

1. Each training example is a **window** of length `Tw`, a snapshot of the
   most recent measurements. From this window, we want to predict whether
   a storm occurs in the next `H` steps (the horizon).

2. On a time series, we **never** shuffle indices randomly. A chronological
   split protects against information leakage and reflects the real-world
   constraint: at prediction time, only past data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CANAUX = ("pressure", "temp", "humidity", "wind")


@dataclass(frozen=True)
class WindowedDataset:
    """Windowed dataset, ready to feed a model.

    Attributes
    ----------
    X : np.ndarray, shape (N, Tw, V)
        The windows. V = 4 physical channels in the order given by `CANAUX`.
    y : np.ndarray, shape (N,)
        Binary labels, 1 if at least one storm onset falls within the `H`
        steps following the window.
    timestamps : np.ndarray, shape (N,)
        Timestamp of the **end** of each window. Used for the split and for
        debugging.
    Tw : int
        Window length (number of time steps).
    H : int
        Prediction horizon (number of time steps).
    """

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    Tw: int
    H: int

    def __len__(self) -> int:
        return len(self.y)


def make_windows(
    df: pd.DataFrame,
    Tw: int = 96,
    H: int = 48,
    stride: int = 1,
    label_col: str = "storm_onset",
    canaux: tuple[str, ...] = CANAUX,
) -> WindowedDataset:
    """Slice a time series into windows, with a horizon label.

    For each window end at index `t`, the window covers `[t-Tw+1, t]` and
    the horizon covers `[t+1, t+H]`. We assign `y=1` if `label_col` is True
    at least once in that horizon, else `y=0`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in `canaux` and `label_col`. Assumes
        a regular time step.
    Tw : int
        Input window length.
    H : int
        Prediction horizon, in time steps.
    stride : int
        Offset between two successive windows. `1` maximises the number of
        examples; `Tw` removes any overlap.
    label_col : str
        Boolean column used to build `y`. Defaults to predicting the onset
        of a new storm (`storm_onset`).
    canaux : tuple[str, ...]
        Input columns in the desired order for tensor X.

    Returns
    -------
    WindowedDataset
    """
    if Tw < 1 or H < 1:
        raise ValueError("Tw et H doivent être strictement positifs")
    if stride < 1:
        raise ValueError("stride doit être >= 1")
    n = len(df)
    if n < Tw + H:
        raise ValueError(
            f"série trop courte : {n} pas, il en faut au moins Tw+H={Tw + H}"
        )

    X_arr = df[list(canaux)].to_numpy(dtype=np.float32)
    label = df[label_col].to_numpy(dtype=bool)
    ts = df["timestamp"].to_numpy()

    # valid window-end positions such that the horizon still fits
    ends = np.arange(Tw - 1, n - H, stride)
    N = len(ends)

    X = np.empty((N, Tw, len(canaux)), dtype=np.float32)
    y = np.empty(N, dtype=np.int64)
    end_ts = np.empty(N, dtype=ts.dtype)

    for i, t in enumerate(ends):
        X[i] = X_arr[t - Tw + 1 : t + 1]
        y[i] = int(label[t + 1 : t + 1 + H].any())
        end_ts[i] = ts[t]

    return WindowedDataset(X=X, y=y, timestamps=end_ts, Tw=Tw, H=H)


def chronological_split(
    ds: WindowedDataset,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Split a `WindowedDataset` into train, val, test in chronological order.

    Ratios are normalised to 1. The split occurs along the window axis in
    the order they were produced (chronological). No shuffling.

    Returns
    -------
    (train, val, test)
    """
    if len(ds) == 0:
        raise ValueError("dataset vide")
    s = sum(ratios)
    r_train, r_val, r_test = (r / s for r in ratios)
    n = len(ds)
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    # everything left goes to test, to keep every window
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    return (
        _slice(ds, idx_train),
        _slice(ds, idx_val),
        _slice(ds, idx_test),
    )


def _slice(ds: WindowedDataset, s: slice) -> WindowedDataset:
    return WindowedDataset(
        X=ds.X[s],
        y=ds.y[s],
        timestamps=ds.timestamps[s],
        Tw=ds.Tw,
        H=ds.H,
    )


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and standard deviation over all windows.

    Useful to normalise train, val and test with train-only statistics,
    which prevents any information leakage.

    Note: on very large datasets (millions of samples), we force accumulation
    in float64 to avoid the precision loss of a float32 sum (which can yield
    absurd results on ~10^8 values).
    """
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel normalisation, in float32."""
    return ((X - mean) / std).astype(np.float32)
