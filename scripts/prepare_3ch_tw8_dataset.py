"""Slice the Tw=32 3-channel dataset to Tw=8 (24 h of lookback).

Chapter 20 motivation: a hiker's smartphone will hold at most a day of
BLE sensor history, not 96 h. Reslice keeps only the last 8 patches
(=24 h at 3 h step) of each window, aligning the training distribution
with the deployment distribution.

Labels, timestamps, station ids and split membership stay the same.
Only X is truncated to its last 8 time steps.

Output: data/real_combined_3ch_tw8_windows.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "real_combined_3ch_windows.npz"
DST = ROOT / "data" / "real_combined_3ch_tw8_windows.npz"

TW_NEW = 8


def main():
    print(f"Loading {SRC}")
    z = np.load(SRC, allow_pickle=True)

    def slice_x(x):
        return x[:, -TW_NEW:, :].astype(np.float32)

    X_train = slice_x(z["X_train"])
    X_val = slice_x(z["X_val"])
    X_test = slice_x(z["X_test"])
    print(f"  X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")

    print(f"Saving {DST}")
    np.savez_compressed(
        DST,
        X_train=X_train,
        y_train=z["y_train"],
        ts_train=z["ts_train"],
        station_train=z["station_train"],
        X_val=X_val,
        y_val=z["y_val"],
        ts_val=z["ts_val"],
        station_val=z["station_val"],
        X_test=X_test,
        y_test=z["y_test"],
        ts_test=z["ts_test"],
        station_test=z["station_test"],
        mean=z["mean"],
        std=z["std"],
        canaux=z["canaux"],
        Tw=TW_NEW,
        H=z["H"],
        step_minutes=z["step_minutes"],
        source=str(z["source"]) + f" | Tw sliced to {TW_NEW} (24 h)",
    )
    print(f"OK: {DST.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
