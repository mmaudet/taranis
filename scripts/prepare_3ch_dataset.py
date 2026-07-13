"""Slice the combined stations dataset to the 3 channels a portable sensor provides.

Sensor targeted for Taranis field use is RuuviTag Pro (or equivalent BLE),
which measures pressure, temperature, humidity. It does not have wind or
gust sensors. So we amputate the SYNOP windows to those 3 channels only,
keeping everything else identical (splits, timestamps, labels, stations).

Output: data/real_combined_3ch_windows.npz, same schema, 3 channels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "real_combined_stations_windows.npz"
DST = ROOT / "data" / "real_combined_3ch_windows.npz"


def main():
    print(f"Loading {SRC}")
    z = np.load(SRC, allow_pickle=True)
    n_ch_keep = 3  # pressure, temp, humidity

    def slice_x(x):
        return x[:, :, :n_ch_keep].astype(np.float32)

    print(f"Slicing to first {n_ch_keep} channels ({list(z['canaux'][:n_ch_keep])})")
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
        mean=z["mean"][:n_ch_keep],
        std=z["std"][:n_ch_keep],
        canaux=z["canaux"][:n_ch_keep],
        Tw=z["Tw"],
        H=z["H"],
        step_minutes=z["step_minutes"],
        source=str(z["source"]) + " | sliced to 3 channels for portable sensor",
    )
    print(f"OK: {DST.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
