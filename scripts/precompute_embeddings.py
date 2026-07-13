"""Precompute encoder embeddings (mean-pooled) for the whole dataset.

Reused by all rigorous evaluation scripts. Runs once, produces a numpy
file with Z_train, Z_val, Z_test (N, D) plus the encoder's config info.

Very useful because computing embeddings for 2M+ windows is by far the
slowest step. All downstream analyses (bootstrap, LOO, cross-regime) can
then work in pure numpy on (Z, y) pairs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from taranis.eval import load_encoder_from_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def encode_all(encoder, X, batch=512):
    """Mean-pool encoder over all windows, in batches."""
    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = torch.from_numpy(X[i:i + batch]).float()
            p = encoder.patch_embed(b)
            positions = torch.arange(p.size(1))
            p = p + encoder.pos_embed(positions).unsqueeze(0)
            z = encoder.encoder(p).mean(dim=1).numpy()
            outs.append(z)
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="runs/tsjepa_real_full_rich")
    ap.add_argument("--dataset", default="data/real_combined_stations_windows.npz")
    ap.add_argument("--out", default="data/embeddings_combined_stations.npz")
    args = ap.parse_args()

    print(f"Loading encoder: {args.encoder}")
    enc = load_encoder_from_checkpoint(ROOT / args.encoder)

    print(f"Loading dataset: {args.dataset}")
    z = np.load(ROOT / args.dataset, allow_pickle=True)

    t0 = time.time()
    print(f"Encoding train ({len(z['X_train']):,} windows)...")
    Z_train = encode_all(enc, z["X_train"].astype(np.float32))
    print(f"  done in {time.time() - t0:.1f} s, shape {Z_train.shape}")

    t0 = time.time()
    print(f"Encoding val ({len(z['X_val']):,})...")
    Z_val = encode_all(enc, z["X_val"].astype(np.float32))
    print(f"  done in {time.time() - t0:.1f} s")

    t0 = time.time()
    print(f"Encoding test ({len(z['X_test']):,})...")
    Z_test = encode_all(enc, z["X_test"].astype(np.float32))
    print(f"  done in {time.time() - t0:.1f} s")

    out = ROOT / args.out
    print(f"\nSaving to {out}...")
    np.savez_compressed(
        out,
        Z_train=Z_train.astype(np.float32),
        Z_val=Z_val.astype(np.float32),
        Z_test=Z_test.astype(np.float32),
        y_train=z["y_train"],
        y_val=z["y_val"],
        y_test=z["y_test"],
        ts_train=z["ts_train"],
        ts_val=z["ts_val"],
        ts_test=z["ts_test"],
        station_train=z["station_train"],
        station_val=z["station_val"],
        station_test=z["station_test"],
        encoder_config=str(enc.config.__dict__),
    )
    print(f"OK: {out}, {out.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
