"""Train TS-JEPA on a prepared dataset, YAML config.

Usage:

    uv run python scripts/train_tsjepa.py configs/tsjepa_synth.yaml
    uv run python scripts/train_tsjepa.py configs/tsjepa_real.yaml

Outputs:

- `<output>/model.pt`   : model checkpoint (state_dict + config).
- `<output>/log.json`   : loss history + embedding statistics.
- `<output>/config.yaml`: copy of the config used.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from taranis.models import TSJEPAConfig
from taranis.train import TrainerConfig, TSJEPATrainer

ROOT = Path(__file__).resolve().parent.parent


def load_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    return z["X_train"].astype(np.float32), z["X_val"].astype(np.float32)


def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())

    dataset_path = ROOT / cfg["dataset"]
    output_path = ROOT / cfg["output"]
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset : {dataset_path}")
    X_train, X_val = load_dataset(dataset_path)
    print(f"  X_train {X_train.shape}, X_val {X_val.shape}")

    mcfg = TSJEPAConfig(**cfg["model"])
    tcfg = TrainerConfig(**cfg["trainer"])

    print(
        f"\nModèle : Tw={mcfg.Tw}, patch_len={mcfg.patch_len}, "
        f"n_patches={mcfg.n_patches}, d_model={mcfg.d_model}, "
        f"couches {mcfg.n_layers_encoder}+{mcfg.n_layers_predictor}"
    )
    print(
        f"Entraînement : {tcfg.n_steps} pas, batch={tcfg.batch_size}, "
        f"lr={tcfg.lr_max}, warmup={tcfg.warmup_steps}, "
        f"tau {tcfg.tau_start}->{tcfg.tau_end}\n"
    )

    trainer = TSJEPATrainer(mcfg, tcfg, X_train, X_val)
    trainer.fit(verbose=True)
    trainer.save(output_path)
    shutil.copy(config_path, output_path / "config.yaml")
    print(f"\nSauvegardé dans {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: train_tsjepa.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
