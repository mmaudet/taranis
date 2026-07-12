"""Entraîne la baseline physique M0 sur les données Météo-France réelles.

Sortie :

- `runs/baseline_real/report.json` : métriques test + coefficients + seuil.
- `docs/assets/step2_real_curves.png` : ROC et courbe précision-rappel test.

Ce script est jumeau de `train_baseline.py`. Différences :

- Pas de temps 3 heures (SYNOP) au lieu de 10 minutes,
- Lags physiques adaptés : court = 3 h, long = 12 h,
- Fichier d'entrée `data/real_windows.npz` au lieu du synthétique.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from taranis.data import CANAUX
from taranis.eval import classification_report, pr_curve, roc_curve, tune_threshold_on_val
from taranis.models import BaselinePhysics
from taranis.models.baseline_physics import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs" / "baseline_real"
ASSETS = ROOT / "docs" / "assets"
RUNS.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)


def load_windows():
    z = np.load(DATA / "real_windows.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def dénormalise(X_norm, mean, std):
    return (X_norm * std + mean).astype(np.float32)


def plot_curves(y_test, score_test, filename):
    fpr, tpr, _ = roc_curve(y_test, score_test)
    prec, rec, _ = pr_curve(y_test, score_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="tab:blue")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[0].set_xlabel("Taux de faux positifs")
    axes[0].set_ylabel("Taux de vrais positifs")
    axes[0].set_title("ROC (test réel, 2024)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="tab:orange")
    axes[1].set_xlabel("Rappel")
    axes[1].set_ylabel("Précision")
    axes[1].set_title("Précision-rappel (test réel, 2024)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ASSETS / filename, dpi=120)
    plt.close(fig)


def main():
    data = load_windows()
    mean, std = data["mean"], data["std"]
    step_min = int(data["step_minutes"].item())

    X_train = dénormalise(data["X_train"], mean, std)
    X_val = dénormalise(data["X_val"], mean, std)
    X_test = dénormalise(data["X_test"], mean, std)
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    print(f"Canaux : {list(data['canaux'])}, cohérence : {list(CANAUX)}")
    print(f"Train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"Prévalence : train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # sur des données à pas 3h : court = 3h (1 pas), long = 12h (4 pas)
    model = BaselinePhysics(
        step_minutes=step_min,
        short_lag_min=180,
        long_lag_min=720,
    ).fit(X_train, y_train)
    score_val = model.predict_proba(X_val)
    score_test = model.predict_proba(X_test)

    thr = tune_threshold_on_val(y_val, score_val, criterion="f1")
    report = classification_report(y_test, score_test, threshold=thr)
    print("\nRapport test :", report.as_line())

    coefs = model.coefficients()
    print("\nCoefficients (features standardisées) :")
    for name, w in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:24s}  {w:+.3f}")

    report_dict = {
        "model": "baseline_physics_real",
        "auc": report.auc,
        "average_precision": report.average_precision,
        "threshold": report.threshold,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "prevalence": report.prevalence,
        "n_test": report.n,
        "coefficients": coefs,
        "features": list(FEATURE_NAMES),
        "source": str(data["source"]),
    }
    (RUNS / "report.json").write_text(json.dumps(report_dict, indent=2))
    print(f"\nRapport sauvegardé : {RUNS / 'report.json'}")

    plot_curves(y_test, score_test, "step2_real_curves.png")
    print(f"Courbes ROC/PR : {ASSETS / 'step2_real_curves.png'}")


if __name__ == "__main__":
    main()
