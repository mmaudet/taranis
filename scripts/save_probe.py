"""Save the trained downstream linear probe for inference use.

Inputs: a TS-JEPA checkpoint plus a prepared dataset (normalised X + y).
Output: `runs/probe/<name>/probe.pkl` with scaler, classifier, threshold,
normalisation stats, and information about the associated checkpoint.

This is what the API loads at startup. Saving the probe is separated
from training it so we can produce several variants (WMO labels, rain
labels, etc.) and point the API at one of them.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from taranis.eval import (
    LinearProbe,
    calibrate_alert_thresholds,
    classification_report,
    load_encoder_from_checkpoint,
    tune_threshold_on_val,
)

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description="Entraîne et sauve une sonde aval")
    p.add_argument("--encoder", required=True, help="chemin vers runs/tsjepa_<nom>")
    p.add_argument("--dataset", required=True, help="chemin vers data/<nom>_windows.npz")
    p.add_argument("--out", required=True, help="chemin vers runs/probe/<nom>")
    p.add_argument("--criterion", default="f1", choices=["f1", "recall"])
    p.add_argument("--recall-orange", type=float, default=0.70,
                   help="Cible de rappel pour le seuil ORANGE (vigilance)")
    p.add_argument("--recall-rouge", type=float, default=0.30,
                   help="Cible de rappel pour le seuil ROUGE (alerte)")
    args = p.parse_args()

    enc_path = ROOT / args.encoder
    ds_path = ROOT / args.dataset
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement encodeur : {enc_path}")
    enc = load_encoder_from_checkpoint(enc_path)

    print(f"Chargement dataset : {ds_path}")
    z = np.load(ds_path, allow_pickle=True)
    X_train = z["X_train"].astype(np.float32)
    X_val = z["X_val"].astype(np.float32)
    X_test = z["X_test"].astype(np.float32)
    y_train, y_val, y_test = z["y_train"], z["y_val"], z["y_test"]
    mean, std = z["mean"], z["std"]
    canaux = list(z["canaux"])

    print("Entraînement sonde…")
    probe = LinearProbe(enc).fit(X_train, y_train)
    print("Prédiction val + test…")
    score_val = probe.predict_proba(X_val)
    score_test = probe.predict_proba(X_test)
    thr = tune_threshold_on_val(y_val, score_val, criterion=args.criterion)
    report = classification_report(y_test, score_test, threshold=thr)
    print(f"  Point F1-max : {report.as_line()}")

    calib = calibrate_alert_thresholds(
        y_val, score_val,
        target_recall_orange=args.recall_orange,
        target_recall_rouge=args.recall_rouge,
    )
    print(
        f"\nCalibration seuils d'alerte (validation) :\n"
        f"  ORANGE  seuil={calib.orange:.3f}  rappel={calib.recall_orange:.3f}  "
        f"précision={calib.precision_orange:.3f}\n"
        f"  ROUGE   seuil={calib.rouge:.3f}  rappel={calib.recall_rouge:.3f}  "
        f"précision={calib.precision_rouge:.3f}\n"
        f"  prévalence val = {calib.prevalence:.4f}  n_val = {calib.n_val}"
    )

    payload = {
        "encoder_config": enc.config.__dict__,
        "encoder_path": str(enc_path),
        "dataset_path": str(ds_path),
        "canaux": canaux,
        "mean": mean,
        "std": std,
        "scaler_mean": probe.scaler.mean_,
        "scaler_scale": probe.scaler.scale_,
        "clf_coef": probe.clf.coef_,
        "clf_intercept": probe.clf.intercept_,
        "threshold": thr,
        "criterion": args.criterion,
        "orange_threshold": calib.orange,
        "rouge_threshold": calib.rouge,
        "calibration": {
            "target_recall_orange": args.recall_orange,
            "target_recall_rouge": args.recall_rouge,
            "recall_orange_val": calib.recall_orange,
            "precision_orange_val": calib.precision_orange,
            "recall_rouge_val": calib.recall_rouge,
            "precision_rouge_val": calib.precision_rouge,
            "prevalence_val": calib.prevalence,
            "n_val": calib.n_val,
        },
        "report": {
            "auc": report.auc,
            "average_precision": report.average_precision,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "prevalence": report.prevalence,
            "n": report.n,
        },
    }
    out = out_dir / "probe.pkl"
    with out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Sonde sauvée : {out}")


if __name__ == "__main__":
    main()
