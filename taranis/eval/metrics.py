"""Métriques d'évaluation communes à toutes les baselines et modèles.

On garde une API simple qui prend `y_true` (0/1) et `y_score` (probabilité)
et retourne un rapport structuré. Le seuil est ajusté sur le split de
validation, jamais sur le test. C'est explicitement séparé pour éviter la
fuite d'information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    roc_curve as sk_roc_curve,
)


@dataclass
class ClassificationReport:
    auc: float
    average_precision: float
    threshold: float
    precision: float
    recall: float
    f1: float
    prevalence: float
    n: int
    extra: dict = field(default_factory=dict)

    def as_line(self) -> str:
        return (
            f"AUC={self.auc:.3f}  AP={self.average_precision:.3f}  "
            f"thr={self.threshold:.3f}  P={self.precision:.3f}  "
            f"R={self.recall:.3f}  F1={self.f1:.3f}  "
            f"prev={self.prevalence:.3f}  n={self.n}"
        )


def tune_threshold_on_val(
    y_val: np.ndarray, score_val: np.ndarray, criterion: str = "f1"
) -> float:
    """Retourne le seuil qui maximise `criterion` sur la validation.

    En sécurité, on privilégiera plus tard `recall_at_precision` ou une
    contrainte de rappel minimal. Pour l'instant, `f1` sert de point de
    départ raisonnable.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, score_val)
    # `thresholds` a une longueur de `n - 1` par rapport à precisions/recalls
    p = precisions[:-1]
    r = recalls[:-1]
    if criterion == "f1":
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = 2 * p * r / np.where(p + r > 0, p + r, 1.0)
        best = int(np.nanargmax(f1))
    elif criterion == "recall":
        best = int(np.nanargmax(r))
    else:
        raise ValueError(f"critère inconnu : {criterion}")
    return float(thresholds[best])


def classification_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> ClassificationReport:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return ClassificationReport(
        auc=auc,
        average_precision=ap,
        threshold=threshold,
        precision=p,
        recall=r,
        f1=f1,
        prevalence=float(y_true.mean()),
        n=int(len(y_true)),
    )


def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """FPR, TPR, seuils."""
    return sk_roc_curve(y_true, y_score)


def pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Précisions, rappels, seuils."""
    return precision_recall_curve(y_true, y_score)
