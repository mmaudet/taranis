"""Evaluation metrics shared by all baselines and models.

Simple API: it takes `y_true` (0/1) and `y_score` (probability) and returns
a structured report. The threshold is tuned on the validation split, never
on test. This separation is enforced to prevent information leakage.
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
    """Return the threshold that maximises `criterion` on validation.

    For safety-critical use, we will later prefer `recall_at_precision` or
    a minimal-recall constraint. For now, `f1` is a reasonable default.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, score_val)
    # `thresholds` has length `n - 1` relative to precisions/recalls
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
    """FPR, TPR, thresholds."""
    return sk_roc_curve(y_true, y_score)


def pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Precisions, recalls, thresholds."""
    return precision_recall_curve(y_true, y_score)


@dataclass
class AlertThresholds:
    """GREEN/ORANGE/RED thresholds tuned by target recall on validation.

    - `orange`: threshold above which a vigilance (ORANGE) is raised.
    - `rouge` : threshold above which a strong alert (RED) is raised.

    We also expose the **actual** recall and precision obtained on
    validation at each threshold, to make the chosen trade-off visible.
    """

    orange: float
    rouge: float
    recall_orange: float
    precision_orange: float
    recall_rouge: float
    precision_rouge: float
    prevalence: float
    n_val: int


def calibrate_alert_thresholds(
    y_val: np.ndarray,
    score_val: np.ndarray,
    target_recall_orange: float = 0.70,
    target_recall_rouge: float = 0.30,
) -> AlertThresholds:
    """Choose two alert thresholds from recall targets on validation.

    Optimisation is oriented toward **mountain safety**: pick a target
    recall first (the fraction of true storms we want to catch), then find
    the matching threshold. Precision becomes a consequence, not an
    objective.

    - `target_recall_orange` (~0.70): catch ~70 % of storms for the
      vigilance level. Precision will be modest (many false positives), but
      only one storm out of three is missed.
    - `target_recall_rouge` (~0.30): only fire the strong alert when the
      model is really sure (top 30 % of storms); precision should be much
      higher so we do not trigger too often.

    Returns: `AlertThresholds` with the thresholds and their actual metrics.
    """
    if not (0 < target_recall_rouge <= target_recall_orange <= 1):
        raise ValueError(
            "attendu 0 < target_recall_rouge <= target_recall_orange <= 1"
        )

    y = np.asarray(y_val).astype(int)
    s = np.asarray(score_val).astype(float)
    precisions, recalls, thresholds = precision_recall_curve(y, s)
    # sklearn: precisions[i] / recalls[i] correspond to thresholds[i-1] (i>=1).
    # We align by ignoring the last value (recall=0, precision=1) and looking
    # only at points tied to actual thresholds.
    p = precisions[:-1]
    r = recalls[:-1]
    t = thresholds

    def _thr_for_recall(target):
        # recalls decrease as the threshold increases (indexed by t increasing);
        # we look for the LARGEST threshold guaranteeing recall >= target.
        idx = np.where(r >= target)[0]
        if len(idx) == 0:
            return float(t.min()), float(r.max()), float(p[int(np.argmax(r))])
        # take the last index (highest threshold that still meets the target)
        k = int(idx[-1])
        return float(t[k]), float(r[k]), float(p[k])

    thr_o, rec_o, prec_o = _thr_for_recall(target_recall_orange)
    thr_r, rec_r, prec_r = _thr_for_recall(target_recall_rouge)

    # sanity: RED > ORANGE (otherwise the RED recall target is too high)
    if thr_r < thr_o:
        thr_r = thr_o

    return AlertThresholds(
        orange=thr_o, rouge=thr_r,
        recall_orange=rec_o, precision_orange=prec_o,
        recall_rouge=rec_r, precision_rouge=prec_r,
        prevalence=float(y.mean()),
        n_val=int(len(y)),
    )
