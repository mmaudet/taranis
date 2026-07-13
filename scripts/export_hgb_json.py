"""Serialize the trained HGB-3ch to a compact JSON tree bundle for the PWA.

Rationale: skl2onnx 1.20 has a boolean-in-attribute bug on sklearn 1.9's
HistGradientBoosting. Rather than fight the ONNX converter, we export the
tree ensemble to a simple JSON format that a 40-line JavaScript evaluator
can read. Result: the PWA does not need onnxruntime-web at all for HGB.

JSON layout:
{
  "n_features": 17,
  "feature_names": [...],
  "base_score": float,          # baseline prediction (mean of training log-odds)
  "trees": [
    {
      "feature": [int, ...],    # feature index per node, -1 for leaves
      "threshold": [float, ...],
      "left": [int, ...],       # index of left child, -1 for leaves
      "right": [int, ...],
      "value": [float, ...],    # leaf output value, 0 for internal nodes
    },
    ...
  ]
}

Prediction: start at node 0 of each tree, if feature[node] < threshold[node]
go left, else go right. When a leaf is reached, accumulate value[node].
Sum across all trees, add base_score, pass through logistic sigmoid.

Verifies output matches sklearn's predict_proba on 200 random windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from taranis.models import Baseline3chHGB
from taranis.models.baseline_3ch import FEATURE_NAMES_3CH, build_features_3ch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC_MODELS = ROOT / "taranis" / "infer" / "static" / "models"
STATIC_MODELS.mkdir(parents=True, exist_ok=True)


def calibrate_thresholds(y_val, s_val, target_o=0.70, target_r=0.30):
    _, r, thr = precision_recall_curve(y_val, s_val)
    r = r[:-1]
    idx_o = np.where(r >= target_o)[0]
    idx_r = np.where(r >= target_r)[0]
    thr_o = float(thr[idx_o[-1]]) if len(idx_o) else float(thr.min())
    thr_r = float(thr[idx_r[-1]]) if len(idx_r) else float(thr.max())
    if thr_r < thr_o:
        thr_r = thr_o
    return thr_o, thr_r


def denorm(X_norm, mean, std):
    return (X_norm * std + mean).astype(np.float32)


def tree_to_dict(predictor):
    """Convert one sklearn HGB tree predictor to a plain dict.

    HGB's TreePredictor.nodes is a numpy structured array where each row
    has the fields:
        value, count, feature_idx, num_threshold,
        missing_go_to_left, left, right, gain, depth, is_leaf, ...
    """
    nodes = predictor.nodes
    n = len(nodes)
    feature = []
    threshold = []
    left = []
    right = []
    value = []
    for i in range(n):
        node = nodes[i]
        is_leaf = bool(node["is_leaf"])
        if is_leaf:
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(float(node["value"]))
        else:
            feature.append(int(node["feature_idx"]))
            threshold.append(float(node["num_threshold"]))
            left.append(int(node["left"]))
            right.append(int(node["right"]))
            value.append(0.0)
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "value": value,
    }


def evaluate_trees(bundle, F):
    """Reference Python implementation of the JS evaluator.

    Used to verify parity vs sklearn's predict_proba.
    """
    N = F.shape[0]
    scores = np.full(N, bundle["base_score"], dtype=np.float64)
    for tree in bundle["trees"]:
        feat = np.asarray(tree["feature"])
        thr = np.asarray(tree["threshold"])
        left = np.asarray(tree["left"])
        right = np.asarray(tree["right"])
        val = np.asarray(tree["value"])
        for i in range(N):
            node = 0
            while feat[node] >= 0:
                if F[i, feat[node]] <= thr[node]:
                    node = left[node]
                else:
                    node = right[node]
            scores[i] += val[node]
    return 1.0 / (1.0 + np.exp(-scores))


def main():
    print("Loading 3-channel dataset...")
    d = np.load(DATA / "real_combined_3ch_windows.npz", allow_pickle=True)
    mean, std = d["mean"], d["std"]
    X_train = denorm(d["X_train"].astype(np.float32), mean, std)
    X_val = denorm(d["X_val"].astype(np.float32), mean, std)
    X_test = denorm(d["X_test"].astype(np.float32), mean, std)
    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]

    print(f"  train {len(y_train):,}, val {len(y_val):,}, test {len(y_test):,}")

    print("\nFitting HGB-3ch (200 trees, depth 6, lr 0.1)...")
    model = Baseline3chHGB(step_minutes=180, max_iter=200, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    s_val = model.predict_proba(X_val)
    s_test = model.predict_proba(X_test)
    auc_test = roc_auc_score(y_test, s_test)
    print(f"  AUC test = {auc_test:.4f}")

    thr_o, thr_r = calibrate_thresholds(y_val, s_val, target_o=0.70, target_r=0.30)
    print(f"  thresholds: ORANGE={thr_o:.4f}  ROUGE={thr_r:.4f}")

    clf = model.clf
    # HGB internals: clf._predictors is list of list of TreePredictor
    # For binary classification, shape is (n_iter, 1)
    trees = []
    for iter_preds in clf._predictors:
        # binary: iter_preds has length 1
        for tp in iter_preds:
            trees.append(tree_to_dict(tp))
    print(f"  serialized {len(trees)} trees")

    base_score = float(np.ravel(clf._baseline_prediction)[0])
    print(f"  base_score = {base_score:.6f}")

    bundle = {
        "model": "HGB-3ch",
        "features": list(FEATURE_NAMES_3CH),
        "channels": ["pressure", "temp", "humidity"],
        "tw": int(d["Tw"]),
        "step_minutes": int(d["step_minutes"]),
        "auc_test": float(auc_test),
        "orange_threshold": thr_o,
        "rouge_threshold": thr_r,
        "sensor_mean": mean.tolist(),
        "sensor_std": std.tolist(),
        "base_score": base_score,
        "trees": trees,
    }
    out_path = STATIC_MODELS / "hgb_3ch.json"
    out_path.write_text(json.dumps(bundle))  # compact
    size_kb = out_path.stat().st_size / 1024
    print(f"\nSaved {out_path} ({size_kb:.1f} KB)")

    print("\nVerifying JSON evaluator matches sklearn (200 random windows)...")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_test), size=200, replace=False)
    F_test = build_features_3ch(X_test[idx], step_minutes=180)
    py_probs = clf.predict_proba(F_test)[:, 1]
    js_probs = evaluate_trees(bundle, F_test)
    max_diff = float(np.abs(py_probs - js_probs).max())
    print(f"  max abs diff sklearn vs JSON reference: {max_diff:.2e}")
    if max_diff > 1e-6:
        print("  WARNING: divergence > 1e-6, investigate before shipping")
    else:
        print("  OK, JSON evaluator matches sklearn to machine precision")


if __name__ == "__main__":
    main()
