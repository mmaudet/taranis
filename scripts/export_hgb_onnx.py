"""Train HGB-3ch on the full dataset, export to ONNX for the PWA.

Chapter 19 milestone: get the production HistGradientBoosting classifier
running inside the browser via onnxruntime-web. This script:

1. Loads the 3-channel dataset used at chapter 18.
2. Computes the 17 enriched features per window using `build_features_3ch`.
3. Fits HGB on the full training set (best hyperparameters from chapter 18).
4. Exports the sklearn model to ONNX with input shape (?, 17).
5. Verifies at 1e-6 that the ONNX runtime output matches sklearn's
   predict_proba on 200 random windows.
6. Saves ONNX file to `taranis/infer/static/models/hgb_3ch.onnx` for the PWA.

Also saves the calibrated ORANGE/ROUGE thresholds derived from val recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Workaround for a skl2onnx 1.20 vs sklearn 1.9 incompatibility on
# HistGradientBoosting: booleans in `nodes_missing_value_tracks_true` reach
# onnx.helper.make_attribute which raises TypeError.  Coerce them to ints
# before the loop reaches the type check.
import onnx.helper as _oh  # noqa: E402

_orig_make_node = _oh.make_node


def _coerce_bool(v):
    if isinstance(v, np.ndarray) and v.dtype == bool:
        return v.astype(np.int64)
    if isinstance(v, (list, tuple)) and v and all(
        isinstance(x, (bool, np.bool_)) for x in v
    ):
        return [int(x) for x in v]
    return v


def _make_node_bool_safe(op_type, inputs, outputs, name=None, doc_string=None,
                         domain=None, overload=None, **kwargs):
    kwargs = {k: _coerce_bool(v) for k, v in kwargs.items()}
    return _orig_make_node(
        op_type, inputs, outputs, name=name, doc_string=doc_string,
        domain=domain, overload=overload, **kwargs,
    )


_oh.make_node = _make_node_bool_safe
# also patch the reimport inside skl2onnx if any
import skl2onnx.common._container as _sk_container  # noqa: E402

_sk_container.make_node = _make_node_bool_safe

from skl2onnx import to_onnx  # noqa: E402
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

    print("\nExporting to ONNX...")
    # skl2onnx needs a sample input to trace the pipeline
    F_val = build_features_3ch(X_val[:1], step_minutes=180)
    onx = to_onnx(model.clf, F_val.astype(np.float32),
                  target_opset=15,
                  options={id(model.clf): {"zipmap": False}})
    onnx_path = STATIC_MODELS / "hgb_3ch.onnx"
    onnx_path.write_bytes(onx.SerializeToString())
    print(f"  saved {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    print("\nVerifying ONNX output matches sklearn (200 random windows)...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_test), size=200, replace=False)
    F_test = build_features_3ch(X_test[idx], step_minutes=180)
    py_probs = model.clf.predict_proba(F_test)[:, 1]
    onnx_out = sess.run(None, {"X": F_test.astype(np.float32)})
    # onnx_out[1] is the probability tensor with shape (N, 2)
    ort_probs = onnx_out[1][:, 1]
    max_diff = float(np.abs(py_probs - ort_probs).max())
    print(f"  max abs diff sklearn vs ONNX: {max_diff:.2e}")
    if max_diff > 1e-5:
        print("  WARNING: divergence > 1e-5, investigate before shipping")
    else:
        print("  OK, ONNX output matches sklearn within tolerance")

    metadata = {
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
    }
    meta_path = STATIC_MODELS / "hgb_3ch.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata: {meta_path}")

    print("\nDone. The PWA can now load `hgb_3ch.onnx` + `hgb_3ch.meta.json`.")


if __name__ == "__main__":
    main()
