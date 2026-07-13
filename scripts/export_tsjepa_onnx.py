"""Export TS-JEPA-3ch (encoder + LogReg probe) as one ONNX graph.

Wraps the encoder + mean-pool + linear head + sigmoid into a single
torch.nn.Module, then uses torch.onnx.export.  Verifies parity with the
Python inference at 1e-5.

Input tensor: normalized window (B, Tw=32, C=3).
Output tensor: probability of storm (B,).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from taranis.eval import load_encoder_from_checkpoint

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


class TaranisJEPA(torch.nn.Module):
    """Encoder + mean-pool over patches + linear head + sigmoid.

    Takes normalized (B, 32, 3) windows, outputs probability of storm.
    """

    def __init__(self, encoder, w, b, scaler_mean, scaler_scale):
        super().__init__()
        self.encoder = encoder
        # Register head params as buffers so they are frozen in the ONNX graph
        self.register_buffer("w", torch.from_numpy(w.astype(np.float32)))
        self.register_buffer("b", torch.tensor(float(b)))
        self.register_buffer("s_mean", torch.from_numpy(scaler_mean.astype(np.float32)))
        self.register_buffer("s_scale", torch.from_numpy(scaler_scale.astype(np.float32)))

    def _encode(self, x):
        # Reuse the mean-pool logic that precompute_embeddings.py uses
        p = self.encoder.patch_embed(x)
        positions = torch.arange(p.size(1), device=x.device)
        p = p + self.encoder.pos_embed(positions).unsqueeze(0)
        z = self.encoder.encoder(p).mean(dim=1)
        return z

    def forward(self, x):
        z = self._encode(x)
        # standardize embeddings the same way the sklearn StandardScaler did
        z = (z - self.s_mean) / self.s_scale
        logit = (z * self.w).sum(dim=1) + self.b
        return torch.sigmoid(logit)


def main():
    print("Loading TS-JEPA-3ch encoder...")
    enc = load_encoder_from_checkpoint(ROOT / "runs" / "tsjepa_3ch")
    enc.eval()

    print("Loading precomputed embeddings...")
    e = np.load(DATA / "embeddings_3ch.npz", allow_pickle=True)
    Z_train, Z_val, Z_test = e["Z_train"], e["Z_val"], e["Z_test"]
    y_train, y_val, y_test = e["y_train"], e["y_val"], e["y_test"]

    print("Fitting LogReg probe...")
    scaler = StandardScaler().fit(Z_train)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"
    ).fit(scaler.transform(Z_train), y_train)
    s_val = clf.predict_proba(scaler.transform(Z_val))[:, 1]
    s_test = clf.predict_proba(scaler.transform(Z_test))[:, 1]
    auc_test = roc_auc_score(y_test, s_test)
    print(f"  AUC test = {auc_test:.4f}")

    thr_o, thr_r = calibrate_thresholds(y_val, s_val, target_o=0.70, target_r=0.30)
    print(f"  thresholds: ORANGE={thr_o:.4f}  ROUGE={thr_r:.4f}")

    print("Wrapping in TaranisJEPA...")
    w = clf.coef_[0].astype(np.float32)
    b = float(clf.intercept_[0])
    model = TaranisJEPA(enc, w, b, scaler.mean_, scaler.scale_).eval()

    print("Loading test windows for parity check...")
    d = np.load(DATA / "real_combined_3ch_windows.npz", allow_pickle=True)
    X_test_norm = d["X_test"][:5].astype(np.float32)
    with torch.no_grad():
        py_probs = model(torch.from_numpy(X_test_norm)).numpy()
    print(f"  Python probs (5 windows): {py_probs}")

    print("\nExporting to ONNX...")
    onnx_path = STATIC_MODELS / "tsjepa_3ch.onnx"
    dummy = torch.from_numpy(X_test_norm[:1])
    # Use legacy TorchScript-based export; it handles nn.MultiheadAttention
    # dynamic batch dim more reliably than dynamo for our tiny transformer.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["window"],
        output_names=["proba"],
        dynamic_axes={"window": {0: "batch"}, "proba": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  saved {onnx_path} ({size_kb:.1f} KB)")

    print("\nVerifying ONNX Runtime output matches PyTorch...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    ort_probs = sess.run(None, {"window": X_test_norm})[0]
    max_diff = float(np.abs(py_probs - ort_probs).max())
    print(f"  max abs diff torch vs ONNX: {max_diff:.2e}")
    if max_diff > 1e-5:
        print("  WARNING: divergence > 1e-5")
    else:
        print("  OK, ONNX output matches PyTorch within tolerance")

    metadata = {
        "model": "TS-JEPA-3ch",
        "channels": ["pressure", "temp", "humidity"],
        "tw": int(d["Tw"]),
        "step_minutes": int(d["step_minutes"]),
        "auc_test": float(auc_test),
        "orange_threshold": thr_o,
        "rouge_threshold": thr_r,
        # Windows must be normalized as (raw - sensor_mean) / sensor_std before
        # being passed to the ONNX graph. The graph then handles the internal
        # embedding standardization.
        "sensor_mean": d["mean"].tolist(),
        "sensor_std": d["std"].tolist(),
        "sample_probs": py_probs.tolist(),
    }
    meta_path = STATIC_MODELS / "tsjepa_3ch.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata: {meta_path}")

    print("\nDone. The PWA can load `tsjepa_3ch.onnx` + `tsjepa_3ch.meta.json`.")


if __name__ == "__main__":
    main()
