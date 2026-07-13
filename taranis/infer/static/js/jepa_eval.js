// TS-JEPA-3ch inference in the browser via onnxruntime-web.
//
// The ONNX graph takes a normalized window (B, 32, 3) and returns a
// probability (B,).  Normalization must match the training data:
//   x_norm = (x_raw - sensor_mean) / sensor_std
// where mean/std are stored alongside the model in tsjepa_3ch.meta.json.

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.mjs";

let _ort = null;
let _session = null;

async function loadOrt() {
    if (_ort) return _ort;
    _ort = await import(ORT_CDN);
    // Ask the runtime to fetch its wasm assets from the same CDN.
    _ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
    return _ort;
}

export async function loadJepaSession(modelUrl) {
    const ort = await loadOrt();
    _session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
    });
    return _session;
}

export async function predictJepa(bundle, window) {
    const ort = await loadOrt();
    const mean = bundle.sensor_mean;
    const std = bundle.sensor_std;
    // Flatten (Tw x 3) into a Float32Array and normalize on the fly.
    const Tw = window.length;
    const flat = new Float32Array(Tw * 3);
    for (let i = 0; i < Tw; i++) {
        for (let c = 0; c < 3; c++) {
            flat[i * 3 + c] = (window[i][c] - mean[c]) / std[c];
        }
    }
    const tensor = new ort.Tensor("float32", flat, [1, Tw, 3]);
    const out = await _session.run({ window: tensor });
    return out.proba.data[0];
}

export function alertFromProbaJepa(bundle, proba) {
    if (proba >= bundle.rouge_threshold) return "ROUGE";
    if (proba >= bundle.orange_threshold) return "ORANGE";
    return "VERT";
}
