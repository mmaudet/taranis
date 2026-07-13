// Node parity test for the JS HGB port.
// Loads the model + a test sample, computes probabilities in the same JS
// engine the browser will use, and prints max abs diff vs Python reference.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildFeatures3ch } from "../taranis/infer/static/js/features.js";
import { predictHGB } from "../taranis/infer/static/js/hgb_eval.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = join(__dirname, "..", "taranis", "infer", "static");

const EXPECTED = [
    0.5690484034481972,
    0.4922003219462329,
    0.3632262648956853,
    0.3947839504137553,
    0.3923577094070101,
];

const bundle = JSON.parse(await readFile(join(STATIC, "models", "hgb_3ch.json"), "utf8"));
const sample = JSON.parse(await readFile(join(STATIC, "models", "test_sample.json"), "utf8"));

let maxDiff = 0;
console.log("window  proba JS       proba Py       diff       alerte");
for (let i = 0; i < sample.windows.length; i++) {
    const f = buildFeatures3ch(sample.windows[i]);
    const p = predictHGB(bundle, f);
    const diff = Math.abs(p - EXPECTED[i]);
    maxDiff = Math.max(maxDiff, diff);
    const level = p >= bundle.rouge_threshold ? "ROUGE"
                : p >= bundle.orange_threshold ? "ORANGE" : "VERT";
    console.log(`  ${i}     ${p.toFixed(10)}  ${EXPECTED[i].toFixed(10)}  ${diff.toExponential(2)}  ${level}`);
}
console.log(`\nmax abs diff = ${maxDiff.toExponential(2)}`);
if (maxDiff > 1e-6) {
    console.log("FAIL: diff > 1e-6");
    process.exit(1);
} else {
    console.log("OK: JS port matches Python to <= 1e-6");
}
