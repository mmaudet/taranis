// End-to-end simulation of the PWA data path in Node.
// Skips IndexedDB (browser-only) and uses plain arrays, but calls the
// exact same features + prediction + buildWindow code the browser will use.
//
// Success = the mock scenario (falling pressure + rising humidity over 30 h)
// produces a probability that reasonably escalates from VERT to ORANGE or ROUGE.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildFeatures3ch } from "../taranis/infer/static/js/features.js";
import { predictHGB, alertFromProba } from "../taranis/infer/static/js/hgb_eval.js";
import { buildWindow } from "../taranis/infer/static/js/buffer.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC = join(__dirname, "..", "taranis", "infer", "static");
const bundle = JSON.parse(await readFile(join(STATIC, "models", "hgb_3ch.json"), "utf8"));

const NOW = Date.now();
const HOUR = 3600 * 1000;

function baseState(hoursAgo) {
    const t = 100 - hoursAgo;
    const p = 1013 - 0.20 * t + 1.5 * Math.sin(t / 6);
    const temp = 18 + 4 * Math.sin(((t + 6) / 24) * 2 * Math.PI);
    const h = Math.min(96, 55 + 0.45 * t + 8 * Math.sin(t / 8));
    return { p, temp, h };
}

// Simulate ~1 sample every 10 minutes for the last 100 h (matches mock.js)
const samples = [];
for (let hoursAgo = 100; hoursAgo >= 0; hoursAgo -= 10 / 60) {
    const s = baseState(hoursAgo);
    samples.push({
        t: NOW - hoursAgo * HOUR,
        p: s.p,
        temp: s.temp,
        h: s.h,
    });
}
console.log(`Mock samples: ${samples.length} over ${((samples.at(-1).t - samples[0].t) / HOUR).toFixed(1)} h`);

const window = buildWindow(samples, NOW, 32, 180);
if (!window) throw new Error("buildWindow returned null");
console.log(`Window: ${window.length} steps x 3 channels`);
console.log(`First step (oldest): P=${window[0][0].toFixed(1)}  T=${window[0][1].toFixed(1)}  HR=${window[0][2].toFixed(1)}`);
console.log(`Last step (newest):  P=${window.at(-1)[0].toFixed(1)}  T=${window.at(-1)[1].toFixed(1)}  HR=${window.at(-1)[2].toFixed(1)}`);

const features = buildFeatures3ch(window);
const proba = predictHGB(bundle, features);
const level = alertFromProba(bundle, proba);
console.log(`\nProba orage à H+24 h: ${(proba * 100).toFixed(2)} %`);
console.log(`Alerte: ${level}`);
console.log(`Seuils: ORANGE=${(bundle.orange_threshold * 100).toFixed(1)}%  ROUGE=${(bundle.rouge_threshold * 100).toFixed(1)}%`);

// Sanity: on a synoptic pattern with pressure falling 10 hPa and humidity
// rising 40 pts, the model should escalate away from VERT.
if (level === "VERT") {
    console.log("\nNote: le mock ne pousse pas suffisamment pour dépasser ORANGE. Ajuster mock.js si besoin.");
} else {
    console.log(`\nOK: le mock déclenche bien ${level} en fin de fenêtre.`);
}
