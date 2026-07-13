// Reproduce the "VERT for a second then ORANGE" bug end-to-end.
//
// Simulates the exact browser boot path:
// 1. Load HGB-Tw8 bundle (what the PWA now uses by default)
// 2. Load HGB-Tw32 bundle (what previous SW cache might still serve)
// 3. Load TS-JEPA bundle metadata (engine option that user might have)
// 4. Fetch real Open-Meteo backfill for Noja
// 5. Build windows with each bundle's tw + predict + report level
// 6. Also mix in mock samples to test the "stale buffer" hypothesis

import { readFile } from "node:fs/promises";
import { buildWindow } from "../taranis/infer/static/js/buffer.js";
import { buildFeatures3ch } from "../taranis/infer/static/js/features.js";
import { predictHGB, alertFromProba } from "../taranis/infer/static/js/hgb_eval.js";
import { fetchOpenMeteoBackfill } from "../taranis/infer/static/js/openmeteo.js";

const NOJA = { lat: 43.48, lon: -3.53 };
const NOW = Date.now();
const HOUR = 3600 * 1000;

async function loadJson(path) {
    const url = new URL(path, import.meta.url);
    return JSON.parse(await readFile(url, "utf8"));
}

function mockSample(hoursAgo) {
    // Same math as mock.js
    const t = 100 - hoursAgo;
    const p = 1013 - 0.20 * t + 1.5 * Math.sin(t / 6);
    const temp = 18 + 4 * Math.sin(((t + 6) / 24) * 2 * Math.PI);
    const h = Math.min(96, 55 + 0.45 * t + 8 * Math.sin(t / 8));
    return { t: NOW - hoursAgo * HOUR, p, temp, h };
}

function seedMock(nMinutesStep = 10) {
    const samples = [];
    for (let ha = 100; ha >= 0; ha -= nMinutesStep / 60) {
        samples.push(mockSample(ha));
    }
    return samples;
}

function runScenario(name, samples, bundle) {
    samples.sort((a, b) => a.t - b.t);
    const tw = bundle.tw;
    const step = bundle.step_minutes;
    const win = buildWindow(samples, NOW, tw, step);
    if (!win) {
        console.log(`${name}: window null`);
        return null;
    }
    const features = buildFeatures3ch(win);
    const proba = predictHGB(bundle, features);
    const level = alertFromProba(bundle, proba);
    const winFirst = win[0].map(v => v.toFixed(1)).join(",");
    const winLast = win[win.length - 1].map(v => v.toFixed(1)).join(",");
    console.log(
        `${name.padEnd(50)}`
        + ` samples=${String(samples.length).padStart(3)}`
        + ` tw=${tw}`
        + ` window[0]=${winFirst.padStart(20)}`
        + ` window[-1]=${winLast.padStart(20)}`
        + ` proba=${(proba * 100).toFixed(2).padStart(6)}%`
        + ` ${level}`,
    );
    return { proba, level };
}

const hgbTw8 = await loadJson("../taranis/infer/static/models/hgb_3ch_tw8.json");
const hgbTw32 = await loadJson("../taranis/infer/static/models/hgb_3ch.json");

console.log("\n=== Fetching real Noja data via Open-Meteo ===");
const omSamples = await fetchOpenMeteoBackfill(NOJA.lat, NOJA.lon, 100);
console.log(`Got ${omSamples.length} hourly samples over ${((omSamples.at(-1).t - omSamples[0].t) / HOUR).toFixed(1)} h`);

console.log("\n=== Scenario A: Open-Meteo alone, correct model (Tw=8) ===");
runScenario("Open-Meteo only, HGB-Tw8", [...omSamples], hgbTw8);

console.log("\n=== Scenario B: Open-Meteo alone, stale Tw=32 model ===");
runScenario("Open-Meteo only, HGB-Tw32", [...omSamples], hgbTw32);

console.log("\n=== Scenario C: buffer has BOTH mock (from prior session) AND Open-Meteo ===");
const mixed = [...seedMock(), ...omSamples];
runScenario("Mock+Open-Meteo mixed, HGB-Tw8", [...mixed], hgbTw8);
runScenario("Mock+Open-Meteo mixed, HGB-Tw32", [...mixed], hgbTw32);

console.log("\n=== Scenario D: mock buffer alone (previous session leftover) ===");
runScenario("Mock only, HGB-Tw8", seedMock(), hgbTw8);
runScenario("Mock only, HGB-Tw32", seedMock(), hgbTw32);
