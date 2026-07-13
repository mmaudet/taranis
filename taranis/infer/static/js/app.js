// Taranis PWA controller. Wires the Claude Design layout to the real
// prediction pipeline (features -> HGB -> level). Preserves data-taranis-*
// hooks from the design so future skin changes don't break the logic.

import { buildFeatures3ch } from "./features.js";
import { predictHGB, alertFromProba } from "./hgb_eval.js";
import { loadJepaSession, predictJepa, alertFromProbaJepa } from "./jepa_eval.js";
import { loadSettings, saveSettings } from "./settings.js";
import { readAll, buildWindow, clearAll, trimBuffer } from "./buffer.js";
import { seedMockBuffer, startMockLive, stopMockLive } from "./mock.js";
import { pairRuuvi, isWebBluetoothAvailable, isIOS } from "./ble.js";
import { requestGeolocation, formatLocation, reverseGeocode } from "./geoloc.js";
import { fetchOpenMeteoBackfill, fetchOpenMeteoContext } from "./openmeteo.js";
import { pushSample } from "./buffer.js";
import {
    $, ringOffsetForProba, glyphSvg,
    levelLabel, regimeLabel, heroNoteLabel, preavisNoteLabel,
    openDrawer, closeDrawer, toast, formatClock,
} from "./ui.js";
import { t, applyI18n, setLang, detectBrowserLang, AVAILABLE_LANGS } from "./i18n.js";

const STATE = {
    bundle: null,
    settings: loadSettings(),
    lastProba: 0,
    lastLevel: "VERT",
    lastPrediction: null,
    // Points arrays used by the crosshair. Each entry has { t, p } for
    // pressure charts; extended if a chart carries more channels.
    chartPts: { live: null, hist: null },
};

async function loadBundle(engine) {
    if (engine === "jepa") {
        const r = await fetch("./models/tsjepa_3ch.meta.json");
        if (!r.ok) throw new Error(t("error.model.jepa"));
        const bundle = await r.json();
        bundle.engine = "jepa";
        await loadJepaSession("./models/tsjepa_3ch.onnx");
        return bundle;
    }
    // Default HGB uses the 24h Tw=8 model. Better generalization + less
    // buffer required. See chapter 20 justification (Noja false alarm fix).
    const r = await fetch("./models/hgb_3ch_tw8.json");
    if (!r.ok) throw new Error(t("error.model.hgb"));
    const bundle = await r.json();
    bundle.engine = "hgb";
    return bundle;
}

async function scoreWindow(window) {
    if (STATE.bundle.engine === "jepa") {
        const proba = await predictJepa(STATE.bundle, window);
        const level = alertFromProbaJepa(STATE.bundle, proba);
        return { proba, level };
    }
    const features = buildFeatures3ch(window);
    const proba = predictHGB(STATE.bundle, features);
    const level = alertFromProba(STATE.bundle, proba);
    return { proba, level };
}

function applyLevel(level) {
    document.documentElement.setAttribute("data-level", level.toLowerCase() === "vert" ? "green"
        : level.toLowerCase() === "orange" ? "amber" : "red");
    STATE.lastLevel = level;
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
}

let _omTimer = null;

async function activateDataSource(source, showToast) {
    const tag = $("#sensor-tag");
    stopMockLive();
    stopOpenMeteoPoll();
    await clearAll();

    if (source === "mock") {
        tag.innerHTML = `<span class="dot"></span>Mock`;
        tag.style.color = "var(--amber)";
        await seedMockBuffer();
        startMockLive();
        if (showToast) toast(t("toast.mock_active"));
    } else if (source === "openmeteo") {
        tag.innerHTML = `<span class="dot"></span>Open-Meteo`;
        tag.style.color = "var(--thunder)";
        try {
            await seedOpenMeteoBuffer();
            startOpenMeteoPoll();
            if (showToast) toast(t("toast.openmeteo_active"));
        } catch (e) {
            toast(e.message || t("error.openmeteo"));
            $("#opt-source").value = "mock";
            STATE.settings = saveSettings({ dataSource: "mock" });
            tag.innerHTML = `<span class="dot"></span>Mock`;
            tag.style.color = "var(--amber)";
            await seedMockBuffer();
            startMockLive();
        }
    } else {
        tag.innerHTML = `<span class="dot"></span>BLE`;
        tag.style.color = "var(--dim)";
    }
}

async function seedOpenMeteoBuffer() {
    const loc = STATE.settings.location || { lat: 45.90, lon: 6.87 };
    const samples = await fetchOpenMeteoBackfill(loc.lat, loc.lon, 100);
    if (samples.length === 0) throw new Error("Open-Meteo n'a renvoyé aucune mesure");
    for (const s of samples) await pushSample(s);
    await trimBuffer(Date.now());
}

function startOpenMeteoPoll() {
    stopOpenMeteoPoll();
    // Poll every 30 minutes to fetch the newest hourly observation.
    _omTimer = setInterval(async () => {
        try {
            const loc = STATE.settings.location || { lat: 45.90, lon: 6.87 };
            const recent = await fetchOpenMeteoBackfill(loc.lat, loc.lon, 2);
            for (const s of recent) await pushSample(s);
            await trimBuffer(Date.now());
        } catch (e) {
            console.warn("Open-Meteo poll fail", e);
        }
    }, 30 * 60 * 1000);
}

function stopOpenMeteoPoll() {
    if (_omTimer) { clearInterval(_omTimer); _omTimer = null; }
}

async function refreshContextPanel() {
    const header = $("#context-header");
    const panel = $("#context-panel");
    if (!STATE.settings.meteoOnline || !navigator.onLine) {
        header.style.display = "none";
        panel.style.display = "none";
        return;
    }
    try {
        const loc = STATE.settings.location || { lat: 45.90, lon: 6.87 };
        const ctx = await fetchOpenMeteoContext(loc.lat, loc.lon);
        header.style.display = "flex";
        panel.style.display = "block";
        $("#ctx-wind").textContent = Math.round(ctx.windKmh);
        $("#ctx-gusts").textContent = Math.round(ctx.gustKmh);
        $("#ctx-precip").textContent = ctx.precipNext6h.toFixed(1);
        $("#ctx-pop").textContent = Math.round(ctx.popPeak6h);
        $("#ctx-cape").textContent = Math.round(ctx.capePeak6h);

        // Verdict badge: green if all indicators calm, amber otherwise.
        const risky = ctx.popPeak6h >= 40 || ctx.capePeak6h >= 500 || ctx.precipNext6h >= 2;
        const kEl = $("#ctx-verdict-k");
        const vEl = $("#ctx-verdict-v");
        if (risky) {
            kEl.textContent = t("context.storm_risk");
            vEl.textContent = "⚠";
            vEl.style.color = "var(--amber)";
        } else {
            kEl.textContent = t("context.no_storm");
            vEl.textContent = "✓";
            vEl.style.color = "var(--green)";
        }
    } catch (e) {
        header.style.display = "none";
        panel.style.display = "none";
    }
}

function updateLocationLabel() {
    const loc = STATE.settings.location || {};
    const hasGps = loc.lat && loc.lon && loc.label && loc.label.match(/\d/);
    if (hasGps) {
        const altPart = loc.altitude ? `${Math.round(loc.altitude)} m · ` : "";
        const detail = `${altPart}${loc.label}`;
        // Locality name via reverse geocoding when available, else the
        // neutral "GPS position" translation. Coordinates always shown
        // as the second line so the user can verify.
        $("#place-name").textContent = loc.name || t("home.gps_position");
        $("#place-detail").textContent = detail;
        $("#loc-help").textContent = loc.name ? `${loc.name} · ${detail}` : detail;
    } else {
        const label = (loc.label || t("home.default_place"));
        const lat = (loc.lat || 45.9).toFixed(2);
        const lon = (loc.lon || 6.87).toFixed(2);
        $("#loc-help").textContent = `${label}, ${lat}°N, ${lon}°E`;
        $("#place-name").textContent = label;
        $("#place-detail").textContent = `2340 m · ${lat}°N`;
    }
}

// --- Home screen renderers ---
function renderHome(proba, level) {
    const glyphKind = level === "VERT" ? "check" : level === "ORANGE" ? "tri" : "bolt";
    $("#vital-glyph").innerHTML = glyphSvg(glyphKind, 38);
    $("#vital-status").textContent = levelLabel(level);
    $("#vital-regime").textContent = regimeLabel(level);
    $("#home-updated").textContent = formatClock();
    const offset = ringOffsetForProba(proba, STATE.bundle.orange_threshold, STATE.bundle.rouge_threshold);
    $("#ring-fg").setAttribute("stroke-dashoffset", offset);

    const p = $("#preavis");
    $("#preavis-note").textContent = preavisNoteLabel(level);
    p.classList.toggle("calm", level === "VERT");
}

// --- Live screen renderers ---
function renderLive(samples) {
    $("#live-clock").textContent = formatClock();
    if (samples.length === 0) return;

    const last = samples[samples.length - 1];
    $("#press-val").textContent = last.p.toFixed(0);
    $("#temp-val").textContent = last.temp.toFixed(0);
    $("#hum-val").textContent = last.h.toFixed(0);

    // 3-h trend on pressure
    const now = last.t;
    const threeAgo = now - 3 * 3600 * 1000;
    const early = samples.find(s => s.t >= threeAgo);
    if (early) {
        const dp = last.p - early.p;
        const arrow = dp > 0.4 ? "↑" : dp < -0.4 ? "↓" : "→";
        $("#press-arrow").textContent = arrow;
        $("#press-trend").textContent = (dp > 0 ? "+" : "") + dp.toFixed(1);
    }

    // 1-h temp trend
    const oneAgo = now - 3600 * 1000;
    const tempEarly = samples.find(s => s.t >= oneAgo);
    if (tempEarly) {
        const dt = last.temp - tempEarly.temp;
        $("#temp-sub").textContent = `${dt > 0 ? "↑" : dt < 0 ? "↓" : "→"} ${Math.abs(dt).toFixed(1)} °C · 1h`;
        const dh = last.h - tempEarly.h;
        $("#hum-sub").textContent = `${dh > 0 ? "↑" : dh < 0 ? "↓" : "→"} ${Math.abs(dh).toFixed(0)} % · 1h`;
    }

    // Hero pressure sparkline: last 6 h, ~48 downsampled points
    const sixAgo = now - 6 * 3600 * 1000;
    const recent = samples.filter(s => s.t >= sixAgo);
    if (recent.length >= 4) {
        drawHeroChart(recent);
        const first = recent[0].t;
        const mid = recent[Math.floor(recent.length / 2)].t;
        const end = recent[recent.length - 1].t;
        $("#live-tick-0").textContent = formatClock(new Date(first));
        $("#live-tick-1").textContent = formatClock(new Date(mid));
        $("#live-tick-2").textContent = formatClock(new Date(end));
    }

    $("#hero-note").textContent = heroNoteLabel(STATE.lastLevel);
}

function drawHeroChart(samples) {
    const N = samples.length;
    const step = Math.max(1, Math.floor(N / 60));
    const pts = [];
    for (let i = 0; i < N; i += step) pts.push(samples[i]);
    if (pts[pts.length - 1].t !== samples[N - 1].t) pts.push(samples[N - 1]);

    const vals = pts.map(s => s.p);
    const mn = Math.min(...vals) - 0.5;
    const mx = Math.max(...vals) + 0.5;
    const range = Math.max(0.5, mx - mn);
    const W = 320, H = 118;
    const coord = pts.map((s, i) => {
        const x = (i / (pts.length - 1)) * W;
        const y = H - ((s.p - mn) / range) * (H - 14) - 4;
        return [+x.toFixed(1), +y.toFixed(1)];
    });
    const line = coord.map(p => p.join(",")).join(" ");
    const area = `0,${H} ${line} ${W},${H}`;
    const last = coord[coord.length - 1];

    $("#press-line").setAttribute("points", line);
    $("#press-area").setAttribute("points", area);
    $("#press-dot").setAttribute("cx", last[0]);
    $("#press-dot").setAttribute("cy", last[1]);
    $("#press-dot-pulse").setAttribute("cx", last[0]);
    $("#press-dot-pulse").setAttribute("cy", last[1]);

    STATE.chartPts.live = pts.map((s, i) => ({ t: s.t, p: s.p, x: coord[i][0], y: coord[i][1] }));
}

// --- History screen renderers ---
function renderHistory(samples) {
    if (samples.length < 4) return;
    const now = samples[samples.length - 1].t;
    const sixAgo = now - 6 * 3600 * 1000;
    const recent = samples.filter(s => s.t >= sixAgo);
    if (recent.length < 4) return;

    const step = Math.max(1, Math.floor(recent.length / 40));
    const pts = [];
    for (let i = 0; i < recent.length; i += step) pts.push(recent[i]);
    if (pts[pts.length - 1].t !== recent[recent.length - 1].t) pts.push(recent[recent.length - 1]);
    const vals = pts.map(s => s.p);
    const mn = Math.min(...vals) - 1.0;
    const mx = Math.max(...vals) + 1.0;
    const range = Math.max(1.0, mx - mn);
    const W = 320, H = 130;
    const coord = pts.map((s, i) => {
        const x = (i / (pts.length - 1)) * W;
        const y = H - ((s.p - mn) / range) * (H - 20) - 6;
        return [+x.toFixed(1), +y.toFixed(1)];
    });

    // Color each segment by local pressure rate (hPa/h). Falling pressure
    // is the physically meaningful storm precursor, so we colour on that,
    // not on time position.
    const segments = [];
    for (let i = 0; i < pts.length - 1; i++) {
        const dtH = (pts[i + 1].t - pts[i].t) / 3600000;
        const dp = pts[i + 1].p - pts[i].p;
        const rate = dtH > 0 ? dp / dtH : 0;
        let color;
        if (rate <= -1.5) color = "var(--red)";
        else if (rate <= -0.3) color = "var(--amber)";
        else color = "var(--green)";
        segments.push(
            `<line x1="${coord[i][0]}" y1="${coord[i][1]}"`
            + ` x2="${coord[i + 1][0]}" y2="${coord[i + 1][1]}"`
            + ` stroke="${color}" stroke-width="2.5"`
            + ` stroke-linecap="round" stroke-linejoin="round"/>`,
        );
    }
    $("#hist-segments").innerHTML = segments.join("");

    STATE.chartPts.hist = pts.map((s, i) => ({ t: s.t, p: s.p, x: coord[i][0], y: coord[i][1] }));

    const tstart = new Date(recent[0].t);
    const tmid = new Date(recent[Math.floor(recent.length / 2)].t);
    const tend = new Date(recent[recent.length - 1].t);
    $("#hist-tick-0").textContent = formatClock(tstart);
    $("#hist-tick-1").textContent = formatClock(tmid);
    $("#hist-tick-2").textContent = formatClock(tend);

    renderEvents(recent);
}

function renderEvents(samples) {
    // Very simple event detector: look for 30-min windows where pressure drops
    // faster than 1 hPa/h. Also emit "pressure stable" if no such window.
    const events = [];
    const seen = new Set();
    let i = 0;
    while (i < samples.length) {
        const from = samples[i];
        const target = from.t + 30 * 60 * 1000;
        let j = i + 1;
        while (j < samples.length && samples[j].t < target) j++;
        if (j >= samples.length) break;
        const to = samples[j];
        const dtH = (to.t - from.t) / (3600 * 1000);
        const dp = to.p - from.p;
        const rate = dp / dtH;
        if (rate < -1.5) {
            const bucket = `red-${to.t}`;
            if (!seen.has(bucket)) events.push({
                time: formatClock(new Date(to.t)),
                label: t("events.crash"),
                val: `${rate.toFixed(1)} hPa/h`,
                color: "var(--red)",
            });
            seen.add(bucket);
        } else if (rate < -0.5) {
            const bucket = `amber-${to.t}`;
            if (!seen.has(bucket)) events.push({
                time: formatClock(new Date(to.t)),
                label: t("events.drop"),
                val: `${rate.toFixed(1)} hPa/h`,
                color: "var(--amber)",
            });
            seen.add(bucket);
        }
        i = j;
    }
    if (events.length === 0) {
        events.push({
            time: formatClock(new Date(samples[0].t)),
            label: t("events.stable"),
            val: `${samples[0].p.toFixed(0)} hPa`,
            color: "var(--green)",
        });
    }
    const html = events.slice(-6).map(ev => `
      <div class="event">
        <span class="time">${ev.time}</span>
        <span class="dot" style="background:${ev.color};box-shadow:0 0 8px ${ev.color}"></span>
        <span class="label">${ev.label}</span>
        <span class="val">${ev.val}</span>
      </div>
    `).join("");
    $("#events").innerHTML = html;
}

// --- Alert overlay ---
function maybeShowAlert(proba, samples) {
    const overlay = $("#alert-overlay");
    if (STATE.lastLevel !== "ROUGE") {
        overlay.classList.remove("open");
        return;
    }
    // Only auto-open once per level transition
    if (overlay.dataset.opened === "true") return;
    overlay.dataset.opened = "true";
    overlay.classList.add("open");

    const now = samples[samples.length - 1].t;
    const oneAgo = now - 3600 * 1000;
    const early = samples.find(s => s.t >= oneAgo);
    if (early) {
        const rate = (samples[samples.length - 1].p - early.p);
        $("#alert-drop").textContent = rate.toFixed(1);
    }
    $("#alert-eta").textContent = "~12";
}

function resetAlertLatch() {
    const overlay = $("#alert-overlay");
    overlay.dataset.opened = "false";
    overlay.classList.remove("open");
}

// --- Prediction cycle ---
async function tick() {
    const samples = await readAll();
    if (samples.length < 4) return;
    samples.sort((a, b) => a.t - b.t);

    const tw = STATE.bundle.tw || 32;
    const step = STATE.bundle.step_minutes || 180;
    const win = buildWindow(samples, Date.now(), tw, step);
    if (!win) return;

    const { proba, level } = await scoreWindow(win);

    STATE.lastProba = proba;
    if (level !== STATE.lastLevel) resetAlertLatch();
    applyLevel(level);
    STATE.lastPrediction = { proba, level, at: Date.now() };

    renderHome(proba, level);
    renderLive(samples);
    renderHistory(samples);
    renderSensorGroupHeader();
    maybeShowAlert(proba, samples);
}

function installCrosshair(key, wrapId, overlayId, tipId, lineId, dotId, vbHeight) {
    const wrap = document.getElementById(wrapId);
    const overlay = document.getElementById(overlayId);
    const tip = document.getElementById(tipId);
    const line = document.getElementById(lineId);
    const dot = document.getElementById(dotId);
    if (!wrap || !overlay || !tip || !line || !dot) return;

    function place(clientX) {
        const rect = overlay.getBoundingClientRect();
        const xClient = clientX - rect.left;
        if (rect.width <= 0) return;
        const clampedX = Math.max(0, Math.min(rect.width, xClient));
        const pts = STATE.chartPts[key];
        if (!pts || pts.length === 0) return hide();
        const vbX = (clampedX / rect.width) * 320;
        let best = pts[0];
        let bestDist = Math.abs(pts[0].x - vbX);
        for (let i = 1; i < pts.length; i++) {
            const d = Math.abs(pts[i].x - vbX);
            if (d < bestDist) { bestDist = d; best = pts[i]; }
        }
        line.setAttribute("x1", best.x);
        line.setAttribute("x2", best.x);
        line.setAttribute("y2", vbHeight);
        dot.setAttribute("cx", best.x);
        dot.setAttribute("cy", best.y);
        const time = formatClock(new Date(best.t));
        tip.textContent = `${time} · ${best.p.toFixed(1)} hPa`;
        // Clamp tip to stay inside the overlay horizontally.
        const bestClient = (best.x / 320) * rect.width;
        const half = tip.offsetWidth / 2 || 60;
        const leftPx = Math.max(half, Math.min(rect.width - half, bestClient));
        tip.style.left = leftPx + "px";
        wrap.classList.add("active");
    }

    function hide() { wrap.classList.remove("active"); }

    let active = false;
    function down(e) {
        active = true;
        try { overlay.setPointerCapture(e.pointerId); } catch (_) { /* not fatal */ }
        place(e.clientX);
        e.preventDefault();
    }
    function move(e) {
        if (!active) return;
        place(e.clientX);
        e.preventDefault();
    }
    function up(e) {
        active = false;
        try { overlay.releasePointerCapture(e.pointerId); } catch (_) {}
        hide();
    }
    overlay.addEventListener("pointerdown", down);
    overlay.addEventListener("pointermove", move);
    overlay.addEventListener("pointerup", up);
    overlay.addEventListener("pointercancel", up);
    overlay.addEventListener("pointerleave", up);
    // Touch fallback for browsers without full Pointer Events on SVG.
    overlay.addEventListener("touchstart", (e) => {
        if (e.touches.length === 0) return;
        active = true;
        place(e.touches[0].clientX);
        e.preventDefault();
    }, { passive: false });
    overlay.addEventListener("touchmove", (e) => {
        if (!active || e.touches.length === 0) return;
        place(e.touches[0].clientX);
        e.preventDefault();
    }, { passive: false });
    overlay.addEventListener("touchend", () => { active = false; hide(); });
    overlay.addEventListener("touchcancel", () => { active = false; hide(); });
}

function renderSensorGroupHeader() {
    const sub = $("#sensor-group-sub");
    const src = STATE.settings.dataSource;
    if (src === "sensor") sub.textContent = t("sensor_group.hint");
    else if (src === "openmeteo") sub.textContent = t("context.subtitle");
    else if (src === "mock") sub.textContent = "Mock";
    else sub.textContent = t("sensor_group.offline");
}

// --- Screen router ---
function showScreen(name) {
    document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
    document.querySelectorAll(".nav button").forEach(b => b.classList.remove("active"));
    document.getElementById(`screen-${name}`).classList.add("active");
    document.getElementById(`nav-${name}`).classList.add("active");
}

// --- Setup drawer ---
async function setupDrawer() {
    const s = STATE.settings;

    const langSel = $("#opt-lang");
    langSel.value = s.language || detectBrowserLang();
    langSel.addEventListener("change", () => {
        const lang = langSel.value;
        setLang(lang);
        applyI18n();
        STATE.settings = saveSettings({ language: lang });
        // Redraw dynamic labels immediately (level status, regime text, etc.)
        if (STATE.lastPrediction) {
            renderHome(STATE.lastPrediction.proba, STATE.lastPrediction.level);
            $("#hero-note").textContent = heroNoteLabel(STATE.lastLevel);
        }
    });

    const engSel = $("#opt-engine");
    engSel.value = s.engine;
    engSel.addEventListener("change", async () => {
        const next = engSel.value;
        try {
            STATE.bundle = await loadBundle(next);
            STATE.settings = saveSettings({ engine: next });
            toast(`${t("toast.engine_switched")}: ${next.toUpperCase()}`);
            tick();
        } catch (e) {
            toast(e.message);
            engSel.value = STATE.settings.engine;
        }
    });

    const themeSel = $("#opt-theme");
    themeSel.value = s.theme || "night";
    themeSel.addEventListener("change", () => {
        applyTheme(themeSel.value);
        STATE.settings = saveSettings({ theme: themeSel.value });
    });

    const sourceSel = $("#opt-source");
    sourceSel.value = s.dataSource || "mock";
    sourceSel.addEventListener("change", async () => {
        const src = sourceSel.value;
        STATE.settings = saveSettings({ dataSource: src });
        await activateDataSource(src, true);
        tick();
    });

    const meteoSw = $("#opt-meteo");
    meteoSw.checked = s.meteoOnline;
    meteoSw.addEventListener("change", () => {
        STATE.settings = saveSettings({ meteoOnline: meteoSw.checked });
        toast(meteoSw.checked ? t("toast.meteo_on") : t("toast.meteo_off"));
        refreshContextPanel();
    });

    $("#opt-clear").addEventListener("click", async () => {
        stopMockLive();
        await clearAll();
        toast(t("toast.buffer_cleared"));
        applyLevel("VERT");
        tick();
    });

    $("#opt-reload").addEventListener("click", async () => {
        toast(t("toast.purging"));
        try {
            if ("caches" in window) {
                const keys = await caches.keys();
                await Promise.all(keys.map(k => caches.delete(k)));
            }
            if ("serviceWorker" in navigator) {
                const regs = await navigator.serviceWorker.getRegistrations();
                await Promise.all(regs.map(r => r.unregister()));
            }
        } catch (e) { /* ignore */ }
        window.location.reload();
    });

    // Location: default label + optional user geolocation
    updateLocationLabel();
    $("#opt-geoloc").addEventListener("click", async () => {
        const btn = $("#opt-geoloc");
        btn.textContent = "…";
        btn.disabled = true;
        try {
            const pos = await requestGeolocation();
            const label = formatLocation(pos.lat, pos.lon);
            let name = null;
            try {
                const rev = await reverseGeocode(pos.lat, pos.lon, STATE.settings.language || "en");
                name = rev.name;
            } catch (_) { /* fallback to raw coordinates */ }
            STATE.settings = saveSettings({
                location: {
                    lat: pos.lat,
                    lon: pos.lon,
                    label: label,
                    name: name,
                    altitude: pos.altitude || null,
                },
            });
            updateLocationLabel();
            toast(`${t("toast.position_ok")} (±${Math.round(pos.accuracy)} m)`);
        } catch (e) {
            toast(e.message);
        } finally {
            btn.textContent = t("settings.location.locate");
            btn.disabled = false;
        }
    });

    $("#cog").addEventListener("click", openDrawer);
    $("#drawer-close").addEventListener("click", closeDrawer);
    $("#drawer-backdrop").addEventListener("click", closeDrawer);
}

// --- Boot ---
async function boot() {
    // i18n first so the drawer defaults draw correctly
    const chosenLang = STATE.settings.language
        || (AVAILABLE_LANGS.includes(detectBrowserLang()) ? detectBrowserLang() : "en");
    setLang(chosenLang);
    if (!STATE.settings.language) {
        STATE.settings = saveSettings({ language: chosenLang });
    }
    applyI18n();

    applyTheme(STATE.settings.theme || "night");
    await setupDrawer();

    try {
        STATE.bundle = await loadBundle(STATE.settings.engine);
    } catch (e) {
        toast(e.message);
        return;
    }

    if (isIOS() && !isWebBluetoothAvailable()) {
        const el = document.createElement("div");
        el.className = "notice";
        el.textContent = t("error.ble.ios");
        document.getElementById("ios-notice").replaceWith(el);
    }

    await activateDataSource(STATE.settings.dataSource || "mock", false);

    // Nav wiring
    document.querySelectorAll(".nav button").forEach(b => {
        b.addEventListener("click", () => showScreen(b.dataset.screen));
    });

    // Alert ack
    $("#alert-ack").addEventListener("click", () => {
        $("#alert-overlay").classList.remove("open");
    });

    // Pairing
    $("#pair-btn").addEventListener("click", async () => {
        try {
            await pairRuuvi();
            toast(t("toast.pair_deferred"));
            $("#pair-icon").classList.add("connected");
            $("#pair-headline").textContent = t("sensor.connected");
            $("#pair-headline-sub").textContent = t("sensor.connected_sub");
            $("#pair-stats").style.display = "flex";
        } catch (e) {
            toast(e.message || t("toast.pair_denied"));
        }
    });

    // Clock ticker
    setInterval(() => {
        $("#clock").textContent = formatClock();
    }, 5000);
    $("#clock").textContent = formatClock();

    // Interactive crosshair on both charts.
    installCrosshair("live", "live-chart-wrap", "live-chart-overlay", "live-crosshair-tip",
                     "live-crosshair-line", "live-crosshair-dot", 118);
    installCrosshair("hist", "hist-chart-wrap", "hist-chart-overlay", "hist-crosshair-tip",
                     "hist-crosshair-line", "hist-crosshair-dot", 130);

    // Version indicator: fetch the served sw.js CACHE_VERSION and show it
    // in the settings drawer so the user can visually confirm at a glance
    // whether the phone is on the freshly deployed code.
    (async () => {
        try {
            const r = await fetch("./sw.js", { cache: "no-cache" });
            const text = await r.text();
            const m = text.match(/CACHE_VERSION\s*=\s*"([^"]+)"/);
            if (m) $("#app-version").textContent = m[1];
        } catch (_) { /* silent */ }
    })();

    // Tick loop
    tick();
    setInterval(tick, 15 * 1000);

    // Open-Meteo context panel refresh loop (every 15 min).
    refreshContextPanel();
    setInterval(refreshContextPanel, 15 * 60 * 1000);
    window.addEventListener("online", refreshContextPanel);
    window.addEventListener("offline", refreshContextPanel);
}

if ("serviceWorker" in navigator) {
    let _swReloaded = false;
    navigator.serviceWorker.register("./sw.js").catch(() => {});
    // When a new service worker activates (after we bumped CACHE_VERSION),
    // the browser transfers page control to it. Reload once so the fresh
    // JS + models are used instead of the stale in-memory copies.
    navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (_swReloaded) return;
        _swReloaded = true;
        window.location.reload();
    });
}

document.addEventListener("DOMContentLoaded", boot);
