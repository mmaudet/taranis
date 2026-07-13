// DOM helpers + inline SVG glyphs + level/regime text via i18n.

import { t } from "./i18n.js";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export const RING_C = 666;

export function ringOffsetForProba(proba, orangeThr, redThr) {
    let pct;
    if (proba < orangeThr) {
        const s = Math.min(1, Math.max(0, proba / Math.max(1e-6, orangeThr)));
        pct = 0.95 - 0.33 * s;
    } else if (proba < redThr) {
        const s = (proba - orangeThr) / Math.max(1e-6, redThr - orangeThr);
        pct = 0.62 - 0.30 * s;
    } else {
        const s = Math.min(1, (proba - redThr) / Math.max(1e-6, 1 - redThr));
        pct = 0.32 - 0.22 * s;
    }
    return RING_C - RING_C * Math.max(0.05, pct);
}

export function glyphSvg(kind, size = 38) {
    if (kind === "check") {
        return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
                <path d="M20 6 L9 17 L4 12"/></svg>`;
    }
    if (kind === "tri") {
        return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 3 L22 20 L2 20 Z"/><path d="M12 10 L12 14"/>
                <circle cx="12" cy="17" r="0.4" fill="currentColor" stroke="currentColor"/></svg>`;
    }
    if (kind === "bolt") {
        return `<svg width="${size}" height="${size}" viewBox="0 0 24 24"
                fill="currentColor" stroke="none">
                <path d="M13 2 L5 13 L11 13 L9 22 L19 10 L13 10 Z"/></svg>`;
    }
    return "";
}

export function levelLabel(level) {
    return level === "VERT" ? t("home.status.calm")
         : level === "ORANGE" ? t("home.status.caution")
         : t("home.status.danger");
}

export function regimeLabel(level) {
    if (level === "VERT") return t("home.regime.calm");
    if (level === "ORANGE") return t("home.regime.caution");
    return t("home.regime.danger");
}

export function heroNoteLabel(level) {
    if (level === "VERT") return t("live.hero.calm");
    if (level === "ORANGE") return t("live.hero.caution");
    return t("live.hero.danger");
}

export function preavisNoteLabel(level) {
    if (level === "VERT") return t("home.preavis.calm");
    if (level === "ORANGE") return t("home.preavis.caution");
    return t("home.preavis.danger");
}

export function openDrawer() {
    $("#drawer").classList.add("open");
    $("#drawer-backdrop").classList.add("open");
}
export function closeDrawer() {
    $("#drawer").classList.remove("open");
    $("#drawer-backdrop").classList.remove("open");
}

export function toast(message, ms = 2500) {
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), ms);
}

export function formatClock(date = new Date()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
