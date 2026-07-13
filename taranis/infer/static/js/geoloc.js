// Opt-in browser geolocation. Only invoked when the user explicitly taps
// "Localiser" in the settings drawer. We never call it automatically.

export function isGeolocationAvailable() {
    return typeof navigator !== "undefined"
        && typeof navigator.geolocation === "object";
}

export function requestGeolocation(options = {}) {
    return new Promise((resolve, reject) => {
        if (!isGeolocationAvailable()) {
            reject(new Error("Géolocalisation non supportée par ce navigateur."));
            return;
        }
        // Modern browsers require a secure context (HTTPS) for geolocation.
        // Localhost is the only HTTP exception. On plain http:// over an IP
        // the API silently returns permission-denied with no prompt.
        if (typeof window !== "undefined" && window.isSecureContext === false) {
            reject(new Error(
                "Géolocalisation bloquée : ouvrez cette page en HTTPS. "
                + "Sur http://, seul localhost est autorisé."
            ));
            return;
        }
        const timeoutMs = options.timeoutMs || 12000;
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({
                lat: pos.coords.latitude,
                lon: pos.coords.longitude,
                accuracy: pos.coords.accuracy,
                altitude: pos.coords.altitude,
            }),
            (err) => {
                if (err.code === 1) reject(new Error("Permission refusée dans le navigateur."));
                else if (err.code === 2) reject(new Error("Position indisponible."));
                else if (err.code === 3) reject(new Error("Délai dépassé."));
                else reject(new Error(err.message || "Erreur de géolocalisation."));
            },
            {
                enableHighAccuracy: true,
                timeout: timeoutMs,
                maximumAge: 300000,
            },
        );
    });
}

// Format lat/lon in a compact "43.48°N, 3.53°W" style.
export function formatLocation(lat, lon) {
    const latHem = lat >= 0 ? "N" : "S";
    const lonHem = lon >= 0 ? "E" : "W";
    return `${Math.abs(lat).toFixed(2)}°${latHem}, ${Math.abs(lon).toFixed(2)}°${lonHem}`;
}

// Reverse geocode via Nominatim (OpenStreetMap). Free, no key. Their
// usage policy asks apps to identify themselves via User-Agent, but
// browsers refuse to override User-Agent from fetch(), so we identify
// via the Referer that Chrome sends and stay under the 1 req/s ceiling
// by caching results in localStorage.
const NOMINATIM = "https://nominatim.openstreetmap.org/reverse";
const CACHE_KEY = "taranis.geocode.v1";

function cacheKey(lat, lon) {
    // Round to ~1 km grid so nearby ticks reuse the same lookup.
    return `${lat.toFixed(2)},${lon.toFixed(2)}`;
}
function loadCache() {
    try { return JSON.parse(localStorage.getItem(CACHE_KEY) || "{}"); }
    catch { return {}; }
}
function saveCache(cache) {
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(cache)); }
    catch { /* quota full, ignore */ }
}

export async function reverseGeocode(lat, lon, lang) {
    const key = cacheKey(lat, lon);
    const cache = loadCache();
    if (cache[key]) return cache[key];

    const langs = lang ? `${lang},en` : "en";
    const url = `${NOMINATIM}?format=jsonv2&lat=${lat}&lon=${lon}`
        + `&zoom=12&addressdetails=1&accept-language=${langs}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Nominatim ${r.status}`);
    const d = await r.json();
    const a = d.address || {};
    const name = a.city || a.town || a.village || a.municipality
        || a.hamlet || a.county
        || (d.display_name ? d.display_name.split(",")[0] : null);
    const country = a.country || null;
    const country_code = (a.country_code || "").toUpperCase() || null;
    const result = { name: name || null, country, country_code };
    cache[key] = result;
    saveCache(cache);
    return result;
}
