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

// Approximate a place name from lat/lon.  Reverse geocoding without API
// key is fragile: we fall back to a friendly "45.9°N, 6.87°E" label.
export function formatLocation(lat, lon) {
    const latHem = lat >= 0 ? "N" : "S";
    const lonHem = lon >= 0 ? "E" : "W";
    return `${Math.abs(lat).toFixed(2)}°${latHem}, ${Math.abs(lon).toFixed(2)}°${lonHem}`;
}
