// Open-Meteo integration.  Two roles:
//   1. Context panel: fetch current + short forecast for the settings drawer
//      "meteo online" toggle.
//   2. Synthetic sensor: seed the buffer with real hourly observations
//      when the user picks "Open-Meteo" as data source (demo mode until
//      the RuuviTag arrives).
//
// The API is public, no auth, no key.  Rate limit is 10 000 calls/day
// for non-commercial use, we call at most every 30 minutes.

const OM_URL = "https://api.open-meteo.com/v1/forecast";

// Fetch past + present hourly data suitable for filling the 96 h buffer.
export async function fetchOpenMeteoBackfill(lat, lon, pastHours = 100) {
    const url = `${OM_URL}?latitude=${lat}&longitude=${lon}`
        + `&hourly=surface_pressure,temperature_2m,relative_humidity_2m`
        + `&past_hours=${pastHours}&forecast_hours=1&timezone=UTC`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Open-Meteo indisponible (${r.status})`);
    const data = await r.json();

    const times = data.hourly.time;
    const p = data.hourly.surface_pressure;
    const t = data.hourly.temperature_2m;
    const h = data.hourly.relative_humidity_2m;

    const samples = [];
    for (let i = 0; i < times.length; i++) {
        if (p[i] == null || t[i] == null || h[i] == null) continue;
        samples.push({
            t: new Date(times[i] + "Z").getTime(),
            p: p[i],
            temp: t[i],
            h: h[i],
        });
    }
    return samples;
}

// Compact snapshot for the context panel (current values + short forecast).
// Returns physical values for the sensor-comparable channels plus the
// weather-station-only channels (wind, gusts, precipitation forecast,
// storm indicators).
export async function fetchOpenMeteoContext(lat, lon) {
    const url = `${OM_URL}?latitude=${lat}&longitude=${lon}`
        + `&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_gusts_10m,precipitation,wind_direction_10m`
        + `&hourly=precipitation_probability,precipitation,cape,wind_speed_10m,wind_gusts_10m`
        + `&forecast_hours=12&timezone=UTC`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Open-Meteo context indisponible (${r.status})`);
    const data = await r.json();
    const cur = data.current;
    const hourly = data.hourly;

    // Cumulative rain over next 6 h.
    const precipNext6h = (hourly.precipitation || []).slice(0, 6).reduce((s, x) => s + (x || 0), 0);
    const popPeak6h = Math.max(0, ...(hourly.precipitation_probability || []).slice(0, 6).filter(x => x != null));
    const capePeak6h = Math.max(0, ...(hourly.cape || []).slice(0, 6).filter(x => x != null));
    const gustPeak6h = Math.max(0, ...(hourly.wind_gusts_10m || []).slice(0, 6).filter(x => x != null));

    return {
        temp: cur.temperature_2m,
        humidity: cur.relative_humidity_2m,
        pressure: cur.surface_pressure,
        windKmh: cur.wind_speed_10m,
        windDegrees: cur.wind_direction_10m,
        gustKmh: cur.wind_gusts_10m,
        precipMm: cur.precipitation,
        precipNext6h,
        popPeak6h,
        capePeak6h,
        gustPeak6h,
        model: "AROME/IFS via Open-Meteo",
    };
}
