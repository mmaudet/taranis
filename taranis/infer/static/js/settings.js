// Persisted user preferences, backed by localStorage.
// Keys used elsewhere in the app must match exactly.

const KEY = "taranis.settings.v1";

const DEFAULTS = {
  engine: "hgb",              // 'hgb' or 'jepa'
  theme: "night",             // 'night' or 'day'
  dataSource: "openmeteo",    // default: real data at user's location.
                              // Falls back to mock only if network is off
                              // or geolocation was declined.
  meteoOnline: true,          // Open-Meteo context panel visible by default
  locationRefreshMin: 10,     // Poll GPS every N minutes; 0 = manual only.
                              // 10 min matches the "hiking" use case where
                              // the hiker moves a few km/h and wants the
                              // context (region name + Open-Meteo backfill)
                              // to follow. 300 m threshold before we act.
  location: {
    lat: 45.90,               // Chamonix default until geolocation runs
    lon: 6.87,
    label: "Chamonix",
  },
};

// Back-compat: v1 stored `mockMode` boolean; translate on load.
function migrate(obj) {
  if (obj.mockMode !== undefined && obj.dataSource === undefined) {
    obj.dataSource = obj.mockMode ? "mock" : "sensor";
    delete obj.mockMode;
  }
  // Migration to v6: if user had "mock" from old default, and geolocation
  // was already granted, upgrade to openmeteo silently.
  if (obj.dataSource === "mock" && obj.location && obj.location.label
      && obj.location.label.match(/\d/)) {
    obj.dataSource = "openmeteo";
  }
  return obj;
}

export function loadSettings() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = migrate(JSON.parse(raw));
    return { ...DEFAULTS, ...parsed, location: { ...DEFAULTS.location, ...(parsed.location || {}) } };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(patch) {
  const cur = loadSettings();
  const next = { ...cur, ...patch };
  if (patch.location) next.location = { ...cur.location, ...patch.location };
  localStorage.setItem(KEY, JSON.stringify(next));
  return next;
}
