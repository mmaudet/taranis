// 24 h rolling buffer of (timestamp, pressure, temp, humidity) samples.
// Backed by IndexedDB so the buffer survives PWA restarts. The buffer is
// physically stored at arbitrary sensor cadence, and the app downsamples
// to a 3 h grid before running features + HGB.

const DB_NAME = "taranis";
const DB_VERSION = 1;
const STORE = "samples";
const KEEP_HOURS = 30;  // hold slightly more than 24 h for cross-day windows

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "t" });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

export async function pushSample({ t, p, temp, h }) {
  const db = await openDb();
  const tx = db.transaction(STORE, "readwrite");
  tx.objectStore(STORE).put({ t, p, temp, h });
  return new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
}

export async function trimBuffer(nowMs = Date.now()) {
  const db = await openDb();
  const cutoff = nowMs - KEEP_HOURS * 3600 * 1000;
  const tx = db.transaction(STORE, "readwrite");
  const st = tx.objectStore(STORE);
  const req = st.openCursor(IDBKeyRange.upperBound(cutoff));
  return new Promise((res, rej) => {
    req.onsuccess = () => {
      const c = req.result;
      if (c) { c.delete(); c.continue(); } else res();
    };
    req.onerror = () => rej(req.error);
  });
}

export async function readAll() {
  const db = await openDb();
  const tx = db.transaction(STORE, "readonly");
  return new Promise((res, rej) => {
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

export async function clearAll() {
  const db = await openDb();
  const tx = db.transaction(STORE, "readwrite");
  tx.objectStore(STORE).clear();
  return new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
}

// Build a (Tw x 3) window ending at nowMs by averaging over step_minutes bins.
// If a bin has no sample, we fall back to the previous bin's value; if no prior
// bin exists we return null (buffer not full enough for a prediction).
export function buildWindow(samples, nowMs, tw = 32, stepMinutes = 180) {
  const stepMs = stepMinutes * 60 * 1000;
  const bins = [];
  let anchor = nowMs;
  for (let i = 0; i < tw; i++) {
    const bEnd = anchor - i * stepMs;
    const bStart = bEnd - stepMs;
    const inBin = samples.filter(s => s.t >= bStart && s.t < bEnd);
    if (inBin.length === 0) { bins.push(null); continue; }
    const p = inBin.reduce((s, x) => s + x.p, 0) / inBin.length;
    const temp = inBin.reduce((s, x) => s + x.temp, 0) / inBin.length;
    const h = inBin.reduce((s, x) => s + x.h, 0) / inBin.length;
    bins.push([p, temp, h]);
  }
  // reverse so index 0 is oldest, index Tw-1 is newest
  bins.reverse();
  // First forward-fill from any first non-null bin. Then back-fill leading
  // nulls with the first non-null value that follows. This lets the sensor
  // produce a plausible window even when the buffer covers less than the
  // model's full lookback (Tw * step_minutes hours).
  const anyNonNull = bins.find(b => b !== null);
  if (!anyNonNull) return null;
  let lastGood = anyNonNull;
  const filled = bins.map(b => (b !== null ? (lastGood = b) : lastGood));
  return filled;
}
