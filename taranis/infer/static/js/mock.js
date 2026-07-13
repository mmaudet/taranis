// Mock data generator for development before the RuuviTag arrives.
// Produces a plausible P/T/HR trace over 30 h with a slow synoptic
// pattern (falling pressure + rising humidity) that will trigger
// an ORANGE alert once the buffer has enough coverage.

import { pushSample, trimBuffer, clearAll } from "./buffer.js";

const HOUR = 3600 * 1000;

// Model expects Tw=32 x step=3h = 96 h of lookback. Mock covers 100 h to
// leave a comfortable margin. In real sensor usage the buffer will grow
// over time; buildWindow gracefully back-fills leading empty bins so the
// prediction remains meaningful even with a short recent history.
function baseState(hoursAgo) {
  const t = 100 - hoursAgo;
  const p = 1013 - 0.20 * t + 1.5 * Math.sin(t / 6);            // pressure drops 20 hPa over 100h
  const temp = 18 + 4 * Math.sin((t + 6) / 24 * 2 * Math.PI);    // day/night 14-22
  const h = Math.min(96, 55 + 0.45 * t + 8 * Math.sin(t / 8));   // humidity climbs 55 to 96
  return { p, temp, h };
}

// Populate the buffer with ~1 sample every 10 minutes for the last 100 h.
export async function seedMockBuffer() {
  const now = Date.now();
  await clearAll();
  for (let hoursAgo = 100; hoursAgo >= 0; hoursAgo -= 10 / 60) {
    const t = now - hoursAgo * HOUR;
    const s = baseState(hoursAgo);
    const jitter = () => (Math.random() - 0.5) * 0.4;
    await pushSample({
      t,
      p: s.p + jitter(),
      temp: s.temp + jitter(),
      h: s.h + jitter() * 4,
    });
  }
  await trimBuffer(now);
}

// One live tick simulating a RuuviTag broadcast every ~5 s.
let liveTimer = null;
export function startMockLive(onSample) {
  stopMockLive();
  let hoursAgo = -0.01;
  liveTimer = setInterval(() => {
    hoursAgo -= 5 / 3600;
    const s = baseState(hoursAgo);
    const jitter = () => (Math.random() - 0.5) * 0.4;
    const sample = {
      t: Date.now(),
      p: s.p + jitter(),
      temp: s.temp + jitter(),
      h: s.h + jitter() * 4,
    };
    pushSample(sample);
    if (onSample) onSample(sample);
  }, 5000);
}

export function stopMockLive() {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
}
