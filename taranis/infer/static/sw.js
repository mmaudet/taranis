// Taranis service worker: aggressive offline caching for the shell + model.
// Cache-first strategy so the PWA works with zero connectivity once installed.
// Open-Meteo requests are never cached (opt-in live weather only).

const CACHE_VERSION = "taranis-20260713-195648";
const SHELL = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./css/app.css",
  "./js/app.js",
  "./js/features.js",
  "./js/hgb_eval.js",
  "./js/jepa_eval.js",
  "./js/buffer.js",
  "./js/mock.js",
  "./js/settings.js",
  "./js/ui.js",
  "./js/ble.js",
  "./js/geoloc.js",
  "./js/openmeteo.js",
  "./js/i18n.js",
  "./models/hgb_3ch.json",
  "./models/hgb_3ch_tw8.json",
  "./models/tsjepa_3ch.onnx",
  "./models/tsjepa_3ch.meta.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  // Never intercept Open-Meteo or any external API request.
  if (url.origin !== self.location.origin) return;
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
