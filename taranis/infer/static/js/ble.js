// Web Bluetooth pairing with a RuuviTag (day 3 will complete this).
// This day-2 stub only checks capability and exposes a placeholder to
// prevent the UI from crashing if the user taps "Coupler capteur".

const RUUVI_MANUFACTURER_ID = 0x0499;  // Ruuvi Innovations Oy

export function isWebBluetoothAvailable() {
  return typeof navigator !== "undefined"
      && typeof navigator.bluetooth === "object"
      && typeof navigator.bluetooth.requestDevice === "function";
}

export function isIOS() {
  const ua = navigator.userAgent || "";
  return /iPhone|iPad|iPod/.test(ua);
}

// Placeholder: will be replaced day 3 with actual advertisement watcher
// (navigator.bluetooth.requestLEScan or requestDevice).
export async function pairRuuvi() {
  if (!isWebBluetoothAvailable()) {
    throw new Error(isIOS()
      ? "iOS Safari ne supporte pas Web Bluetooth. Ouvrez cette page dans l'app Bluefy (App Store, gratuit) pour coupler un capteur."
      : "Web Bluetooth n'est pas disponible dans ce navigateur. Essayez Chrome ou Edge sur Android.");
  }
  // Placeholder implementation: request a device by manufacturer ID
  return navigator.bluetooth.requestDevice({
    filters: [{ manufacturerData: [{ companyIdentifier: RUUVI_MANUFACTURER_ID }] }],
    optionalServices: [],
  });
}
