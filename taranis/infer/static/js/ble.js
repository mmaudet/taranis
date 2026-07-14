// Web Bluetooth ingestion of a RuuviTag Pro 4-in-1.
//
// Two ingestion paths, depending on browser support:
//
// 1. `navigator.bluetooth.requestLEScan` (Chrome flag
//    `#enable-experimental-web-platform-features` or Chrome for Android
//    with the experimental Web Bluetooth Scanning enabled) lets us listen
//    to advertisements passively, without pairing. This is the "native"
//    RuuviTag mode: it broadcasts every second, no connection required.
//
// 2. `navigator.bluetooth.requestDevice` + GATT connection fallback:
//    slower to set up (user has to pick the device from a chooser), but
//    works on stock Chrome/Edge and on iOS Bluefy. The Nordic UART
//    Service (NUS) or the Ruuvi Environmental Sensing Service exposes
//    the same fields as the advertisement.
//
// The parser is shared: parseRuuviV5() consumes the raw 24-byte
// manufacturer specific payload from either path.

import {
    parseRuuviV5,
    ruuviFrameToSample,
    RUUVI_MANUFACTURER_ID,
} from "./ruuvi.js";
import { pushSample } from "./buffer.js";

// Nordic UART Service (used by RuuviTag firmware v3+ for GATT streaming)
const NUS_SERVICE = "6e400001-b5a3-f393-e0a3-a9e50e24dcca";
const NUS_TX_CHARACTERISTIC = "6e400003-b5a3-f393-e0a3-a9e50e24dcca";

export function isWebBluetoothAvailable() {
    return typeof navigator !== "undefined"
        && typeof navigator.bluetooth === "object"
        && typeof navigator.bluetooth.requestDevice === "function";
}

export function isLEScanAvailable() {
    return isWebBluetoothAvailable()
        && typeof navigator.bluetooth.requestLEScan === "function";
}

export function isIOS() {
    const ua = navigator.userAgent || "";
    return /iPhone|iPad|iPod/.test(ua);
}

/**
 * Pair with a RuuviTag. Chooses the best available path automatically.
 * Returns an object exposing `stop()` to disconnect and a promise that
 * resolves to metadata (name, mac) about the paired device.
 *
 * @param {Object} opts
 * @param {(sample: {t:number,p:number,temp:number,h:number,batteryMv?:number,mac?:string}) => void} opts.onSample
 * @param {(state: string, detail?: any) => void} [opts.onStatus]
 */
export async function pairRuuvi({ onSample, onStatus = () => {} }) {
    if (!isWebBluetoothAvailable()) {
        throw new Error(isIOS()
            ? "iOS Safari ne supporte pas Web Bluetooth. Ouvrez cette page dans l'app Bluefy (App Store, gratuit) pour coupler un capteur."
            : "Web Bluetooth n'est pas disponible dans ce navigateur. Essayez Chrome ou Edge sur Android.");
    }

    // Path 1: passive advertisement scan (best UX, no pairing prompt)
    if (isLEScanAvailable()) {
        onStatus("scan_start");
        try {
            const scan = await navigator.bluetooth.requestLEScan({
                filters: [{ manufacturerData: [{ companyIdentifier: RUUVI_MANUFACTURER_ID }] }],
                keepRepeatedDevices: true,
            });
            const listener = (event) => {
                const mfr = event.manufacturerData.get(RUUVI_MANUFACTURER_ID);
                if (!mfr) return;
                const frame = parseRuuviV5(mfr);
                const sample = ruuviFrameToSample(frame);
                if (sample) {
                    pushSample(sample);
                    onSample(sample);
                    onStatus("sample", { rssi: event.rssi, mac: sample.mac });
                }
            };
            navigator.bluetooth.addEventListener("advertisementreceived", listener);
            onStatus("scan_active");
            return {
                mode: "scan",
                stop: () => {
                    scan.stop();
                    navigator.bluetooth.removeEventListener("advertisementreceived", listener);
                    onStatus("stopped");
                },
            };
        } catch (e) {
            // Fall through to GATT path
            onStatus("scan_failed", e.message);
        }
    }

    // Path 2: requestDevice + GATT connection fallback
    onStatus("gatt_start");
    const device = await navigator.bluetooth.requestDevice({
        filters: [
            { manufacturerData: [{ companyIdentifier: RUUVI_MANUFACTURER_ID }] },
            { namePrefix: "Ruuvi" },
        ],
        optionalServices: [NUS_SERVICE],
    });
    onStatus("gatt_selected", { name: device.name || null });

    const server = await device.gatt.connect();
    onStatus("gatt_connected");

    let stopped = false;
    const nusService = await server.getPrimaryService(NUS_SERVICE).catch(() => null);
    if (nusService) {
        const tx = await nusService.getCharacteristic(NUS_TX_CHARACTERISTIC);
        await tx.startNotifications();
        tx.addEventListener("characteristicvaluechanged", (event) => {
            const dv = event.target.value; // DataView
            const frame = parseRuuviV5(dv);
            const sample = ruuviFrameToSample(frame);
            if (sample) {
                pushSample(sample);
                onSample(sample);
                onStatus("sample", { mac: sample.mac });
            }
        });
        onStatus("gatt_streaming");
    } else {
        onStatus("gatt_no_service");
    }

    device.addEventListener("gattserverdisconnected", () => {
        if (!stopped) onStatus("disconnected");
    });

    return {
        mode: "gatt",
        stop: () => {
            stopped = true;
            try { server.disconnect(); } catch (_) { /* ignore */ }
            onStatus("stopped");
        },
    };
}
