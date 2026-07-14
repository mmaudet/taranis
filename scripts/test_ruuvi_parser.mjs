// Node parity test for the Ruuvi v5 parser.
//
// Reference frames come from the official Ruuvi documentation
// (https://docs.ruuvi.com/communication/bluetooth-advertisements/data-format-5-rawv2)
// and known-answer values so the parser doesn't drift.

import { parseRuuviV5, ruuviFrameToSample } from "../taranis/infer/static/js/ruuvi.js";

function hexToBytes(hex) {
    const clean = hex.replace(/[\s:]/g, "");
    const b = new Uint8Array(clean.length / 2);
    for (let i = 0; i < b.length; i++) b[i] = parseInt(clean.substr(i * 2, 2), 16);
    return b;
}

let pass = 0, fail = 0;
function expect(label, actual, expected, epsilon = 1e-6) {
    const ok = Math.abs(actual - expected) < epsilon;
    if (ok) {
        console.log(`  ✓ ${label}: ${actual}`);
        pass++;
    } else {
        console.log(`  ✗ ${label}: got ${actual}, expected ${expected}`);
        fail++;
    }
}

// ---- Vector 1: valid frame from Ruuvi documentation ----
// 05 12FC 5394 C37C 0004 FFFC 040C AC36 4200 CBB8 334C 884F
// Expected: temp = +24.3 °C, hum = 53.49 %, pressure = 1000.44 hPa
// battery raw 1723 mV, tx +4 dBm, movement 66, sequence 205, MAC CB:B8:33:4C:88:4F
console.log("Test 1: valid frame from Ruuvi docs");
{
    const bytes = hexToBytes("05 12FC 5394 C37C 0004 FFFC 040C AC36 4200 CDCB B833 4C88 4F");
    const frame = parseRuuviV5(bytes);
    if (!frame) { console.log("  ✗ frame is null"); fail++; }
    else {
        expect("format", frame.format, 5);
        expect("temp", frame.temp, 24.3, 0.005);
        expect("humidity", frame.humidity, 53.49, 0.005);
        expect("pressure", frame.pressure, 1000.44, 0.01);
        expect("acc.x", frame.acceleration.x, 0.004, 0.001);
        expect("acc.y", frame.acceleration.y, -0.004, 0.001);
        expect("acc.z", frame.acceleration.z, 1.036, 0.001);
        expect("batteryMv", frame.batteryMv, 2977);
        expect("txPowerDbm", frame.txPowerDbm, 4);
        expect("movementCounter", frame.movementCounter, 66);
        expect("sequence", frame.sequence, 205);
        if (frame.mac === "CB:B8:33:4C:88:4F") { console.log(`  ✓ mac: ${frame.mac}`); pass++; }
        else { console.log(`  ✗ mac: got ${frame.mac}, expected CB:B8:33:4C:88:4F`); fail++; }
    }
}

// ---- Vector 2: sample conversion (only physical channels) ----
console.log("\nTest 2: ruuviFrameToSample maps to Taranis format");
{
    const bytes = hexToBytes("05 12FC 5394 C37C 0004 FFFC 040C AC36 4200 CDCB B833 4C88 4F");
    const frame = parseRuuviV5(bytes);
    const s = ruuviFrameToSample(frame, 1_700_000_000_000);
    if (!s) { console.log("  ✗ sample is null"); fail++; }
    else {
        expect("t", s.t, 1_700_000_000_000);
        expect("p", s.p, 1000.44, 0.01);
        expect("temp", s.temp, 24.3, 0.005);
        expect("h", s.h, 53.49, 0.005);
        if (s.mac === "CB:B8:33:4C:88:4F") { console.log(`  ✓ mac: ${s.mac}`); pass++; }
        else { console.log(`  ✗ mac: ${s.mac}`); fail++; }
    }
}

// ---- Vector 3: invalid values (sensor not populated) ----
console.log("\nTest 3: reserved sentinel values return null channel");
{
    // Format 5, temp = 0x8000 (invalid), hum = 0xFFFF, press = 0xFFFF, 24 bytes total
    const bytes = hexToBytes("058000FFFFFFFF0000000000000000000000000000000000");  // 24 bytes
    const frame = parseRuuviV5(bytes);
    if (frame.temp === null && frame.humidity === null && frame.pressure === null) {
        console.log("  ✓ all three sensor channels are null");
        pass++;
    } else {
        console.log(`  ✗ expected null null null, got ${frame.temp}, ${frame.humidity}, ${frame.pressure}`);
        fail++;
    }
    const s = ruuviFrameToSample(frame);
    if (s === null) { console.log("  ✓ sample rejected"); pass++; }
    else { console.log("  ✗ sample should have been rejected"); fail++; }
}

// ---- Vector 4: wrong format id ----
console.log("\nTest 4: non-v5 format returns null");
{
    const bytes = hexToBytes("030000000000000000000000000000000000000000000000");  // 24 bytes, format 0x03
    const frame = parseRuuviV5(bytes);
    if (frame === null) { console.log("  ✓ format 3 rejected"); pass++; }
    else { console.log("  ✗ format 3 accepted"); fail++; }
}

// ---- Vector 5: short payload ----
console.log("\nTest 5: truncated payload rejected");
{
    const bytes = hexToBytes("05 12FC 5394"); // only 5 bytes
    const frame = parseRuuviV5(bytes);
    if (frame === null) { console.log("  ✓ short payload rejected"); pass++; }
    else { console.log("  ✗ short payload accepted"); fail++; }
}

console.log(`\n=== Summary ===`);
console.log(`passed: ${pass}   failed: ${fail}`);
if (fail > 0) process.exit(1);
