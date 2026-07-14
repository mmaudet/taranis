// RuuviTag Data Format 5 (RAWv2) parser.
//
// Reference: https://docs.ruuvi.com/communication/bluetooth-advertisements/data-format-5-rawv2
//
// The RuuviTag Pro 4-in-1 broadcasts every second a BLE advertisement whose
// manufacturer specific data starts with:
//   byte 0    : data format = 0x05
//   byte 1-2  : temperature   int16,  0.005 °C per LSB
//   byte 3-4  : humidity      uint16, 0.0025 % per LSB
//   byte 5-6  : pressure      uint16, in Pa, offset +50000
//   byte 7-12 : acceleration  int16 x/y/z, 0.001 g per LSB (unused here)
//   byte 13   : power info    (5 bits battery voltage + 3 bits tx power)
//   byte 14   : movement counter
//   byte 15-16: measurement sequence number
//   byte 17-22: MAC address
//
// Reserved values (0x8000 for temp, 0xFFFF for hum/press) mean "sensor
// invalid or not populated"; we return null in that case.

export const RUUVI_MANUFACTURER_ID = 0x0499;
export const RUUVI_FORMAT_V5 = 0x05;

/**
 * Parse a RAWv2 (format 5) manufacturer data payload.
 *
 * @param {DataView|Uint8Array|ArrayBuffer} data
 * @returns {null|{
 *   format: number,
 *   temp: number|null,     // deg C
 *   humidity: number|null, // %
 *   pressure: number|null, // hPa
 *   batteryMv: number|null,
 *   txPowerDbm: number|null,
 *   movementCounter: number,
 *   sequence: number,
 *   mac: string,
 * }}
 */
export function parseRuuviV5(data) {
    const dv = toDataView(data);
    if (!dv || dv.byteLength < 24) return null;

    const format = dv.getUint8(0);
    if (format !== RUUVI_FORMAT_V5) return null;

    // temperature: 0.005 deg C per LSB, reserved 0x8000
    const tempRaw = dv.getInt16(1, false);
    const temp = tempRaw === -0x8000 ? null : tempRaw * 0.005;

    // humidity: 0.0025 % per LSB, reserved 0xFFFF
    const humRaw = dv.getUint16(3, false);
    const humidity = humRaw === 0xFFFF ? null : humRaw * 0.0025;

    // pressure: Pa, offset +50000, reserved 0xFFFF
    const presRaw = dv.getUint16(5, false);
    const pressure = presRaw === 0xFFFF ? null : (presRaw + 50000) / 100.0; // Pa -> hPa

    // acceleration x, y, z: int16 signed, 0.001 g per LSB
    const accX = dv.getInt16(7, false) * 0.001;
    const accY = dv.getInt16(9, false) * 0.001;
    const accZ = dv.getInt16(11, false) * 0.001;

    // power info: byte 13-14 = 11 bits battery mv + 5 bits tx power
    const powerRaw = dv.getUint16(13, false);
    const battRaw = (powerRaw >> 5) & 0x7FF;
    const txRaw = powerRaw & 0x1F;
    const batteryMv = battRaw === 0x7FF ? null : 1600 + battRaw;
    const txPowerDbm = txRaw === 0x1F ? null : -40 + txRaw * 2;

    const movementCounter = dv.getUint8(15);
    const sequence = dv.getUint16(16, false);

    // MAC bytes 18..23
    const macBytes = [];
    for (let i = 18; i < 24; i++) macBytes.push(dv.getUint8(i).toString(16).padStart(2, "0"));
    const mac = macBytes.join(":").toUpperCase();

    return {
        format,
        temp,
        humidity,
        pressure,
        acceleration: { x: accX, y: accY, z: accZ },
        batteryMv,
        txPowerDbm,
        movementCounter,
        sequence,
        mac,
    };
}

/** Normalise input types to DataView big-endian. */
function toDataView(x) {
    if (!x) return null;
    if (x instanceof DataView) return x;
    if (x instanceof Uint8Array) return new DataView(x.buffer, x.byteOffset, x.byteLength);
    if (x instanceof ArrayBuffer) return new DataView(x);
    return null;
}

/**
 * Convert a parsed Ruuvi frame into a Taranis sample tuple.
 * Rejects frames with any missing physical channel we depend on.
 *
 * @param {ReturnType<typeof parseRuuviV5>} frame
 * @param {number} [now=Date.now()]
 * @returns {null|{t:number, p:number, temp:number, h:number, batteryMv?:number, mac?:string}}
 */
export function ruuviFrameToSample(frame, now = Date.now()) {
    if (!frame) return null;
    if (frame.pressure == null || frame.temp == null || frame.humidity == null) return null;
    return {
        t: now,
        p: frame.pressure,
        temp: frame.temp,
        h: frame.humidity,
        batteryMv: frame.batteryMv,
        mac: frame.mac,
    };
}
