// Build the 17 physical features on a 3-channel window (P, T, HR).
// Direct port of taranis.models.baseline_3ch.build_features_3ch.
//
// Input: window is a (Tw x 3) array of arrays [[p0, t0, h0], [p1, t1, h1], ...],
//        each row a physical measurement in original units.
// Output: Float32Array of length 17 in the exact feature order the
//         production HGB model expects.

const STEP_MINUTES = 180;
const A_MAGNUS = 17.625;
const B_MAGNUS = 243.04;

function lag(hours, Tw) {
    return Math.max(1, Math.min(Math.floor((hours * 60) / STEP_MINUTES), Tw - 1));
}

function dewPointC(tempC, rhPct) {
    const rh = Math.min(100.0, Math.max(0.1, rhPct));
    const lam = Math.log(rh / 100.0) + (A_MAGNUS * tempC) / (B_MAGNUS + tempC);
    return (B_MAGNUS * lam) / (A_MAGNUS - lam);
}

export function buildFeatures3ch(window) {
    const Tw = window.length;
    if (Tw < 4) throw new Error(`window too short: ${Tw}`);

    const p = window.map(r => r[0]);
    const t = window.map(r => r[1]);
    const h = window.map(r => r[2]);

    const lag3 = lag(3, Tw);
    const lag6 = lag(6, Tw);
    const lag12 = lag(12, Tw);
    const lag24 = lag(24, Tw);
    const last = Tw - 1;

    const pLast = p[last];
    const pTrend3 = pLast - p[last - lag3];
    const pTrend6 = pLast - p[last - lag6];
    const pTrend12 = pLast - p[last - lag12];

    // slice(-1 - lag24) matches Python's arr[-1 - lag24:]; produces the last (lag24 + 1) elements
    const sliceLast = (arr, n) => arr.slice(Math.max(0, arr.length - n));
    const window24 = sliceLast(p, lag24 + 1);
    const window12t = sliceLast(t, lag12 + 1);
    const window12h = sliceLast(h, lag12 + 1);
    const window6h = sliceLast(h, lag6 + 1);

    const pMin24 = Math.min(...window24);
    const meanArr = arr => arr.reduce((s, x) => s + x, 0) / arr.length;
    const stdArr = arr => {
        const m = meanArr(arr);
        return Math.sqrt(meanArr(arr.map(x => (x - m) ** 2)));
    };
    const pStd24 = stdArr(window24);

    const tLast = t[last];
    const tAmp = Math.max(...t) - Math.min(...t);
    const tMean12 = meanArr(window12t);
    const tDropoff3 = tLast - t[last - lag3];

    const hLast = h[last];
    const hMean6 = meanArr(window6h);
    const hDelta6 = hLast - h[last - lag6];
    const hMax12 = Math.max(...window12h);

    const td = dewPointC(tLast, hLast);
    const tdSpread = tLast - td;

    const px = pTrend6 * hLast;

    const f = new Float32Array(17);
    f[0] = pLast;      f[1] = pTrend3;   f[2] = pTrend6;
    f[3] = pTrend12;   f[4] = pMin24;    f[5] = pStd24;
    f[6] = tLast;      f[7] = tAmp;      f[8] = tMean12;   f[9] = tDropoff3;
    f[10] = hLast;     f[11] = hMean6;   f[12] = hDelta6;  f[13] = hMax12;
    f[14] = td;        f[15] = tdSpread;
    f[16] = px;
    return f;
}
