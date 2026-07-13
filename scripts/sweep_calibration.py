"""Sweep multiple threshold calibrations for the combined probe and compare.

For each (recall_orange, recall_rouge) pair, we:
- calibrate thresholds on val,
- measure on test the detection rate ORANGE + ROUGE,
- measure the median lead time (in the sense of "earliest crossover within
  a wide lookback window"),
- measure the false alarm rate (fraction of non-onset windows that trigger).

Produces a comparison table + one figure with three curves: detection rate,
median lead, false alarm rate, all as function of recall target.

Runs on the precomputed embeddings (fast, no re-encoding).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "docs" / "assets"
RUNS = ROOT / "runs" / "eval"


def fit_probe(Z_train, y_train):
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z_train)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"
    ).fit(Zs, y_train)
    return scaler, clf


def score(scaler, clf, Z):
    return clf.predict_proba(scaler.transform(Z))[:, 1]


def threshold_for_recall(y_val, s_val, target):
    _, r, thr = precision_recall_curve(y_val, s_val)
    r = r[:-1]
    idx = np.where(r >= target)[0]
    if len(idx) == 0:
        return float(thr.min())
    return float(thr[idx[-1]])


def compute_metrics(y_test, s_test, ts_test, sid_test, orange, rouge,
                    step_hours=3, back_hours=120):
    """Detection rate, false alarm rate, lead time distribution.

    Lead time = earliest crossing in [-back_hours, 0] before an ONSET event.
    Lookback pushed to 120h (5 days) to escape the 45h saturation seen
    at chapter 15.
    """
    ts_test_dt = ts_test.astype("datetime64[ns]")
    back_steps = max(1, back_hours // step_hours)

    orange_leads = []
    rouge_leads = []
    n_events = 0
    n_false_orange = 0
    n_false_rouge = 0
    n_non_onset = 0

    for sid in np.unique(sid_test):
        m = sid_test == sid
        idx_s = np.flatnonzero(m)
        order = np.argsort(ts_test_dt[m])
        idx_s_sorted = idx_s[order]
        y_s = y_test[idx_s_sorted]
        s_s = s_test[idx_s_sorted]
        t_s = ts_test_dt[idx_s_sorted]

        prev = 0
        run_starts = []
        for j, v in enumerate(y_s):
            if v == 1 and prev == 0:
                run_starts.append(j)
            prev = v

        for pos in run_starts:
            n_events += 1
            onset_time = t_s[pos]
            back_limit = max(0, pos - back_steps)
            eo, er = None, None
            for k in range(back_limit, pos):
                if eo is None and s_s[k] >= orange:
                    eo = (onset_time - t_s[k]) / np.timedelta64(1, "h")
                if er is None and s_s[k] >= rouge:
                    er = (onset_time - t_s[k]) / np.timedelta64(1, "h")
                if eo is not None and er is not None:
                    break
            if eo is not None:
                orange_leads.append(float(eo))
            if er is not None:
                rouge_leads.append(float(er))

        # false alarms: score >= threshold on windows with y=0 AND no y=1 in following back_steps
        # Simpler: on y=0 windows, how many are ORANGE / ROUGE
        m_neg = y_s == 0
        n_non_onset += int(m_neg.sum())
        n_false_orange += int((s_s[m_neg] >= orange).sum())
        n_false_rouge += int((s_s[m_neg] >= rouge).sum())

    def _pct(x, y):
        return float(x) / float(max(1, y))

    return {
        "n_events": n_events,
        "n_non_onset": n_non_onset,
        "detection_orange": _pct(len(orange_leads), n_events),
        "detection_rouge": _pct(len(rouge_leads), n_events),
        "false_alarm_orange": _pct(n_false_orange, n_non_onset),
        "false_alarm_rouge": _pct(n_false_rouge, n_non_onset),
        "median_lead_orange_h": float(np.median(orange_leads)) if orange_leads else None,
        "median_lead_rouge_h": float(np.median(rouge_leads)) if rouge_leads else None,
        "p10_lead_orange_h": float(np.percentile(orange_leads, 10)) if orange_leads else None,
        "p90_lead_orange_h": float(np.percentile(orange_leads, 90)) if orange_leads else None,
        "orange_leads": orange_leads,
        "rouge_leads": rouge_leads,
    }


def main():
    print("Loading precomputed embeddings...")
    z = np.load(DATA / "embeddings_combined_stations.npz", allow_pickle=True)
    Z_train, Z_val, Z_test = z["Z_train"], z["Z_val"], z["Z_test"]
    y_train, y_val, y_test = z["y_train"], z["y_val"], z["y_test"]
    sid_test = z["station_test"]
    ts_test = z["ts_test"]

    print("Fitting probe once...")
    t0 = time.time()
    scaler, clf = fit_probe(Z_train, y_train)
    s_val = score(scaler, clf, Z_val)
    s_test = score(scaler, clf, Z_test)
    auc = roc_auc_score(y_test, s_test)
    print(f"  done in {time.time() - t0:.1f} s  AUC={auc:.4f}")

    # Grid of calibrations to compare
    configs = [
        ("actuel", 0.80, 0.40),
        ("modéré", 0.70, 0.30),
        ("prudent", 0.60, 0.25),
        ("strict", 0.50, 0.20),
        ("très strict", 0.40, 0.15),
    ]

    results = []
    print("\n== Sweep calibration ==")
    print(f"{'Config':16s}  {'target_O':>8s} {'target_R':>8s}  {'thr_O':>6s} {'thr_R':>6s}  "
          f"{'det_O':>6s} {'det_R':>6s}  {'fa_O':>6s} {'fa_R':>6s}  "
          f"{'lead_med_O':>10s} {'p10_O':>6s} {'p90_O':>6s}")
    for name, recall_o, recall_r in configs:
        thr_o = threshold_for_recall(y_val, s_val, recall_o)
        thr_r = threshold_for_recall(y_val, s_val, recall_r)
        if thr_r < thr_o:
            thr_r = thr_o
        m = compute_metrics(y_test, s_test, ts_test, sid_test, thr_o, thr_r)
        row = {
            "name": name, "recall_orange": recall_o, "recall_rouge": recall_r,
            "thr_orange": thr_o, "thr_rouge": thr_r, **m,
        }
        results.append(row)
        med_o = f"{m['median_lead_orange_h']:>7.1f}h" if m['median_lead_orange_h'] else "  none "
        p10_o = f"{m['p10_lead_orange_h']:>4.1f}h" if m['p10_lead_orange_h'] else " -- "
        p90_o = f"{m['p90_lead_orange_h']:>4.1f}h" if m['p90_lead_orange_h'] else " -- "
        print(
            f"{name:16s}  {recall_o:>8.2f} {recall_r:>8.2f}  "
            f"{thr_o:>6.3f} {thr_r:>6.3f}  "
            f"{m['detection_orange']:>6.2%} {m['detection_rouge']:>6.2%}  "
            f"{m['false_alarm_orange']:>6.2%} {m['false_alarm_rouge']:>6.2%}  "
            f"{med_o:>10s} {p10_o:>6s} {p90_o:>6s}"
        )

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    names = [r["name"] for r in results]

    ax = axes[0]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [r["detection_orange"] for r in results],
            width=0.4, color="#E37400", label="ORANGE")
    ax.bar([i + 0.2 for i in x], [r["detection_rouge"] for r in results],
            width=0.4, color="#B00020", label="ROUGE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Taux de détection")
    ax.set_title("Détection des vrais événements")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.bar([i - 0.2 for i in x], [r["false_alarm_orange"] for r in results],
            width=0.4, color="#E37400", label="ORANGE")
    ax.bar([i + 0.2 for i in x], [r["false_alarm_rouge"] for r in results],
            width=0.4, color="#B00020", label="ROUGE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Taux de fausses alarmes")
    ax.set_title("Fausses alarmes (fenêtres sans onset)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    ax = axes[2]
    med_o = [r["median_lead_orange_h"] or 0 for r in results]
    med_r = [r["median_lead_rouge_h"] or 0 for r in results]
    ax.plot(x, med_o, color="#E37400", marker="o", label="médiane ORANGE")
    ax.plot(x, med_r, color="#B00020", marker="o", label="médiane ROUGE")
    ax.axhline(120, color="#9E9E9E", linestyle=":", label="borne lookback 120h")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Préavis médian (heures)")
    ax.set_title("Préavis effectif")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Sweep calibration ORANGE/ROUGE, sonde combined (recall cible sur validation)")
    fig.tight_layout()
    out = ASSETS / "step16_calibration_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nFigure : {out}")

    # save report
    for r in results:
        r.pop("orange_leads", None)
        r.pop("rouge_leads", None)
    (RUNS / "calibration_sweep.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
