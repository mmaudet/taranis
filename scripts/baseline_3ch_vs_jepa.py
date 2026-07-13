"""Sensor-constrained control: 3-channel baselines vs TS-JEPA-3ch.

Chapter 18 of the carnet. Reproduces chapter 17's methodology but on the
amputated 3-channel dataset (pressure, temp, humidity) that a portable
BLE sensor (RuuviTag Pro) can actually deliver.

Three models on the same test set:
  M0-3ch    : 17 hand-crafted features + LogReg
  HGB-3ch   : 17 hand-crafted features + HistGradientBoosting
  M1-3ch    : TS-JEPA-3ch embeddings + LogReg

Metrics: bootstrap AUC + AP + detection + LOO 14 stations + cross-regime.

Outputs:
  docs/assets/step18_baseline_vs_jepa.png
  docs/assets/step18_loo_comparison.png
  docs/assets/step18_cross_regime.png
  runs/eval/baseline_3ch_vs_jepa.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

from taranis.models import Baseline3ch, Baseline3chHGB

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "docs" / "assets"
RUNS = ROOT / "runs" / "eval"
ASSETS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)


LOO_STATIONS = [
    ("07481", "Lyon"),
    ("07510", "Bordeaux"),
    ("07149", "Orly"),
    ("07690", "Nice"),
    ("07747", "Perpignan"),
    ("07110", "Brest"),
    ("07591", "Embrun"),
    ("07471", "Le Puy"),
    ("07558", "Millau"),
    ("07627", "St Girons"),
    ("07621", "Tarbes"),
    ("07460", "Clermont-Fd"),
    ("07299", "Bâle-Mulhouse"),
    ("07037", "Rouen"),
]

PLAINE = ["07481", "07510", "07037", "07149", "07072", "07168",
          "07255", "07240", "07130", "07207", "07690", "07747",
          "07630", "07110"]
MONTAGNE = ["07591", "07471", "07558", "07627", "07621", "07460",
            "07434", "07535", "07299", "07280", "07181", "07190"]


def denorm(X_norm, mean, std):
    return (X_norm * std + mean).astype(np.float32)


def train_m0_3ch(X_train_raw, y_train, X_val_raw, X_test_raw):
    m = Baseline3ch(step_minutes=180)
    m.fit(X_train_raw, y_train)
    return m.predict_proba(X_val_raw), m.predict_proba(X_test_raw)


def train_hgb_3ch(X_train_raw, y_train, X_val_raw, X_test_raw):
    m = Baseline3chHGB(step_minutes=180, max_iter=200, max_depth=6)
    m.fit(X_train_raw, y_train)
    return m.predict_proba(X_val_raw), m.predict_proba(X_test_raw)


def train_m1(Z_train, y_train, Z_val, Z_test):
    scaler = StandardScaler().fit(Z_train)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"
    ).fit(scaler.transform(Z_train), y_train)
    return (clf.predict_proba(scaler.transform(Z_val))[:, 1],
            clf.predict_proba(scaler.transform(Z_test))[:, 1])


def bootstrap_auc(y_test, s_test, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_test)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_test[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_test[idx], s_test[idx]))
    return {
        "mean": float(np.mean(aucs)),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
    }


def detection_and_lead(y_test, s_test, ts_test, sid_test, orange, rouge,
                       back_hours=120, step_hours=3):
    ts = ts_test.astype("datetime64[ns]")
    back_steps = max(1, back_hours // step_hours)
    o_leads, r_leads = [], []
    n_events, n_neg, n_false_o, n_false_r = 0, 0, 0, 0
    for sid in np.unique(sid_test):
        m = sid_test == sid
        idx_s = np.flatnonzero(m)
        order = np.argsort(ts[m])
        idx_s = idx_s[order]
        y_s, s_s, t_s = y_test[idx_s], s_test[idx_s], ts[idx_s]
        prev = 0
        for j, v in enumerate(y_s):
            if v == 1 and prev == 0:
                n_events += 1
                back_limit = max(0, j - back_steps)
                eo, er = None, None
                for k in range(back_limit, j):
                    if eo is None and s_s[k] >= orange:
                        eo = (t_s[j] - t_s[k]) / np.timedelta64(1, "h")
                    if er is None and s_s[k] >= rouge:
                        er = (t_s[j] - t_s[k]) / np.timedelta64(1, "h")
                    if eo is not None and er is not None:
                        break
                if eo is not None:
                    o_leads.append(float(eo))
                if er is not None:
                    r_leads.append(float(er))
            prev = v
        m_neg = y_s == 0
        n_neg += int(m_neg.sum())
        n_false_o += int((s_s[m_neg] >= orange).sum())
        n_false_r += int((s_s[m_neg] >= rouge).sum())
    return {
        "n_events": n_events,
        "detection_orange": len(o_leads) / max(1, n_events),
        "detection_rouge": len(r_leads) / max(1, n_events),
        "false_alarm_orange": n_false_o / max(1, n_neg),
        "false_alarm_rouge": n_false_r / max(1, n_neg),
        "median_lead_orange_h": float(np.median(o_leads)) if o_leads else None,
        "median_lead_rouge_h": float(np.median(r_leads)) if r_leads else None,
    }


def calibrate_thresholds(y_val, s_val, target_o=0.70, target_r=0.30):
    _, r, thr = precision_recall_curve(y_val, s_val)
    r = r[:-1]
    idx_o = np.where(r >= target_o)[0]
    idx_r = np.where(r >= target_r)[0]
    thr_o = float(thr[idx_o[-1]]) if len(idx_o) else float(thr.min())
    thr_r = float(thr[idx_r[-1]]) if len(idx_r) else float(thr.max())
    if thr_r < thr_o:
        thr_r = thr_o
    return thr_o, thr_r


def loo_m0(X_train_raw, y_train, sid_train, X_test_raw, y_test, sid_test, stations):
    rows = []
    for sid, label in stations:
        m_tr = sid_train != sid
        m_te = sid_test == sid
        if m_te.sum() == 0 or len(np.unique(y_test[m_te])) < 2:
            continue
        m = Baseline3ch(step_minutes=180)
        m.fit(X_train_raw[m_tr], y_train[m_tr])
        s = m.predict_proba(X_test_raw[m_te])
        auc = roc_auc_score(y_test[m_te], s)
        rows.append({"station": sid, "label": label, "auc": float(auc)})
        print(f"    M0-3ch LOO  {sid} {label:15s}  n={int(m_te.sum())}  AUC={auc:.3f}")
    return rows


def loo_hgb(X_train_raw, y_train, sid_train, X_test_raw, y_test, sid_test, stations):
    rows = []
    for sid, label in stations:
        m_tr = sid_train != sid
        m_te = sid_test == sid
        if m_te.sum() == 0 or len(np.unique(y_test[m_te])) < 2:
            continue
        m = Baseline3chHGB(step_minutes=180, max_iter=200, max_depth=6)
        m.fit(X_train_raw[m_tr], y_train[m_tr])
        s = m.predict_proba(X_test_raw[m_te])
        auc = roc_auc_score(y_test[m_te], s)
        rows.append({"station": sid, "label": label, "auc": float(auc)})
        print(f"    HGB-3ch LOO {sid} {label:15s}  n={int(m_te.sum())}  AUC={auc:.3f}")
    return rows


def loo_m1(Z_train, y_train, sid_train, Z_test, y_test, sid_test, stations):
    rows = []
    for sid, label in stations:
        m_tr = sid_train != sid
        m_te = sid_test == sid
        if m_te.sum() == 0 or len(np.unique(y_test[m_te])) < 2:
            continue
        scaler = StandardScaler().fit(Z_train[m_tr])
        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"
        ).fit(scaler.transform(Z_train[m_tr]), y_train[m_tr])
        s = clf.predict_proba(scaler.transform(Z_test[m_te]))[:, 1]
        auc = roc_auc_score(y_test[m_te], s)
        rows.append({"station": sid, "label": label, "auc": float(auc)})
        print(f"    M1-3ch LOO  {sid} {label:15s}  n={int(m_te.sum())}  AUC={auc:.3f}")
    return rows


def main():
    print("Loading 3-channel dataset...")
    d = np.load(DATA / "real_combined_3ch_windows.npz", allow_pickle=True)
    X_train_n = d["X_train"].astype(np.float32)
    X_val_n = d["X_val"].astype(np.float32)
    X_test_n = d["X_test"].astype(np.float32)
    y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
    sid_train, sid_test = d["station_train"], d["station_test"]
    ts_test = d["ts_test"]
    mean, std = d["mean"], d["std"]

    X_train_r = denorm(X_train_n, mean, std)
    X_val_r = denorm(X_val_n, mean, std)
    X_test_r = denorm(X_test_n, mean, std)

    print("Loading precomputed 3-channel embeddings...")
    e = np.load(DATA / "embeddings_3ch.npz", allow_pickle=True)
    Z_train, Z_val, Z_test = e["Z_train"], e["Z_val"], e["Z_test"]

    print(f"\n  train {len(y_train):,}, val {len(y_val):,}, test {len(y_test):,}")
    print(f"  prevalence train {y_train.mean():.4f}, test {y_test.mean():.4f}\n")

    all_results = {}
    for name, train_fn, use_emb in [
        ("M0-3ch", train_m0_3ch, False),
        ("HGB-3ch", train_hgb_3ch, False),
        ("M1-3ch (TS-JEPA)", train_m1, True),
    ]:
        print(f"== {name} ==")
        t0 = time.time()
        if use_emb:
            s_val, s_test = train_fn(Z_train, y_train, Z_val, Z_test)
        else:
            s_val, s_test = train_fn(X_train_r, y_train, X_val_r, X_test_r)
        auc_test = roc_auc_score(y_test, s_test)
        ap_test = average_precision_score(y_test, s_test)
        print(f"  fit + predict in {time.time() - t0:.1f} s  AUC={auc_test:.4f}  AP={ap_test:.4f}")
        t1 = time.time()
        boot = bootstrap_auc(y_test, s_test, n_boot=200)
        print(f"  bootstrap: mean={boot['mean']:.4f}  IC95%=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")
        thr_o, thr_r = calibrate_thresholds(y_val, s_val, 0.70, 0.30)
        det = detection_and_lead(y_test, s_test, ts_test, sid_test, thr_o, thr_r)
        print(f"  thr: O={thr_o:.3f} R={thr_r:.3f}")
        print(f"  detection: O={det['detection_orange']:.2%} R={det['detection_rouge']:.2%}")
        print(f"  false alarm: O={det['false_alarm_orange']:.2%} R={det['false_alarm_rouge']:.2%}")
        print(f"  bootstrap took {time.time() - t1:.1f} s\n")
        all_results[name] = {
            "auc": float(auc_test), "ap": float(ap_test), "boot": boot, "det": det,
            "thr_o": thr_o, "thr_r": thr_r,
        }

    print("== Leave-one-station-out (14 stations) ==")
    print("  M0-3ch...")
    t0 = time.time()
    loo_m0_res = loo_m0(X_train_r, y_train, sid_train, X_test_r, y_test, sid_test, LOO_STATIONS)
    print(f"  M0-3ch LOO done in {time.time() - t0:.1f} s")

    print("  HGB-3ch...")
    t0 = time.time()
    loo_hgb_res = loo_hgb(X_train_r, y_train, sid_train, X_test_r, y_test, sid_test, LOO_STATIONS)
    print(f"  HGB-3ch LOO done in {time.time() - t0:.1f} s")

    print("  M1-3ch (TS-JEPA)...")
    t0 = time.time()
    loo_m1_res = loo_m1(Z_train, y_train, sid_train, Z_test, y_test, sid_test, LOO_STATIONS)
    print(f"  M1-3ch LOO done in {time.time() - t0:.1f} s\n")

    print("== Cross regime plaine <-> montagne ==")
    cross = {}
    for tr_name, tr_sids in [("plaine", PLAINE), ("montagne", MONTAGNE)]:
        for te_name, te_sids in [("plaine", PLAINE), ("montagne", MONTAGNE)]:
            m_tr = np.isin(sid_train, tr_sids)
            m_te = np.isin(sid_test, te_sids)
            if m_tr.sum() < 100 or m_te.sum() < 100:
                continue
            key_prefix = f"{tr_name}_to_{te_name}"
            m = Baseline3ch(step_minutes=180)
            m.fit(X_train_r[m_tr], y_train[m_tr])
            s = m.predict_proba(X_test_r[m_te])
            auc_m0 = roc_auc_score(y_test[m_te], s)
            m = Baseline3chHGB(step_minutes=180, max_iter=200, max_depth=6)
            m.fit(X_train_r[m_tr], y_train[m_tr])
            s = m.predict_proba(X_test_r[m_te])
            auc_hgb = roc_auc_score(y_test[m_te], s)
            scaler = StandardScaler().fit(Z_train[m_tr])
            clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(
                scaler.transform(Z_train[m_tr]), y_train[m_tr])
            s = clf.predict_proba(scaler.transform(Z_test[m_te]))[:, 1]
            auc_m1 = roc_auc_score(y_test[m_te], s)
            cross[key_prefix] = {"M0-3ch": float(auc_m0), "HGB-3ch": float(auc_hgb), "M1-3ch": float(auc_m1)}
            print(f"  {key_prefix:22s}  M0-3ch={auc_m0:.3f}  HGB-3ch={auc_hgb:.3f}  M1-3ch={auc_m1:.3f}")

    print("\n== Figures ==")
    plot_summary(all_results, ASSETS / "step18_baseline_vs_jepa.png")
    plot_loo({"M0-3ch": loo_m0_res, "HGB-3ch": loo_hgb_res, "M1-3ch": loo_m1_res},
             ASSETS / "step18_loo_comparison.png")
    plot_cross(cross, ASSETS / "step18_cross_regime.png")

    report = {
        "models": all_results,
        "loo": {"M0-3ch": loo_m0_res, "HGB-3ch": loo_hgb_res, "M1-3ch": loo_m1_res},
        "cross_regime": cross,
    }
    out = RUNS / "baseline_3ch_vs_jepa.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report: {out}")


def plot_summary(results, path):
    models = list(results.keys())
    aucs = [results[m]["auc"] for m in models]
    ap = [results[m]["ap"] for m in models]
    det_o = [results[m]["det"]["detection_orange"] for m in models]
    fa_o = [results[m]["det"]["false_alarm_orange"] for m in models]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    colors = ["#78909C", "#455A64", "#B00020"]
    for ax, vals, title, ylim in [
        (axes[0], aucs, "AUC test", (0.60, 0.80)),
        (axes[1], ap, "Average Precision", (0.0, 0.15)),
        (axes[2], det_o, "Détection ORANGE (rappel 70%)", (0.5, 1.0)),
        (axes[3], fa_o, "Fausse alarme ORANGE", (0.0, 0.5)),
    ]:
        ax.bar(range(len(models)), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v + (ylim[1] - ylim[0]) * 0.02, f"{v:.3f}", ha="center", fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(ylim)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Chapitre 18, sonde capteur 3 canaux (P, T, HR)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_loo(loos, path):
    stations = [r["label"] for r in loos["M0-3ch"] if "auc" in r]
    auc_m0 = [r["auc"] for r in loos["M0-3ch"] if "auc" in r]
    auc_hgb = [next((rr["auc"] for rr in loos["HGB-3ch"] if rr.get("label") == s), np.nan) for s in stations]
    auc_m1 = [next((rr["auc"] for rr in loos["M1-3ch"] if rr.get("label") == s), np.nan) for s in stations]

    order = np.argsort(auc_m1)
    stations = [stations[i] for i in order]
    auc_m0 = [auc_m0[i] for i in order]
    auc_hgb = [auc_hgb[i] for i in order]
    auc_m1 = [auc_m1[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(stations))
    w = 0.27
    ax.barh(x - w, auc_m0, w, color="#78909C", label="M0-3ch (17 features + LogReg)")
    ax.barh(x, auc_hgb, w, color="#455A64", label="HGB-3ch (17 features + HistGB)")
    ax.barh(x + w, auc_m1, w, color="#B00020", label="M1-3ch (TS-JEPA + LogReg)")
    ax.set_yticks(x)
    ax.set_yticklabels(stations, fontsize=9)
    ax.axvline(0.5, color="#9E9E9E", linestyle=":", linewidth=1)
    ax.set_xlabel("AUC (station tenue en dehors)")
    ax.set_xlim(0.4, 0.9)
    ax.set_title("Chapitre 18, LOO 14 stations sur capteur 3 canaux")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cross(cross, path):
    labels = ["plaine->plaine", "plaine->montagne", "montagne->plaine", "montagne->montagne"]
    labels_short = ["P->P", "P->M", "M->P", "M->M"]
    m0 = [cross[k]["M0-3ch"] for k in ["plaine_to_plaine", "plaine_to_montagne",
                                       "montagne_to_plaine", "montagne_to_montagne"]]
    hgb = [cross[k]["HGB-3ch"] for k in ["plaine_to_plaine", "plaine_to_montagne",
                                          "montagne_to_plaine", "montagne_to_montagne"]]
    m1 = [cross[k]["M1-3ch"] for k in ["plaine_to_plaine", "plaine_to_montagne",
                                        "montagne_to_plaine", "montagne_to_montagne"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels_short))
    w = 0.27
    ax.bar(x - w, m0, w, color="#78909C", label="M0-3ch (17 features)")
    ax.bar(x, hgb, w, color="#455A64", label="HGB-3ch")
    ax.bar(x + w, m1, w, color="#B00020", label="M1-3ch (TS-JEPA)")
    for i, (a, b, c) in enumerate(zip(m0, hgb, m1, strict=True)):
        ax.text(i - w, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + w, c + 0.01, f"{c:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in labels_short])
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Chapitre 18, transfert plaine/montagne, sonde capteur 3 canaux")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
