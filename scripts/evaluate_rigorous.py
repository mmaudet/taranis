"""Rigorous evaluation of the Taranis production probe (combined labels).

Runs four axes:
  1. Bootstrap AUC + AP + F1 with 95% percentile CI on the test set.
  2. Distribution of lead time (hours) before a real onset, ORANGE / ROUGE.
  3. Leave-one-station-out on 15 diverse stations.
  4. Cross-regime: train probe on plaines, test on mountains, and inverse.

All axes reuse the precomputed embeddings from
`data/embeddings_combined_stations.npz` for speed.

Outputs:
  - docs/assets/step15_axis1_bootstrap.png (violin plot of AUC bootstrap)
  - docs/assets/step15_axis2_lead_time.png (histogram + CDF)
  - docs/assets/step15_axis3_loo.png (bar chart by station)
  - docs/assets/step15_axis4_cross_regime.png (cross matrix)
  - runs/eval/rigorous_report.json (all raw numbers)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "docs" / "assets"
RUNS = ROOT / "runs" / "eval"
ASSETS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def fit_probe(Z_train, y_train):
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z_train)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"
    ).fit(Zs, y_train)
    return scaler, clf


def score(scaler, clf, Z):
    return clf.predict_proba(scaler.transform(Z))[:, 1]


# ---------------------------------------------------------------------------
# Axis 1: bootstrap
# ---------------------------------------------------------------------------


def axis1_bootstrap(y_test, s_test, n_boot=1000, seed=0):
    """Bootstrap AUC, AP, F1 at fixed 0.5 threshold on the test set.

    Returns dict with mean and (2.5%, 97.5%) percentile CIs.
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    auc_samples, ap_samples, f1_samples = [], [], []
    y_pred_fixed = (s_test >= 0.5).astype(int)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = y_test[idx], s_test[idx]
        p_b = y_pred_fixed[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_samples.append(roc_auc_score(y_b, s_b))
        ap_samples.append(average_precision_score(y_b, s_b))
        f1_samples.append(f1_score(y_b, p_b, zero_division=0))

    def _ci(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "ci_low": float(np.percentile(x, 2.5)),
            "ci_high": float(np.percentile(x, 97.5)),
            "n": int(len(x)),
        }

    return {
        "auc": _ci(auc_samples),
        "ap": _ci(ap_samples),
        "f1_at_0.5": _ci(f1_samples),
        "raw_auc": auc_samples,
    }


def plot_axis1(bs, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bs["raw_auc"], bins=40, color="#1A73E8", alpha=0.75, edgecolor="none")
    ci = bs["auc"]
    ax.axvline(ci["mean"], color="#B00020", linestyle="-", linewidth=2, label=f"mean {ci['mean']:.3f}")
    ax.axvline(ci["ci_low"], color="#B00020", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(ci["ci_high"], color="#B00020", linestyle="--", linewidth=1, alpha=0.7,
                label=f"IC 95% [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    ax.set_xlabel("AUC")
    ax.set_ylabel(f"Fréquence sur {ci['n']} bootstraps")
    ax.set_title("Axe 1, distribution bootstrap de l'AUC test set (sonde combined)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis 2: lead time distribution
# ---------------------------------------------------------------------------


def axis2_lead_time(y_test, s_test, ts_test, sid_test, orange_thr, rouge_thr,
                     step_hours: int = 3, back_hours: int = 45):
    """For every distinct ONSET EVENT in test, compute the earliest time the
    score crossed ORANGE / ROUGE within `back_hours` before the onset.

    A distinct onset event is defined as a **run of consecutive y=1 flags**
    per station. Each real storm produces up to H=8 consecutive y=1 (because
    y=1 = "onset in next 24h"). We only count the FIRST y=1 of each run as
    a unique event.

    Lead time is computed relative to that first y=1 timestamp, going back
    up to `back_hours` and finding the EARLIEST score >= threshold (not the
    last one).
    """
    ts_test_dt = ts_test.astype("datetime64[ns]")
    back_steps = max(1, back_hours // step_hours)

    orange_leads = []
    rouge_leads = []
    n_events = 0

    for sid in np.unique(sid_test):
        m = sid_test == sid
        idx_s = np.flatnonzero(m)
        order = np.argsort(ts_test_dt[m])
        idx_s_sorted = idx_s[order]

        y_s = y_test[idx_s_sorted]
        s_s = s_test[idx_s_sorted]
        t_s = ts_test_dt[idx_s_sorted]

        # identify runs of consecutive y=1
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
            # earliest cross of thresholds in the lookback window
            earliest_orange = None
            earliest_rouge = None
            for k in range(back_limit, pos):
                if earliest_orange is None and s_s[k] >= orange_thr:
                    earliest_orange = (onset_time - t_s[k]) / np.timedelta64(1, "h")
                if earliest_rouge is None and s_s[k] >= rouge_thr:
                    earliest_rouge = (onset_time - t_s[k]) / np.timedelta64(1, "h")
                if earliest_orange is not None and earliest_rouge is not None:
                    break
            if earliest_orange is not None:
                orange_leads.append(float(earliest_orange))
            if earliest_rouge is not None:
                rouge_leads.append(float(earliest_rouge))

    def _stats(x):
        if not x:
            return {"n": 0}
        x = np.asarray(x)
        return {
            "n": int(len(x)),
            "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
            "p90": float(np.percentile(x, 90)),
            "mean": float(np.mean(x)),
        }

    return {
        "orange": _stats(orange_leads),
        "rouge": _stats(rouge_leads),
        "n_events": n_events,
        "detection_rate_orange": len(orange_leads) / max(1, n_events),
        "detection_rate_rouge": len(rouge_leads) / max(1, n_events),
        "orange_leads": orange_leads,
        "rouge_leads": rouge_leads,
    }


def plot_axis2(a2, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    bins = np.arange(0, 46, 3)
    ax.hist(a2["orange_leads"], bins=bins, color="#E37400", alpha=0.6, label="ORANGE")
    ax.hist(a2["rouge_leads"], bins=bins, color="#B00020", alpha=0.6, label="ROUGE")
    ax.set_xlabel("Préavis avant onset (heures)")
    ax.set_ylabel("Nombre d'orages")
    ax.set_title("Distribution des préavis (test 2024-2025)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    if a2["orange_leads"]:
        sorted_o = np.sort(a2["orange_leads"])
        ax.plot(sorted_o, np.arange(1, len(sorted_o) + 1) / a2["n_events"],
                color="#E37400", label=f"ORANGE ({a2['detection_rate_orange']:.0%})")
    if a2["rouge_leads"]:
        sorted_r = np.sort(a2["rouge_leads"])
        ax.plot(sorted_r, np.arange(1, len(sorted_r) + 1) / a2["n_events"],
                color="#B00020", label=f"ROUGE ({a2['detection_rate_rouge']:.0%})")
    ax.set_xlabel("Préavis (heures)")
    ax.set_ylabel("Fraction d'orages détectés")
    ax.set_title("Courbe cumulée de détection")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.suptitle(f"Axe 2, préavis sur {a2['n_events']} vrais événements du test", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis 3: leave-one-station-out
# ---------------------------------------------------------------------------


LOO_STATIONS = [
    ("07481", "Lyon (plaine)"),
    ("07510", "Bordeaux (plaine)"),
    ("07149", "Orly (plaine)"),
    ("07690", "Nice (côte)"),
    ("07690", "Nice"),   # dup for guard, filtered below
    ("07747", "Perpignan (côte)"),
    ("07110", "Brest (Bretagne)"),
    ("07591", "Embrun (Alpes)"),
    ("07471", "Le Puy (Massif Central)"),
    ("07558", "Millau (Causses)"),
    ("07627", "St Girons (Pyrénées)"),
    ("07621", "Tarbes (Pyrénées)"),
    ("07460", "Clermont-Fd"),
    ("07299", "Bâle-Mulhouse"),
    ("07037", "Rouen"),
]
LOO_STATIONS = list(dict.fromkeys(LOO_STATIONS))  # dedup keeping order


def axis3_loo(Z_train, y_train, sid_train, Z_test, y_test, sid_test, stations):
    """Train probe on ALL train stations except `held_out`, then evaluate
    on the held-out station's TEST windows. Repeat for each station in list.
    """
    rows = []
    for sid, label in stations:
        mask_train = sid_train != sid
        mask_test = sid_test == sid
        if mask_test.sum() == 0 or mask_train.sum() == 0:
            rows.append({"station": sid, "label": label, "skipped": True})
            continue
        scaler, clf = fit_probe(Z_train[mask_train], y_train[mask_train])
        s_test = score(scaler, clf, Z_test[mask_test])
        y_t = y_test[mask_test]
        if len(np.unique(y_t)) < 2:
            rows.append({"station": sid, "label": label, "prevalence": float(y_t.mean()),
                         "n": int(mask_test.sum()), "skipped_uniclass": True})
            continue
        auc = roc_auc_score(y_t, s_test)
        ap = average_precision_score(y_t, s_test)
        rows.append({
            "station": sid, "label": label,
            "n_test": int(mask_test.sum()),
            "prevalence": float(y_t.mean()),
            "auc": float(auc), "ap": float(ap),
        })
        print(f"  LOO {sid:8s} {label:28s}  n={int(mask_test.sum()):>6d}  "
              f"prev={y_t.mean():.3f}  AUC={auc:.3f}  AP={ap:.3f}")
    return rows


def plot_axis3(rows, path, baseline_auc):
    rows_ok = [r for r in rows if "auc" in r]
    rows_ok.sort(key=lambda r: r["auc"])
    labels = [r["label"] for r in rows_ok]
    aucs = [r["auc"] for r in rows_ok]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(labels) * 0.35)))
    colors = ["#B00020" if a < 0.6 else "#E37400" if a < 0.7 else "#137333" for a in aucs]
    ax.barh(range(len(labels)), aucs, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(baseline_auc, color="#455A64", linestyle="--", linewidth=1,
                label=f"AUC in-sample ({baseline_auc:.3f})")
    ax.axvline(0.5, color="#9E9E9E", linestyle=":", linewidth=1, label="hasard")
    ax.set_xlabel("AUC (station tenue en dehors)")
    ax.set_xlim(0.45, max(0.85, max(aucs) + 0.05))
    ax.set_title(f"Axe 3, leave-one-station-out sur {len(labels)} stations")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis 4: cross-regime plaine vs montagne
# ---------------------------------------------------------------------------


PLAINE_STATIONS = ["07481", "07510", "07037", "07149", "07072", "07168",
                    "07255", "07240", "07130", "07207", "07690", "07747",
                    "07630", "07110"]
MONTAGNE_STATIONS = ["07591", "07471", "07558", "07627", "07621", "07460",
                      "07434", "07535", "07299", "07280", "07181", "07190"]


def axis4_cross_regime(Z_train, y_train, sid_train, Z_test, y_test, sid_test):
    """Train probe on plaines, test on mountains, and inverse."""
    results = {}
    for train_regime, train_sids in [("plaine", PLAINE_STATIONS),
                                       ("montagne", MONTAGNE_STATIONS)]:
        for test_regime, test_sids in [("plaine", PLAINE_STATIONS),
                                          ("montagne", MONTAGNE_STATIONS)]:
            m_tr = np.isin(sid_train, train_sids)
            m_te = np.isin(sid_test, test_sids)
            if m_tr.sum() < 100 or m_te.sum() < 100:
                results[f"{train_regime}->{test_regime}"] = {"skipped": True}
                continue
            scaler, clf = fit_probe(Z_train[m_tr], y_train[m_tr])
            s = score(scaler, clf, Z_test[m_te])
            y_t = y_test[m_te]
            if len(np.unique(y_t)) < 2:
                continue
            auc = roc_auc_score(y_t, s)
            ap = average_precision_score(y_t, s)
            key = f"{train_regime}->{test_regime}"
            results[key] = {
                "n_train": int(m_tr.sum()),
                "n_test": int(m_te.sum()),
                "prevalence_test": float(y_t.mean()),
                "auc": float(auc),
                "ap": float(ap),
            }
            print(f"  {key:24s}  ntrain={int(m_tr.sum()):>8d}  ntest={int(m_te.sum()):>7d}  "
                  f"prev={y_t.mean():.3f}  AUC={auc:.3f}  AP={ap:.3f}")
    return results


def plot_axis4(results, path):
    regimes = ["plaine", "montagne"]
    auc_matrix = np.full((2, 2), np.nan)
    ap_matrix = np.full((2, 2), np.nan)
    for i, tr in enumerate(regimes):
        for j, te in enumerate(regimes):
            k = f"{tr}->{te}"
            if k in results and "auc" in results[k]:
                auc_matrix[i, j] = results[k]["auc"]
                ap_matrix[i, j] = results[k]["ap"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mat, title, vmin, vmax in [
        (axes[0], auc_matrix, "AUC", 0.5, 0.85),
        (axes[1], ap_matrix, "Average Precision", 0.0, 0.2),
    ]:
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Test {r}" for r in regimes])
        ax.set_yticklabels([f"Train {r}" for r in regimes])
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < (vmin + vmax) / 2 else "black",
                            fontsize=13)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Axe 4, transfert cross-régime plaine/montagne")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading precomputed embeddings...")
    z = np.load(DATA / "embeddings_combined_stations.npz", allow_pickle=True)
    Z_train, Z_val, Z_test = z["Z_train"], z["Z_val"], z["Z_test"]
    y_train, y_val, y_test = z["y_train"], z["y_val"], z["y_test"]
    sid_train, sid_test = z["station_train"], z["station_test"]
    ts_test = z["ts_test"]
    print(f"  Z_train {Z_train.shape}, Z_test {Z_test.shape}")

    # Fit the reference probe once on the whole train, then reuse
    print("\n== Baseline probe on all-train ==")
    t0 = time.time()
    scaler, clf = fit_probe(Z_train, y_train)
    s_val = score(scaler, clf, Z_val)
    s_test = score(scaler, clf, Z_test)
    ref_auc = roc_auc_score(y_test, s_test)
    print(f"  Fit + score in {time.time() - t0:.1f} s  test AUC={ref_auc:.4f}")

    # Recover calibrated thresholds from precision_recall on val
    from sklearn.metrics import precision_recall_curve
    _, r, thr = precision_recall_curve(y_val, s_val)
    r = r[:-1]
    # target recall 80/40 as done in save_probe.py combined
    idx_o = np.where(r >= 0.80)[0]
    idx_r = np.where(r >= 0.40)[0]
    orange_thr = float(thr[idx_o[-1]]) if len(idx_o) else 0.4
    rouge_thr = float(thr[idx_r[-1]]) if len(idx_r) else 0.7
    print(f"  Calibrated thresholds: ORANGE={orange_thr:.3f}, ROUGE={rouge_thr:.3f}")

    report = {"baseline_auc": ref_auc,
              "orange_threshold": orange_thr,
              "rouge_threshold": rouge_thr}

    # ---- Axis 1 ----
    print("\n== Axis 1, bootstrap ==")
    t0 = time.time()
    a1 = axis1_bootstrap(y_test, s_test, n_boot=1000, seed=0)
    print(f"  done in {time.time() - t0:.1f} s")
    print(f"  AUC = {a1['auc']['mean']:.4f}  IC95%=[{a1['auc']['ci_low']:.4f}, {a1['auc']['ci_high']:.4f}]")
    print(f"  AP  = {a1['ap']['mean']:.4f}  IC95%=[{a1['ap']['ci_low']:.4f}, {a1['ap']['ci_high']:.4f}]")
    plot_axis1(a1, ASSETS / "step15_axis1_bootstrap.png")
    report["axis1"] = {k: v for k, v in a1.items() if k != "raw_auc"}

    # ---- Axis 2 ----
    print("\n== Axis 2, lead time ==")
    t0 = time.time()
    a2 = axis2_lead_time(y_test, s_test, ts_test, sid_test, orange_thr, rouge_thr)
    print(f"  done in {time.time() - t0:.1f} s")
    print(f"  ORANGE median lead = {a2['orange']['median']:.1f} h  ({a2['detection_rate_orange']:.0%} of {a2['n_events']} events)")
    print(f"  ROUGE  median lead = {a2['rouge']['median']:.1f} h  ({a2['detection_rate_rouge']:.0%})")
    plot_axis2(a2, ASSETS / "step15_axis2_lead_time.png")
    report["axis2"] = {k: v for k, v in a2.items() if not k.endswith("_leads")}

    # ---- Axis 3 ----
    print("\n== Axis 3, leave-one-station-out ==")
    t0 = time.time()
    a3 = axis3_loo(Z_train, y_train, sid_train, Z_test, y_test, sid_test, LOO_STATIONS)
    print(f"  done in {time.time() - t0:.1f} s")
    plot_axis3(a3, ASSETS / "step15_axis3_loo.png", ref_auc)
    report["axis3"] = a3

    # ---- Axis 4 ----
    print("\n== Axis 4, cross-regime ==")
    t0 = time.time()
    a4 = axis4_cross_regime(Z_train, y_train, sid_train, Z_test, y_test, sid_test)
    print(f"  done in {time.time() - t0:.1f} s")
    plot_axis4(a4, ASSETS / "step15_axis4_cross_regime.png")
    report["axis4"] = a4

    out = RUNS / "rigorous_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {out}")
    print("Figures:")
    for name in ["axis1_bootstrap", "axis2_lead_time", "axis3_loo", "axis4_cross_regime"]:
        print(f"  {ASSETS / f'step15_{name}.png'}")


if __name__ == "__main__":
    main()
