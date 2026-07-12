"""Training figures for step 6.

Three figures:

- step6_courbes.png  : train + val loss + LR + tau for the 2 runs.
- step6_collapse.png : std_moy and effective rank over time, for the 2 runs.
- step6_final.png    : final numerical summary synth vs real.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _load_log(path: Path):
    return json.loads(path.read_text())


def figure_courbes(logs: dict[str, dict]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    couleurs = {"synthétique": "#137333", "réel": "#455A64"}

    ax = axes[0, 0]
    for nom, log in logs.items():
        steps = [e["step"] for e in log["train_losses"]]
        losses = [e["loss"] for e in log["train_losses"]]
        ax.plot(steps, losses, label=nom, color=couleurs[nom], linewidth=1.6)
    ax.set_title("Perte SmoothL1 (train, moyennée par 100 pas)")
    ax.set_xlabel("Pas")
    ax.set_ylabel("Perte")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    for nom, log in logs.items():
        steps = [e["step"] for e in log["val_stats"]]
        losses = [e["val_loss"] for e in log["val_stats"]]
        ax.plot(steps, losses, label=nom, color=couleurs[nom], linewidth=1.6, marker="o", markersize=3)
    ax.set_title("Perte SmoothL1 (validation)")
    ax.set_xlabel("Pas")
    ax.set_ylabel("Perte")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for nom, log in logs.items():
        steps = [e["step"] for e in log["train_losses"]]
        lrs = [e["lr"] for e in log["train_losses"]]
        ax.plot(steps, lrs, label=nom, color=couleurs[nom], linewidth=1.6)
    ax.set_title("Taux d'apprentissage (warmup + cosine)")
    ax.set_xlabel("Pas")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for nom, log in logs.items():
        steps = [e["step"] for e in log["train_losses"]]
        taus = [e["tau"] for e in log["train_losses"]]
        ax.plot(steps, taus, label=nom, color=couleurs[nom], linewidth=1.6)
    ax.set_title("Coefficient EMA de l'encodeur cible (tau)")
    ax.set_xlabel("Pas")
    ax.set_ylabel("tau")
    ax.set_ylim(0.9955, 1.0005)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(ASSETS / "step6_courbes.png", dpi=140)
    plt.close(fig)


def figure_collapse(logs: dict[str, dict]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    couleurs = {"synthétique": "#137333", "réel": "#455A64"}

    ax = axes[0]
    for nom, log in logs.items():
        steps = [e["step"] for e in log["val_stats"]]
        stds = [e["z_ctx_std"] for e in log["val_stats"]]
        ax.plot(steps, stds, label=f"{nom} (contexte)", color=couleurs[nom],
                linewidth=1.6, marker="o", markersize=3)
    ax.axhline(0.1, color="#B00020", linestyle="--", alpha=0.5, label="seuil collapse")
    ax.set_title("Écart-type moyen des embeddings")
    ax.set_xlabel("Pas")
    ax.set_ylabel("std moyen")
    ax.set_ylim(0, 1.2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    d_model = list(logs.values())[0]["model_config"]["d_model"]
    for nom, log in logs.items():
        steps = [e["step"] for e in log["val_stats"]]
        ranks = [e["z_ctx_rank"] for e in log["val_stats"]]
        ax.plot(steps, ranks, label=f"{nom} (contexte)", color=couleurs[nom],
                linewidth=1.6, marker="o", markersize=3)
    ax.axhline(1.0, color="#B00020", linestyle="--", alpha=0.5, label="collapse total")
    ax.axhline(d_model, color="#137333", linestyle=":", alpha=0.5, label=f"dim latente ({d_model})")
    ax.set_title("Rang effectif de la covariance")
    ax.set_xlabel("Pas")
    ax.set_ylabel("Rang effectif")
    ax.set_ylim(0, d_model + 5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "step6_collapse.png", dpi=140)
    plt.close(fig)


def figure_final(logs: dict[str, dict]):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis("off")

    lines = ["Résumé final des deux entraînements TS-JEPA", ""]
    lines.append(f"{'Métrique':30s} {'Synthétique':>15s} {'Réel':>15s}")
    lines.append("-" * 62)

    def last(log, key, path=("val_stats",)):
        if path == ("val_stats",):
            return log["val_stats"][-1][key]
        return log["train_losses"][-1][key]

    synth, real = logs["synthétique"], logs["réel"]

    rows = [
        ("Perte train finale (moy 100 pas)",
         f"{synth['train_losses'][-1]['loss']:.5f}",
         f"{real['train_losses'][-1]['loss']:.5f}"),
        ("Perte val finale",
         f"{synth['val_stats'][-1]['val_loss']:.5f}",
         f"{real['val_stats'][-1]['val_loss']:.5f}"),
        ("z_ctx std final",
         f"{synth['val_stats'][-1]['z_ctx_std']:.3f}",
         f"{real['val_stats'][-1]['z_ctx_std']:.3f}"),
        ("z_ctx rang effectif final",
         f"{synth['val_stats'][-1]['z_ctx_rank']:.1f} / {synth['model_config']['d_model']}",
         f"{real['val_stats'][-1]['z_ctx_rank']:.1f} / {real['model_config']['d_model']}"),
        ("Temps mur (s)",
         f"{synth['wall_time_seconds']:.1f}",
         f"{real['wall_time_seconds']:.1f}"),
    ]

    for label, s, r in rows:
        lines.append(f"{label:30s} {s:>15s} {r:>15s}")

    text = "\n".join(lines)
    ax.text(0.0, 1.0, text, family="monospace", fontsize=10, va="top")
    fig.tight_layout()
    fig.savefig(ASSETS / "step6_final.png", dpi=140)
    plt.close(fig)


def main():
    logs = {
        "synthétique": _load_log(ROOT / "runs" / "tsjepa_synth" / "log.json"),
        "réel": _load_log(ROOT / "runs" / "tsjepa_real" / "log.json"),
    }
    figure_courbes(logs)
    figure_collapse(logs)
    figure_final(logs)
    print(f"Figures écrites dans {ASSETS}")


if __name__ == "__main__":
    main()
