"""Schémas pédagogiques pour les étapes 3 et 4.

Trois figures :

- step3_paradigmes.png : autoencodeur / contrastif / JEPA côte à côte.
- step4_pixel_vs_latent.png : pourquoi prédire dans l'espace latent.
- step4_jepa_arch.png : l'architecture JEPA sur une fenêtre temporelle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, label, facecolor="#E8F0FE", edgecolor="#1A73E8", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2, color="black", style="->", lw=1.2, mutation=12):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "linewidth": lw,
            "mutation_scale": mutation,
        },
    )


def _clean(ax, title=""):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)


def figure_paradigmes():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # ---- 1. Autoencodeur ----
    ax = axes[0]
    _clean(ax, "Autoencodeur")
    _box(ax, 0.05, 0.55, 0.15, 0.15, "x")
    _box(ax, 0.28, 0.55, 0.18, 0.15, "encodeur")
    _box(ax, 0.54, 0.55, 0.10, 0.15, "z")
    _box(ax, 0.70, 0.55, 0.18, 0.15, "décodeur")
    _box(ax, 0.05, 0.15, 0.15, 0.15, "x̂")

    _arrow(ax, 0.20, 0.625, 0.28, 0.625)
    _arrow(ax, 0.46, 0.625, 0.54, 0.625)
    _arrow(ax, 0.64, 0.625, 0.70, 0.625)
    _arrow(ax, 0.79, 0.55, 0.79, 0.30)
    _arrow(ax, 0.79, 0.225, 0.20, 0.225)
    ax.text(0.5, 0.05, "perte : ||x - x̂||\n(espace d'entrée)", ha="center", fontsize=9,
            color="#B00020")

    # ---- 2. Contrastif ----
    ax = axes[1]
    _clean(ax, "Contrastif")
    _box(ax, 0.05, 0.75, 0.15, 0.12, "x, vue 1", facecolor="#E6F4EA", edgecolor="#137333")
    _box(ax, 0.05, 0.55, 0.15, 0.12, "x, vue 2", facecolor="#E6F4EA", edgecolor="#137333")
    _box(ax, 0.05, 0.20, 0.15, 0.12, "y (autre)", facecolor="#FCE8E6", edgecolor="#B00020")

    _box(ax, 0.28, 0.55, 0.18, 0.32, "encodeur\npartagé")
    _box(ax, 0.28, 0.20, 0.18, 0.12, "encodeur\npartagé")

    _box(ax, 0.55, 0.75, 0.12, 0.12, "z₁", facecolor="#E6F4EA", edgecolor="#137333")
    _box(ax, 0.55, 0.55, 0.12, 0.12, "z₂", facecolor="#E6F4EA", edgecolor="#137333")
    _box(ax, 0.55, 0.20, 0.12, 0.12, "z_y", facecolor="#FCE8E6", edgecolor="#B00020")

    _arrow(ax, 0.20, 0.81, 0.28, 0.75)
    _arrow(ax, 0.20, 0.61, 0.28, 0.65)
    _arrow(ax, 0.20, 0.26, 0.28, 0.26)
    _arrow(ax, 0.46, 0.81, 0.55, 0.81)
    _arrow(ax, 0.46, 0.61, 0.55, 0.61)
    _arrow(ax, 0.46, 0.26, 0.55, 0.26)

    # attirance et répulsion
    _arrow(ax, 0.61, 0.75, 0.61, 0.67, color="#137333", style="<->", lw=1.8, mutation=14)
    ax.text(0.72, 0.71, "rapprocher", fontsize=9, color="#137333")

    _arrow(ax, 0.61, 0.55, 0.61, 0.32, color="#B00020", style="<->", lw=1.8, mutation=14)
    ax.text(0.72, 0.44, "éloigner", fontsize=9, color="#B00020")

    ax.text(0.5, 0.05, "perte : InfoNCE (contraste)", ha="center", fontsize=9,
            color="#B00020")

    # ---- 3. JEPA ----
    ax = axes[2]
    _clean(ax, "JEPA (prédictif en latent)")
    _box(ax, 0.02, 0.65, 0.18, 0.15, "x_context")
    _box(ax, 0.02, 0.25, 0.18, 0.15, "x_target")

    _box(ax, 0.25, 0.65, 0.16, 0.15, "encodeur\nonline")
    _box(ax, 0.25, 0.25, 0.16, 0.15, "encodeur\ncible (EMA,\nstop-grad)",
         facecolor="#FDE7C9", edgecolor="#E37400", fontsize=8)

    _box(ax, 0.45, 0.65, 0.12, 0.15, "z_ctx")
    _box(ax, 0.45, 0.25, 0.12, 0.15, "z_tgt", facecolor="#FDE7C9", edgecolor="#E37400")

    _box(ax, 0.62, 0.45, 0.18, 0.15, "prédicteur")
    _box(ax, 0.83, 0.45, 0.14, 0.15, "ẑ_tgt")

    _arrow(ax, 0.20, 0.725, 0.25, 0.725)
    _arrow(ax, 0.20, 0.325, 0.25, 0.325)
    _arrow(ax, 0.41, 0.725, 0.45, 0.725)
    _arrow(ax, 0.41, 0.325, 0.45, 0.325)

    _arrow(ax, 0.57, 0.725, 0.62, 0.60)
    _arrow(ax, 0.80, 0.525, 0.83, 0.525)
    _arrow(ax, 0.83, 0.45, 0.57, 0.32, color="#B00020", style="<->", lw=1.6, mutation=14)

    ax.text(0.55, 0.05, "perte : ||ẑ_tgt - z_tgt||\n(espace latent)",
            ha="center", fontsize=9, color="#B00020")

    fig.suptitle("Trois familles d'apprentissage auto-supervisé", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ASSETS / "step3_paradigmes.png", dpi=140)
    plt.close(fig)


def figure_pixel_vs_latent():
    """Comparaison intuitive prédire les pixels vs prédire dans l'espace latent."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Gauche : prédire les valeurs brutes ----
    ax = axes[0]
    _clean(ax, "Prédire les valeurs brutes (autoencodeur, MAE, MSE-pixel)")
    _box(ax, 0.05, 0.55, 0.20, 0.20, "fenêtre\nvisible")
    _box(ax, 0.30, 0.55, 0.20, 0.20, "modèle")
    _box(ax, 0.55, 0.55, 0.35, 0.20, "prédire chaque valeur\n(pression, humidité,\nvent, ...)",
         facecolor="#FCE8E6", edgecolor="#B00020")

    _arrow(ax, 0.25, 0.65, 0.30, 0.65)
    _arrow(ax, 0.50, 0.65, 0.55, 0.65)

    ax.text(
        0.5,
        0.30,
        "Le modèle doit \"peindre\" chaque détail :\n"
        "bruit de mesure, oscillations sans importance,\n"
        "artefacts du capteur. Effort gaspillé.",
        ha="center",
        fontsize=9,
    )

    # ---- Droite : prédire dans le latent ----
    ax = axes[1]
    _clean(ax, "Prédire dans l'espace des représentations (JEPA)")
    _box(ax, 0.03, 0.55, 0.18, 0.20, "fenêtre\nvisible")
    _box(ax, 0.24, 0.55, 0.16, 0.20, "encodeur")
    _box(ax, 0.43, 0.55, 0.12, 0.20, "z_ctx")
    _box(ax, 0.58, 0.55, 0.18, 0.20, "prédicteur")
    _box(ax, 0.79, 0.55, 0.17, 0.20, "ẑ_tgt",
         facecolor="#E6F4EA", edgecolor="#137333")

    for x1, x2 in [(0.21, 0.24), (0.40, 0.43), (0.55, 0.58), (0.76, 0.79)]:
        _arrow(ax, x1, 0.65, x2, 0.65)

    ax.text(
        0.5,
        0.30,
        "Le modèle prédit l'\"idée\" que se fait l'encodeur\n"
        "de la partie masquée, pas les valeurs exactes.\n"
        "Le bruit ne survit pas à l'encodeur, il n'est plus prédit.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(ASSETS / "step4_pixel_vs_latent.png", dpi=140)
    plt.close(fig)


def figure_jepa_arch():
    """Architecture TS-JEPA sur une fenêtre temporelle."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    _clean(ax, "")

    # patches
    n_patches = 12
    patch_w = 0.06
    patch_h = 0.09
    y_patches = 0.78
    x0 = 0.05

    # scénario : blocs cibles = patches 2-4 et 8-10
    target_ids = {2, 3, 4, 8, 9, 10}
    for i in range(n_patches):
        x = x0 + i * patch_w
        if i in target_ids:
            facecolor = "#FCE8E6"
            edgecolor = "#B00020"
            label = "T"
        else:
            facecolor = "#E8F0FE"
            edgecolor = "#1A73E8"
            label = "C"
        _box(ax, x, y_patches, patch_w * 0.9, patch_h, label, facecolor, edgecolor, fontsize=8)

    ax.text(x0 - 0.03, y_patches + patch_h / 2, "patches",
            ha="right", va="center", fontsize=10)
    ax.text(x0 + n_patches * patch_w + 0.02, y_patches + patch_h / 2,
            "C = contexte  T = cible",
            ha="left", va="center", fontsize=9)

    # encodeur contexte
    _box(ax, 0.15, 0.55, 0.24, 0.10, "encodeur contexte  fθ",
         facecolor="#E8F0FE", edgecolor="#1A73E8")
    _box(ax, 0.55, 0.55, 0.24, 0.10, "encodeur cible  fθ⁻  (EMA, stop-grad)",
         facecolor="#FDE7C9", edgecolor="#E37400")

    # flèches patches -> encodeurs
    _arrow(ax, 0.20, 0.78, 0.22, 0.65)
    _arrow(ax, 0.62, 0.78, 0.62, 0.65)

    # EMA
    _arrow(ax, 0.39, 0.60, 0.55, 0.60, color="#E37400", style="-|>", lw=1.5)
    ax.text(0.47, 0.62, "EMA", ha="center", fontsize=9, color="#E37400")

    # embeddings
    _box(ax, 0.18, 0.35, 0.20, 0.08, "z_context")
    _box(ax, 0.55, 0.35, 0.20, 0.08, "z_target (stop-grad)",
         facecolor="#FDE7C9", edgecolor="#E37400")

    _arrow(ax, 0.27, 0.55, 0.27, 0.43)
    _arrow(ax, 0.67, 0.55, 0.67, 0.43)

    # prédicteur
    _box(ax, 0.18, 0.15, 0.28, 0.10, "prédicteur  gφ  (positions cibles + z_context)")
    _box(ax, 0.55, 0.15, 0.20, 0.10, "ẑ_target")
    _arrow(ax, 0.32, 0.35, 0.32, 0.25)
    _arrow(ax, 0.46, 0.20, 0.55, 0.20)

    # perte
    _arrow(ax, 0.65, 0.35, 0.65, 0.25, color="#B00020", style="<->", lw=1.5)
    ax.text(0.75, 0.28, "perte SmoothL1", fontsize=10, color="#B00020")

    ax.text(0.5, 0.03,
            "gradient : encodeur contexte + prédicteur.  "
            "Encodeur cible mis à jour par EMA.  "
            "Aucune étiquette utilisée.",
            ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "step4_jepa_arch.png", dpi=140)
    plt.close(fig)


def main():
    figure_paradigmes()
    figure_pixel_vs_latent()
    figure_jepa_arch()
    print(f"Figures écrites dans {ASSETS}")


if __name__ == "__main__":
    main()
