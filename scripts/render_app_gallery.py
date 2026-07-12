"""Assemble the three mockups (green, orange, red) side by side for the docs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"


def main():
    files = [
        ("Vert (Embrun, jan 2026)", ASSETS / "step11_mobile_vert.png"),
        ("Orange (Nice, 12 sept 2025)", ASSETS / "step11_mobile_orange.png"),
        ("Rouge (Nice, 11 sept 2025)", ASSETS / "step11_mobile_rouge.png"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    for ax, (title, path) in zip(axes, files, strict=True):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "Trois écrans d'application Taranis, mêmes canaux, trois régimes",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "step11_gallery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Écrit : {ASSETS / 'step11_gallery.png'}")


if __name__ == "__main__":
    main()
