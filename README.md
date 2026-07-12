# Taranis

Système souverain d'alerte météo par JEPA. Cœur pédagogique : construire pas à pas un modèle **TS-JEPA** pour anticiper les orages à partir d'un flux capteur local.

Ce dépôt privilégie la compréhension. Chaque brique du modèle est expliquée avant d'être codée, et chaque étape est un jalon documenté sur le site de documentation.

## Statut

Version 0.2, en cours d'écriture. Lot 0 (socle) en construction.

- Lot 0 : socle projet, générateur synthétique, baseline physique.
- Lot 1 : TS-JEPA de bout en bout sur synthétique puis données réelles Météo-France.
- Lots 2 à 4 : ingestion temps réel, capteur ESP32, application mobile. Reportés.

## Démarrage rapide

```bash
uv sync --extra dev --extra docs
uv run pytest
uv run mkdocs serve
```

Le site pédagogique s'ouvre alors sur `http://127.0.0.1:8000`.

## Licence

AGPL-3.0-or-later. Voir `LICENSE`.

## Références

- Ennadir, Golkar, Sarra. *Joint Embeddings Go Temporal (TS-JEPA)*. arXiv:2509.25449, 2025.
- Assran et al. *I-JEPA*. CVPR 2023.
- Bardes et al. *V-JEPA*. TMLR 2024.
