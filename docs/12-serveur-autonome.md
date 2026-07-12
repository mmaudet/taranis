# Étape 12, le serveur autonome

À l'étape 11 on a spécifié un contrat API et rendu des mockups statiques. À l'étape 12 on assemble : le **serveur FastAPI sert lui-même l'écran mobile**, le modèle tourne au moment de la requête, chaque appel renvoie une prédiction fraîche sur les vraies dernières mesures SYNOP. Un seul processus, un seul port, un seul container Docker. Prêt à déployer partout.

## Ce qui change par rapport à l'étape 11

À l'étape 11 le HTML mobile était **pré-calculé** hors ligne, chaque écran figeait une prédiction. À l'étape 12, le HTML est **servi par le backend**, et il appelle `/stations/{id}/live` à chaque changement de station. Le modèle recalcule au moment de la demande, sur la fenêtre effectivement disponible dans SYNOP.

**Trois conséquences :**

- l'utilisateur peut choisir **n'importe quelle station** SYNOP dans le sélecteur, pas seulement les scénarios pré-embarqués ;
- l'utilisateur peut **rafraîchir** pour prendre la mesure la plus récente sans redéployer ;
- une seule sonde à mettre à jour (variable d'environnement `TARANIS_PROBE`) pour changer de modèle sous-jacent.

## L'architecture, en une image

```
    ┌───────────────────────────────────────────────────────────────┐
    │  navigateur mobile                                            │
    │  ┌────────────────────────────────────────────┐               │
    │  │ index.html (mobile-first, vanilla JS)      │◄─── GET /     │
    │  │ - sélecteur de station                     │               │
    │  │ - fetch('/stations/{id}/live?...')         │               │
    │  │ - rend bandeau d'alerte + sparklines SVG   │               │
    │  └────────────────────────────────────────────┘               │
    │                        │                                      │
    └────────────────────────┼──────────────────────────────────────┘
                             │  HTTPS
                             ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  Serveur Taranis (FastAPI + uvicorn, Python 3.12)             │
    │  ─────────────────────────────────────────────────────────    │
    │  GET  /                            → index.html               │
    │  GET  /health                      → statut + config modèle   │
    │  POST /predict                     → prédiction sur payload   │
    │  GET  /stations/{id}/live?with_window=true                    │
    │         │                                                     │
    │         ├─► requests.get(Opendatasoft SYNOP CSV live)         │
    │         ├─► resample 3h + prepare_station                     │
    │         ├─► Predictor.predict_from_raw(window)                │
    │         └─► JSON prêt à afficher                              │
    └───────────────────────────────────────────────────────────────┘
                             │
                             ▼
                 Opendatasoft (Météo-France SYNOP)
                 public, sans clé, licence Etalab
```

Aucune base de données, aucun cache persistant. Le serveur est **stateless**. On peut le mettre à l'échelle horizontalement sans coordination.

## Lancer en local, une commande

```bash
uv sync --extra api
TARANIS_PROBE=runs/probe/ww_rich \
    uv run uvicorn taranis.infer.api:app --host 0.0.0.0 --port 8000
```

Puis ouvrir `http://localhost:8000` dans un navigateur (mobile ou desktop). La page HTML se charge, le sélecteur est peuplé de 16 stations françaises couvrant montagne, plaine et côte. Le premier chargement déclenche automatiquement une prédiction sur EMBRUN, ensuite chaque changement de station relance un cycle.

Pour exposer le serveur au reste du réseau local :

```bash
uv run uvicorn taranis.infer.api:app --host 0.0.0.0 --port 8000
# accessible sur http://<ip-lan>:8000/ depuis un mobile sur le même réseau Wi-Fi
```

## Le container Docker, prêt à déployer

Le `Dockerfile` livré à la racine du dépôt s'appuie sur `python:3.12-slim` et embarque uv pour installer les dépendances rapidement.

```bash
docker compose up --build
# puis http://localhost:8000
```

L'image finale contient :

- le paquet Python `taranis` (data, models, eval, train, infer) ;
- la sonde entraînée (`runs/probe/ww_rich`) ;
- l'UI HTML (`taranis/infer/static/index.html`) ;
- uvicorn + fastapi + torch CPU-only.

**Taille indicative** : ~1.8 Go (dominée par torch CPU wheel). Une variante ONNX ferait chuter à ~200 Mo, prévu pour l'après.

## Options d'hébergement

Le container tourne partout où Docker tourne. Trois pistes concrètes :

- **Hugging Face Spaces** (recommandé pour prototype public souverain) : simple push d'un `Dockerfile`, HTTPS auto, quota CPU gratuit. URL publique permanente.
- **Fly.io** : `fly launch` détecte le Dockerfile. Gratuit jusqu'à 3 shared-cpu-1x machines. Régions FR ou Amsterdam disponibles.
- **VPS OVH ou Scaleway** : n'importe quel VPS 1 vCPU / 2 Go RAM suffit. Reverse-proxy Caddy pour HTTPS automatique.

Chacune de ces trois cibles utilise **exactement le même container**. Le contrat API est stable, ce qui isole le front d'un éventuel changement d'hébergeur.

## Ce qui reste à faire

- **HTTPS et domaine propre** en production (délégué au reverse-proxy de l'hébergeur).
- **Rate-limiting** léger si l'API est publique (protection contre abus, quotas Opendatasoft).
- **Auto-refresh périodique** côté client (toutes les 30 minutes par exemple) pour une expérience « app permanente ».
- **PWA** : ajouter `manifest.json` et un service worker pour permettre l'installation sur écran d'accueil du mobile.
- **Quantification et export ONNX** de l'encodeur, pour ramener le container sous les 300 Mo et préparer l'exécution locale sur mobile natif.

## Reproduire, tout de bout en bout

```bash
# 1. Sonde entraînée (déjà présente via l'étape 10)
uv run python scripts/save_probe.py \
    --encoder runs/tsjepa_real_full_rich \
    --dataset data/real_full_ww_windows.npz \
    --out runs/probe/ww_rich

# 2. Local
uv sync --extra api
TARANIS_PROBE=runs/probe/ww_rich \
    uv run uvicorn taranis.infer.api:app --host 0.0.0.0 --port 8000

# 3. Container
docker compose up --build

# 4. Push HF Space
# (créer un Space avec SDK "docker" et pousser le dépôt)
```

## Ce qu'il faut retenir

1. Le serveur est **autonome et stateless**. Un container, un port, aucune base.
2. L'UI mobile est **servie par le backend**, avec des appels live à chaque interaction.
3. Le modèle est **rechargé à chaud** au démarrage via la variable `TARANIS_PROBE`, sans changement de code.
4. Le container tourne partout où Docker tourne, y compris sur des offres gratuites souveraines.
