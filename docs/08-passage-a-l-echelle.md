# Étape 8, passage à l'échelle

Les sept premières étapes ont produit un pipeline **complet et honnête**, mais entraîné en configuration compacte sur CPU. À l'étape 7, la baseline physique bat encore TS-JEPA. C'est acceptable, c'est prévu, ce n'est pas satisfaisant. Pour changer ce verdict, il n'y a que trois leviers, et ils sont tous à activer simultanément :

1. **Beaucoup plus de données**, notamment en pré-entraînement non supervisé.
2. **Un modèle plus grand**, capable d'exploiter ces données.
3. **Un ordre de grandeur de calcul en plus**, ce qui exige GPU.

Cette étape documente le passage à l'échelle. Elle est **prête à lancer**, dès que tu disposes d'un GPU et d'un compte CDS Copernicus.

## Le protocole cible

C'est le protocole classique de la recherche récente en auto-supervisé :

1. **Pré-entraîner** TS-JEPA sur un vaste corpus **non annoté** de séries météorologiques cohérentes physiquement. On utilise ici la réanalyse ERA5, qui est un modèle numérique de l'atmosphère assimilé à des observations, disponible sur plusieurs décennies avec une qualité homogène.
2. **Fine-tuner** (ou brancher une sonde) sur le corpus **annoté** de vraies observations SYNOP, où la tâche aval est définie via un proxy de pluie forte.

Cette séparation est importante :

- ERA5 apporte le **volume** (des millions d'observations, sans manquants, sans capteurs défectueux).
- SYNOP apporte la **cible** (observations physiques réelles, telles qu'un capteur portable les verrait, avec le bruit associé).

## Volumétrie visée

En sortie de fetch, avec 10 points de grille sur 15 ans et 62 stations SYNOP sur 16 ans :

| Corpus         | Records bruts | Fenêtres après fenêtrage | Usage         |
|---             |---            |---                       |---            |
| ERA5           | ~1.3 M        | ~1.3 M                   | Pré-entraînement JEPA |
| SYNOP full     | 2.7 M         | **1.9 M** train + 0.34 M val + 0.34 M test | Fine-tuning + évaluation |

Total pré-entraînement disponible en cumul : **> 3 millions d'exemples**. C'est un ordre de grandeur au-dessus de nos runs CPU actuels (36 691 fenêtres train).

## L'infrastructure de dataset

**Ce qui est déjà fait, tu peux le rejouer maintenant** :

```bash
# 62 stations Météo-France, 16 ans, ~400 Mo de CSV puis ~50 Mo NPZ
uv run python scripts/fetch_meteofrance_full.py
uv run python scripts/prepare_real_full_dataset.py
```

Sortie : `data/real_full_windows.npz`, prêt à charger dans le trainer.

**Ce qui te demande de préparer un token CDS** :

```bash
# 1. Compte gratuit
open https://cds.climate.copernicus.eu/

# 2. Token et .cdsapirc
open https://cds.climate.copernicus.eu/how-to-api

# 3. Accepter les conditions du dataset reanalysis-era5-single-levels sur sa fiche

# 4. Installer les deps ERA5
uv sync --extra era5

# 5. Télécharger (plusieurs heures, files d'attente CDS)
uv run python scripts/fetch_era5.py

# 6. Préparer le dataset de pré-entraînement
uv run python scripts/prepare_era5_dataset.py
```

Sortie : `data/era5_pretrain_windows.npz`.

Les scripts sont écrits, testés, prêts à lancer. Ils utilisent les mêmes conventions que le reste du pipeline, donc **tout le code TS-JEPA fonctionne sans modification** sur ces nouveaux datasets.

## Les configs GPU

Deux configurations neuves, prêtes à lancer sur toute carte CUDA (L40S, A100, H100, DGX Spark GB10, etc.) :

- `configs/tsjepa_era5_gpu.yaml` : pré-entraînement massif ERA5.
- `configs/tsjepa_synop_full_gpu.yaml` : entraînement direct sur SYNOP full ou fine-tuning depuis un checkpoint ERA5.

Comparaison rapide avec les configs CPU actuelles :

| Paramètre       | CPU (actuel) | GPU (cible)    |
|---              |---           |---             |
| `d_model`       | 64           | **128**        |
| Couches encodeur | 3           | **6**          |
| Couches prédicteur | 2         | **4**          |
| Têtes           | 4            | **8**          |
| MLP ratio       | 2.0          | **4.0**        |
| Batch size      | 128          | **512**        |
| Pas d'entraînement | 2 000     | **30 000 à 50 000** |
| LR max          | 1e-3         | 5e-4           |
| Warmup          | 100 pas      | 1 500 à 2 000 pas |
| Weight decay    | 1e-4         | **5e-2**       |
| tau EMA fin     | 0.999        | **0.9999**     |

**Ordre de grandeur** :

- Pré-entraînement ERA5 (`50 000 pas × batch 512`) : ~30 M échantillons vus. Sur L40S, comptez **1 à 2 heures**.
- Fine-tuning SYNOP full (`30 000 pas × batch 512`) : ~15 M échantillons vus. Comptez **30 à 60 minutes**.

## La commande de lancement, en tout et pour tout

Une fois les datasets prêts :

```bash
# 1. Pré-entraînement massif ERA5 (donne un encodeur générique)
uv run python scripts/train_tsjepa.py configs/tsjepa_era5_gpu.yaml

# 2. Entraînement / fine-tuning sur SYNOP full
uv run python scripts/train_tsjepa.py configs/tsjepa_synop_full_gpu.yaml

# 3. Sondes et évaluation croisée, comme à l'étape 7
uv run python scripts/evaluate_all.py
```

Le trainer bascule automatiquement sur GPU si `device: cuda` est demandé et que CUDA est disponible. Pas d'autre changement de code.

## Ce qu'on attend de ce passage à l'échelle

Trois résultats sont plausibles, il faut mesurer pour savoir lequel se produit :

- **Scénario 1, TS-JEPA bat la baseline**. AUC dépasse 0.80 sur SYNOP full test, l'encodeur ERA5 pré-entraîné apporte un gain net en transfert. C'est le scénario espéré, qui justifierait de passer à la suite (fusion multi-source, capteur physique).
- **Scénario 2, TS-JEPA rejoint la baseline sans la battre**. AUC autour de 0.72-0.75, ce qui prouve que la mécanique tient mais que la tâche pluie proxy plafonne rapidement. Il faudrait alors changer la source de labels (Blitzortung foudre) plutôt que le modèle.
- **Scénario 3, TS-JEPA reste en dessous**. Signal que le proxy pluie n'est pas la bonne cible pour un encodeur JEPA, ou que la fusion multi-source (radar, satellite) est indispensable. On documente et on pivote.

Dans tous les cas, ce qu'on **acquiert** :

- Un **encodeur générique ERA5** réutilisable pour toute tâche météo aval en France.
- Un **protocole reproductible** pour toute nouvelle source de données.
- Un **coût de calcul mesuré** qui permet de dimensionner la production.

## Ce qui reste hors périmètre à ce stade

Même avec ce passage à l'échelle, deux choses ne seront **pas** faites :

1. **La fusion spatiale** avec des données radar (Météo-France MFR) ou satellite (Meteosat MTG). C'est le prochain grand chantier, prévu comme évolution du PRD.
2. **Le déploiement embarqué** sur ESP32 + application mobile. Reste du ressort des lots 3 et 4 du PRD initial.

## Récapitulatif exécutable

Le tableau ci-dessous liste toutes les commandes de l'étape 8, dans l'ordre. Elles sont **indépendantes**, tu peux relancer chacune sans repartir de zéro.

| # | Commande                                         | Durée      | Prérequis    |
|---|---                                                |---         |---           |
| 1 | `uv run python scripts/fetch_meteofrance_full.py` | ~2 min     | Réseau       |
| 2 | `uv run python scripts/prepare_real_full_dataset.py` | ~1 min  | Étape 1      |
| 3 | `uv sync --extra era5`                            | ~30 s      | Aucun        |
| 4 | `uv run python scripts/fetch_era5.py`             | **plusieurs heures** | Compte CDS + token |
| 5 | `uv run python scripts/prepare_era5_dataset.py`   | ~2 min     | Étape 4      |
| 6 | `uv run python scripts/train_tsjepa.py configs/tsjepa_era5_gpu.yaml` | ~1-2 h GPU | Étape 5 + GPU |
| 7 | `uv run python scripts/train_tsjepa.py configs/tsjepa_synop_full_gpu.yaml` | ~30 min GPU | Étape 2 + GPU |
| 8 | `uv run python scripts/evaluate_all.py`           | ~1 min GPU | Étapes 6+7   |

Étapes 1 et 2 déjà faites dans le dépôt. Étapes 3 à 5 : à toi de préparer le token, ensuite j'ai tout le code sous la main. Étapes 6 à 8 : dès qu'il y a un GPU.
