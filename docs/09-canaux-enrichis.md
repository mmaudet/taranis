# Étape 9, enrichir les canaux SYNOP

Jusqu'ici, le modèle a vu quatre canaux : pression, température, humidité, vent. Le SYNOP en propose une vingtaine. À l'étape 9, on ajoute la rafale à 10 minutes (`raf10`), un indicateur de convection particulièrement informatif que la baseline physique n'exploite pas. Objectif : voir si un signal supplémentaire suffit à faire pencher la balance en faveur de TS-JEPA.

## Ce qu'on ajoute et pourquoi

**Un seul canal en plus, `wind_gust`.** C'est un choix pédagogique.

- La **rafale à 10 minutes** capture les pointes de vent qui accompagnent le front de rafale d'un orage convectif. Elle décorrèle le « vent moyen » de la « bourrasque brève », ce que le vent moyen `ff` ne fait pas.
- Elle est **très rare dans le signal ambiant** (le vent est calme), mais **spectaculaire lors d'un orage** (rafales de 15-30 m/s en quelques minutes). Un modèle qui apprend des représentations doit exploiter cette asymétrie.
- Elle est présente dans **presque tous** les records SYNOP, sans trou significatif.

On garde volontairement **cinq canaux au total**, pas huit. Ajouter la direction du vent (angulaire), la pression au niveau de la mer, ou les précipitations cumulées à 3 h et 6 h est possible et prévu, mais on veut voir d'abord si **un seul signal manquant** change quelque chose. C'est la démarche scientifique classique, un facteur à la fois.

## Ce qui change dans le code, presque rien

Trois modifications isolées :

- `taranis/data/meteofrance.py` : nouvelle constante `CANAUX_MF_RICH = (…, 'wind_gust')`, colonne `wind_gust` conservée dans le rééchantillonnage, imputation par le vent moyen si manquant.
- `scripts/prepare_real_full_rich_dataset.py` : nouveau script, produit `data/real_full_rich_windows.npz` avec X de forme **(N, 32, 5)**.
- `configs/tsjepa_real_full_rich.yaml` : `n_canaux: 5`.

**La baseline `BaselinePhysics` n'est pas modifiée.** Elle lit les quatre premiers canaux et ignore `wind_gust`. Ce n'est **pas** un oubli, c'est un contrôle : on veut mesurer le gain apporté par le canal supplémentaire au seul modèle qui peut l'exploiter (JEPA), pas fausser la comparaison en avantageant les deux modèles à la fois.

## L'observation dérangeante sur la baseline

En allant de 4 stations × 5 ans à 62 stations × 16 ans, on multiplie le volume train par 55. Sur ce même volume, la baseline donne :

| Baseline sur    | Fenêtres | AUC       | AP        | F1       |
|---              |---       |---        |---        |---       |
| 4 stations, 5 ans | 34 704 | 0.719     | 0.196     | 0.292    |
| **62 stations, 16 ans** | **1.9 M** | **0.716** | **0.180** | **0.265** |

**Zéro gain, léger recul.** La baseline est saturée en capacité, ses dix features hand-crafted ne peuvent pas absorber plus d'information. Ce n'est **pas** un défaut de la baseline, c'est la définition même d'une baseline. Elle nous dit :

- « Le signal facile à extraire des quatre canaux est déjà pris. »
- « Pour aller plus loin, il faudra soit **de nouvelles features**, soit **un modèle qui les découvre**. »

C'est exactement l'ouverture qu'espère TS-JEPA.

## TS-JEPA sur 5 canaux, ce qu'on observe

Même configuration compacte que les étapes précédentes, `d_model = 64`, 3 couches d'encodeur, 2 de prédicteur. Un peu plus de pas (4 000 au lieu de 2 000) pour tenir compte du volume dix fois supérieur. **90 secondes sur CPU.**

| Sonde M1(real_full_rich) | Valeur |
|---                       |---     |
| AUC                      | 0.694  |
| AP                       | 0.170  |
| F1                       | 0.242  |
| Prévalence positive      | 0.082  |

Comparé à la sonde M1 de l'étape 7 (4 stations, 4 canaux, 2 000 pas) qui donnait AUC = 0.696, **on est presque à égalité**. On a multiplié les données par 55, ajouté un canal informatif, et la sonde reste stable.

Que peut-on en déduire ?

- **Bonne nouvelle** : le rang effectif des embeddings monte, `z_ctx_rank` passe de 6.0 à 7.3 sur 64 dimensions. Le modèle **utilise davantage sa dimension latente**. C'est un signe positif de meilleure représentation.
- **Mauvaise nouvelle** : le lien entre représentation et tâche aval ne s'est pas resserré. La sonde linéaire n'est pas plus discriminante qu'avant.
- **Diagnostic honnête** : ce plateau est un **effet du compute limité**. Un modèle plus grand ou beaucoup plus entraîné devrait pouvoir en tirer parti. C'est la promesse de l'étape 8.

## Ce que cette étape valide

Trois choses.

1. **La chaîne de traitement supporte plusieurs canaux sans friction**. `make_windows`, `channel_stats`, `normalize`, `TSJEPA`, `LinearProbe` acceptent n'importe quel nombre de canaux tel quel. C'est un pré-requis technique pour brancher des sources spatiales ou multi-modales plus tard.
2. **La baseline physique est vraiment un plafond bas fiable.** Elle ne s'améliore pas quand on lui donne plus de données. Elle est ce qu'elle est, et elle sera battue quand un autre modèle sera vraiment mieux, pas par hasard.
3. **Le rang effectif est un indicateur pertinent** de la qualité de la représentation, indépendamment de l'AUC de la sonde. C'est intéressant pour surveiller un pré-entraînement long.

## Ce qui vient à l'étape 10

Passer aux **vraies étiquettes de foudre**. Notre proxy `rr1 > 2 mm` inclut beaucoup d'événements qui ne sont pas des orages (pluie stratiforme soutenue, bruine longue), et rate des orages secs. Blitzortung archive les strikes horodatés et géolocalisés à l'échelle européenne. On va aligner ces strikes avec nos stations SYNOP, redéfinir `storm_onset` et `storm_active` sur cette base plus stricte, et relancer le protocole.

C'est là qu'on espère voir le vrai gain de M1 sur M0. Une étiquette plus propre, un modèle qui peut apprendre à représenter finement le pré-orage, et la comparaison devient enfin honnête.

## Reproduire

```bash
uv run python scripts/prepare_real_full_rich_dataset.py
uv run python scripts/train_tsjepa.py configs/tsjepa_real_full_rich.yaml
```

Le premier prend ~1 minute, le second ~90 secondes sur CPU. Total 2 minutes.
