# Étape 10, de vrais labels d'orage

À l'étape 9, on a enrichi les canaux et vu la baseline saturer. Le verdict était clair, si TS-JEPA n'apprend pas à dépasser M0, ce n'est pas parce que les représentations sont mauvaises, c'est parce que la **tâche** est mal posée. Le proxy « pluie forte > 2 mm/h » n'est pas de l'orage, c'est un mélange qui inclut beaucoup d'événements non-convectifs et rate les orages secs.

Cette étape corrige la définition du positif. On passe d'un proxy à une **observation directe d'orage**, au sens WMO. Et là, tout change.

## Ce qui existe dans SYNOP, qu'on n'utilisait pas

Chaque record SYNOP contient un champ **`ww`, le code temps présent**, standardisé par l'OMM dans la table 4677 :

- **17** : orage sans précipitation à la station.
- **29** : orage récent, dans la dernière heure, avec ou sans précipitation.
- **91-94** : pluie légère à forte avec orage récent.
- **95** : orage léger ou modéré à l'observation.
- **97** : orage fort avec pluie.
- **99** : orage fort avec grêle.

Ces codes sont **rentrés par un observateur** ou décodés d'un capteur foudre local. Ils ne sont donc pas un mélange orage/pluie stratiforme comme le proxy `rr1 > 2 mm`, ils désignent des orages au sens strict.

**Sur 2 050 001 records valides des 62 stations, on trouve 3 712 observations d'orage** direct (0.18 %). Réparties sur 46 stations différentes, en concentration cohérente : Lyon, Nice, Bâle-Mulhouse, Bastia, Ajaccio arrivent en tête, ce qui correspond à ce qu'un climatologue attendrait.

## Pourquoi c'est mieux qu'un proxy foudre externe

On aurait pu utiliser **Blitzortung**, la référence mondiale du strike foudre. On a examiné la piste. Trois freins :

- L'accès aux archives est **peu documenté**, il faut contacter l'équipe.
- Le format et la licence ne sont pas triviaux à intégrer.
- L'alignement spatial (strike → station) demande un travail supplémentaire.

Or le code `ww` du SYNOP **est déjà à la station**. Un orage passe sur la station à l'instant t, le code est écrit à l'instant t. **Aucun alignement à faire**. C'est de la vérité terrain locale, exactement ce qu'un capteur portable produirait s'il détectait localement.

Blitzortung reste une piste future pour, par exemple, valider des orages à distance ou densifier les stations muettes, mais pour un premier passage aux vraies étiquettes, le code `ww` est **plus honnête et plus simple**.

## Le nouveau label, en trois lignes

```python
CODES_ORAGE_WMO = (17, 29, 91, 92, 93, 94, 95, 96, 97, 98, 99)

is_storm = ww.isin(CODES_ORAGE_WMO)
# extension : orage actif ±1 pas autour de l'observation (±3h)
```

L'extension `±1 pas` (soit ±3 h au pas SYNOP) sert à représenter la durée typique d'un orage : quand un observateur voit un orage à 15h, il y avait probablement quelque chose à 12h et il en restera trace à 18h. Ce n'est pas rigoureux mais c'est raisonnable.

L'onset est le premier pas d'une plage active. La tâche reste la même, prédire un onset dans l'horizon `H = 8 pas` (24 h).

## Volumétrie du nouveau dataset

Fichier `data/real_full_ww_windows.npz` :

| Split | Fenêtres | Onsets positifs | Prévalence |
|---    |---       |---              |---         |
| Train | 1 896 324 | ~18 600      | **0.98 %** |
| Val   | 334 227   | ~1 000       | 0.30 %     |
| Test  | 336 725   | ~1 100       | **0.33 %** |

**~25 fois plus rare** que le proxy pluie. Le dataset est **très déséquilibré**. La conséquence est une chute mécanique de l'Average Precision, comme on va voir tout de suite.

Note : la prévalence est plus élevée sur le train (2010-2021) que sur val/test (2022-2024). Pourquoi ? Parce que les observateurs humains rapportent moins de codes ww précis dans les stations récentes automatisées. Un décalage de distribution qu'il vaut mieux voir tout de suite. Cela reflète la réalité, on ne le corrige pas artificiellement.

## Le résultat, sans complaisance

Même baseline M0 (10 features physiques, régression logistique standardisée). Même encodeur M1 (checkpoint de l'étape 9, non modifié). Seule la définition du label change. La sonde linéaire de M1 est réentraînée sur les nouveaux labels.

| Métrique     | Baseline M0 | Sonde M1 (encodeur étape 9) | Δ                |
|---           |---          |---                           |---               |
| **AUC**      | 0.711       | **0.870**                    | **+16 points**   |
| Average Precision | 0.010  | 0.036                        | ×3.6             |
| Précision    | 0.015       | 0.063                        | ×4.2             |
| Rappel       | 0.184       | 0.149                        | -0.03            |
| F1           | 0.028       | 0.089                        | ×3.2             |
| Prévalence   | 0.003       | 0.003                        | (identique)      |

C'est **le vrai verdict**. M1 dépasse M0 de **seize points d'AUC**. Toutes les autres métriques suivent dans le bon sens, à l'exception d'un rappel légèrement plus bas (que le seuil F1-maximisation cache, un choix de seuil plus « prudent » sur la baseline compense).

**C'est le moment de la surprise**. Pendant sept étapes on a écrit et re-écrit « M0 domine », en croyant que le modèle JEPA n'apprenait rien d'utile. En fait, il apprenait, mais on lui demandait de classer un signal bruité (pluie forte de toute origine). Quand on lui demande de classer l'orage au sens observationnel, **son avance est massive**.

## Ce que ce résultat dit vraiment

1. **La qualité des labels dépasse en importance la taille du modèle** dans notre régime. Passer de proxy pluie à observation WMO change tout, avec un modèle identique et un compute identique.
2. **JEPA a extrait des features fines** que la baseline ne pouvait pas fabriquer à la main. La chute barométrique n'est qu'un signal parmi d'autres ; le modèle a construit une combinaison non linéaire des 5 canaux qui discrimine mieux.
3. **La baseline a plafonné**. Sur cette tâche plus stricte, elle reste à AUC 0.71 alors que M1 monte à 0.87. Écrire de nouvelles features à la main ne l'aurait probablement pas fait monter beaucoup plus, on aurait vite touché la limite de ce qu'un humain peut deviner à la main.

## Une nuance importante sur la précision au seuil

L'AUC de 0.87 est excellente. La précision de 0.063 est basse. Ce n'est **pas** une contradiction, c'est une conséquence directe de la prévalence de 0.3 %.

Un exemple : si le modèle donne un score au-dessus du seuil à 3 % des fenêtres, et que la moitié des vrais positifs sont dans ces 3 %, alors :
- rappel = 50 %,
- précision = (0.5 × 0.003) / 0.03 ≈ 5 %.

C'est exactement le régime qu'on observe. Pour un usage sécurité en montagne, on peut choisir un seuil plus permissif qui monte le rappel à 80 % au prix d'une précision plus basse encore. C'est un compromis explicite, à définir avec l'utilisateur final.

## Ce qu'on peut faire maintenant

Trois pistes s'ouvrent, dans l'ordre de rentabilité :

1. **Long run GPU sur ces mêmes labels**. Avec plus de compute, on peut viser AUC > 0.90 et un rappel significativement plus élevé au même seuil. C'est le protocole de l'étape 8, mais maintenant justifié par un résultat concret.
2. **Analyse par station et par saison**. Les orages ne sont pas répartis uniformément (Corse et Sud > Nord et côte Atlantique, été > hiver). Un modèle par région serait probablement meilleur.
3. **Analyse du lead time**. À quel point d'avance le modèle voit-il un orage arriver ? On sait juste qu'il classe bien, on ne sait pas encore comment se distribue le temps de préavis. C'est **le vrai chiffre** pour un usage sécurité montagne.

## Reproduire

```bash
# 1. Régénérer le dataset avec labels WMO (utilise le CSV téléchargé à l'étape 8)
uv run python scripts/prepare_real_ww_dataset.py

# 2. La baseline se lance en une commande adaptée (voir scripts/train_baseline.py
#    pour synth, ou intégrer dans evaluate_all.py)
# 3. La sonde M1 réutilise l'encodeur de l'étape 9, il n'y a pas besoin de
#    ré-entraîner TS-JEPA. Seule la sonde change.
```

## Ce qu'il faut retenir

1. **Le proxy pluie masquait la valeur de M1.** Sur des labels stricts, JEPA passe de 0.70 AUC à 0.87.
2. **La qualité des labels est le vrai levier** dans notre régime compute-limité.
3. **La baseline a atteint son plafond structurel**, indépendamment de la quantité de données.
4. Le compromis précision/rappel dépend de la prévalence. À 0.3 % de positifs, on ne peut pas avoir précision et rappel élevés simultanément, on choisit consciemment le point de fonctionnement.
5. **Ce résultat justifie enfin** le passage à l'échelle GPU de l'étape 8. On sait maintenant qu'on court après quelque chose de réel.
