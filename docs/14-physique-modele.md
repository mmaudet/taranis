# Étape 14, la physique du modèle Taranis

Une question revient assez naturellement en lisant les treize étapes précédentes : sur quelles **lois physiques** Taranis a-t-il été entraîné ? La réponse honnête et précise tient en une phrase :

> **Aucune loi physique n'est codée en dur dans TS-JEPA.** Le modèle apprend la structure statistique des séries multivariées via l'objectif JEPA, prédiction en espace latent d'une portion masquée à partir d'une portion visible. Ce qui a été fait, c'est **choisir les entrées et les étiquettes** de manière à ce que la physique se manifeste dans le signal.

Ce chapitre décompose la question en cinq couches : ce qui est injecté explicitement par les canaux, ce qui est appris implicitement par la structure JEPA, ce qui est appris via les étiquettes de la sonde aval, ce qui a servi de contrôle physique explicite, et ce qui n'a **pas** été utilisé. C'est un chapitre plus court que les autres, sans code nouveau, mais indispensable pour éviter les malentendus sur ce que Taranis sait et ne sait pas.

## 1. Ce qui est injecté par les canaux d'entrée

TS-JEPA voit **cinq canaux physiques** à chaque pas de temps, et rien de plus.

| Canal        | Grandeur physique              | Ce que le modèle peut y lire                          |
|---           |---                             |---                                                    |
| `pressure`   | pression au sol (hPa)          | équilibre synoptique, chute barométrique pré-orageuse |
| `temp`       | température (°C)               | cycle diurne, saisonnalité, refroidissement pré-frontal |
| `humidity`   | humidité relative (%)          | proximité de la saturation, montée convective         |
| `wind`       | vent moyen (m/s)               | régime advectif, front de rafale naissant             |
| `wind_gust`  | rafale sur 10 min (m/s)        | signature convective courte (front de rafale d'orage) |

Ces cinq canaux **imposent une physique**. Le modèle ne peut apprendre que ce qui vit dans ce sous-espace. Impossible pour lui de raisonner sur le CAPE (énergie potentielle convective disponible), la vorticité, le cisaillement vertical, faute de canaux correspondants. C'est une limite structurelle, assumée, et cohérente avec la contrainte du produit final : un capteur portable ne verra jamais ces variables non plus.

## 2. Ce que JEPA apprend implicitement

L'objectif JEPA force l'encodeur à trouver des **régularités persistantes** dans le signal, sans avoir jamais reçu la moindre étiquette. Sur nos canaux, cela signifie apprendre :

- **Le cycle diurne** (24 heures de période sur température et humidité), reconnu comme prévisible et donc encodable en peu de bits.
- **La respiration synoptique** (2 à 6 jours de période sur pression), même mécanisme.
- **Les corrélations classiques** (pression qui descend + humidité qui monte + vent qui gonfle = quelque chose se passe), sans qu'on lui dise quoi.
- **La dérivée temporelle** (chute barométrique franche sur 6 heures, montée d'humidité sur 12 heures), captée par l'attention transformer sur des positions de patches adjacentes.

Ce qui ne rentre pas dans ces régularités, le bruit du capteur ou les artefacts d'observation, est écrasé par le mécanisme même de la prédiction en espace latent. Pas la peine de prédire du bruit qu'on ne peut pas prédire, l'encodeur a intérêt à l'ignorer.

**C'est là que la « physique implicite » se joue**. TS-JEPA n'apprend pas les équations, il apprend à reconnaître les configurations où ces équations s'expriment nettement. Une chute de pression synoptique produit toujours le même type de rearrangement dans l'espace latent, quel que soit le mois de l'année ou la station.

## 3. Ce qui est appris via les étiquettes de la sonde aval

L'encodeur JEPA est **auto-supervisé**, il ne voit jamais de label. Seule la **sonde linéaire aval** voit les étiquettes de vérité orage, et c'est là que la physique observationnelle rentre :

- **Codes WMO 4677 du temps présent** (`ww` in {17, 29, 91-99}). L'orage tel qu'un observateur météo le rapporte au sens strict. Physique sous-jacente : décharge électrique observée, activité électrique nuage à air, précipitation avec orage.
- **Pluie horaire supérieure à 5 mm**. Proxy d'événement convectif intense. Physique sous-jacente : condensation rapide, cellule pluvieuse ponctuelle.

La sonde apprend, sur ces étiquettes, quelle **combinaison linéaire** des embeddings JEPA discrimine le mieux « orage à venir » vs « temps calme ». C'est à ce moment précis que l'apprentissage devient physique : la sonde sélectionne des directions latentes corrélées à la signature pré-orageuse observée.

## 4. Le contrôle physique explicite, la baseline M0

La **baseline M0** est purement physique. Régression logistique sur dix features fabriquées à la main :

- tendance de pression sur 1 heure et sur 3 heures,
- humidité courante, humidité moyenne sur 1 heure, delta d'humidité sur 1 heure,
- vent courant, rafale maximale sur 1 heure,
- température courante, amplitude thermique.

Elle sert de **plafond bas honnête**. Sur les treize orages historiques du panel d'évaluation, elle atteint AUC 0.72 avec des features explicites de :

- la **loi de Buys-Ballot**, chute de pression égale système dépressionnaire qui approche,
- la **thermodynamique convective**, humidité plus chaleur donne instabilité,
- l'**instabilité barométrique**, tendance sur 3 heures qui bascule.

TS-JEPA doit battre cette baseline pour justifier son existence. Avec les labels combinés WMO plus pluie forte, il la bat, 12/13 en ORANGE contre 6/13 pour M0, avec 24 heures de préavis médian.

**La physique est identique dans les deux modèles.** Elle vient des mêmes canaux d'entrée. La différence est dans la **façon de la représenter**, à la main pour M0, apprise par transformer pour M1.

## 5. Ce qui n'a pas été utilisé

Pour éviter les malentendus sur ce que Taranis sait faire, voici ce qui n'entre **pas** dans son entraînement :

- Aucune équation de **Navier-Stokes** ou de fluide en général.
- Aucune paramétrisation physique de la convection type Kain-Fritsch, Tiedtke ou autre.
- Aucune contrainte de **conservation** d'énergie ou d'humidité.
- Aucun **modèle atmosphérique** GCM ou LAM.
- Aucune donnée **radar** ou **satellite**.
- Aucun **profil vertical**, radiosondage ou réanalyse ERA5 en pression, pour le modèle actuel. ERA5 arrivera au niveau surface uniquement, en pré-entraînement massif à l'étape 8.

C'est un modèle **exclusivement local et de surface**, à horizon 24 heures, entraîné sur un signal multivarié brut. Sa force et sa limite viennent toutes les deux de là. Il ne remplacera jamais AROME ou un modèle physique tridimensionnel. Il peut, en revanche, tourner sur un téléphone à partir de mesures capteur locales, ce qu'aucun modèle physique complet ne peut faire.

## 6. Les conséquences pratiques

Trois implications directes de cette architecture, à garder en tête quand on utilise ou qu'on discute Taranis.

**Le modèle ne « comprend » pas l'orage.** Il reconnaît des configurations statistiques qui, dans le corpus d'entraînement, précèdent souvent un orage. Il peut se tromper sur un régime météorologique qu'il n'a jamais vu (climat de très haute altitude, DOM-TOM, situations exceptionnelles).

**Le modèle est très localisé.** Il ne voit ni le radar au-dessus de lui, ni le nuage à 20 km, ni l'advection prévue par AROME. Il ne voit que les cinq canaux à sa position. Ses prédictions sont donc **complémentaires** des bulletins officiels, pas concurrentes.

**Le modèle transfère mal aux régimes inconnus.** Un orage subtropical à Mayotte a une signature différente d'un orage cévenol. Le modèle actuel, entraîné sur 62 stations SYNOP France métropolitaine plus quelques DOM, aura besoin de fine-tuning localisé pour des zones climatiques très éloignées.

## Résumé, en une phrase

TS-JEPA n'a été entraîné sur **aucune** équation physique explicite. Il a été entraîné sur **cinq canaux physiques observationnels** (pression, température, humidité, vent, rafale) qui **portent la physique en eux**. La différence entre « physique en dur » et « physique portée par les données » est essentielle pour comprendre ce que Taranis fait, ce qu'il ne fait pas, et pourquoi il ne prétend pas remplacer un modèle météorologique complet.
