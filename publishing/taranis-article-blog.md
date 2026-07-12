---
title: "Taranis, apprendre à anticiper l'orage avec un modèle JEPA souverain"
date: 2026-07-13
pubDate: "13 juil. 2026"
category: "Analyse"
tags: ["souveraineté", "IA ouverte", "JEPA", "météo", "logiciel libre", "montagne"]
readingTime: "12 min"
excerpt: "Construire, brique par brique, un modèle TS-JEPA de nowcasting orage à partir de données Météo-France ouvertes. Un carnet pédagogique complet, une baseline honnête à battre, un verdict sans complaisance, et une bascule concrète du prototype au produit."
slug: "taranis-jepa-nowcasting-orage-souverain"
image: "/images/taranis/step1_zoom_orage.png"
---

## En une phrase

**Taranis** est un projet à la fois pédagogique et de recherche appliquée : maîtriser l'architecture **JEPA** (Joint Embedding Predictive Architecture) appliquée aux séries temporelles, en construisant pas à pas un **outil souverain d'alerte orage** à partir de mesures capteur locales.

Le carnet complet, seize étapes documentées, est publié en accès libre sur [maudet.cloud/taranis](/taranis/). Le code, sous licence **AGPL-3.0**, est disponible sur GitHub. Cet article résume la démarche, les résultats obtenus, et les leçons.

## Pourquoi TS-JEPA plutôt qu'un modèle plus classique

L'écosystème des séries temporelles météorologiques est saturé de modèles supervisés qui prédisent la précipitation à partir de features fabriquées. Ils marchent, mais ils dépendent de labels rares et de features souvent inadaptées aux régimes locaux.

**TS-JEPA** (Ennadir, Golkar, Sarra, 2025) est la première adaptation systématique de l'idée JEPA aux séries temporelles. L'idée cœur, en une phrase :

> Un modèle JEPA prend deux morceaux d'un même signal, les encode chacun, et entraîne un prédicteur à retrouver la représentation du morceau caché à partir de celle du morceau visible.

Trois propriétés font que cette famille est **particulièrement adaptée à Taranis** :

1. **Aucune étiquette au pré-entraînement.** Le signal capteur bruté suffit. Or les vraies étiquettes d'orage (foudre, code WMO) sont rares.
2. **Prédiction en espace latent, pas en valeurs brutes.** Le bruit du capteur, les cycles diurnes réguliers, les artefacts électroniques ne survivent pas à l'encodage. Ce qui reste est la structure physique.
3. **Sample-efficient.** Sur des benchmarks image et vidéo, JEPA rivalise avec les approches contrastives avec moins de calcul.

## Le carnet, seize étapes documentées

La progression est délibérément pédagogique : chaque brique est motivée avant d'être codée, chaque baseline est testée avant qu'on prétende faire mieux, chaque échec est documenté.

- **Étapes 1 à 4** : le problème, le dataset synthétique, la baseline physique honnête (M0, régression logistique sur 10 features), pourquoi apprendre des représentations, JEPA en une image.
- **Étapes 5 à 7** : TS-JEPA brique par brique en PyTorch, l'entraînement et la surveillance du collapse (le grand piège classique de la famille JEPA), le premier verdict croisé synthétique / réel.
- **Étape 8** : le passage à l'échelle avec ERA5 (Copernicus, souverain EU) en pré-entraînement et Météo-France SYNOP en fine-tuning.
- **Étapes 9 à 12** : enrichir les canaux, remplacer le proxy pluie par les vrais codes WMO d'orage (le fameux `ww`, 17, 29, 91-99), le serveur autonome FastAPI + Docker, l'application mobile HTML autonome.
- **Étape 13** : le **journal honnête de calibration**, où on documente comment la démo initiale sur Nice septembre 2025 (belle mais trompeuse) a laissé place à une vraie évaluation sur 13 orages tirés objectivement du SYNOP.

## Le pipeline, entièrement souverain

Tout ce qui alimente Taranis est ouvert et européen :

- **Données historiques** : SYNOP essentiel OMM via Opendatasoft (licence Etalab, 2.7 millions de records, 62 stations, 16 ans).
- **Réanalyse** : ERA5 via Copernicus Climate Data Store (Union Européenne, gratuit avec inscription).
- **Étiquettes d'orage** : union du code temps présent WMO 4677 et de la pluie horaire > 5 mm. Cette combinaison rattrape les orages diurnes courts qui échappent au SYNOP horaire.
- **Framework** : PyTorch (Meta, licence BSD), MkDocs Material, FastAPI, tout OSS.

Aucune dépendance à une API propriétaire non substituable. Le container Docker se déploie partout, y compris sur **Hugging Face Spaces** (autre acteur souverain EU), Fly.io ou un VPS OVH/Scaleway.

## Le verdict, sans complaisance

Après huit configurations testées sur un panel de 13 orages historiques tirés objectivement du SYNOP :

| Sonde | Détection ORANGE avant onset | Préavis médian |
|---|---|---|
| Proxy pluie forte seul | Illisible (proba plafonnée) | — |
| Code WMO seul (`ww_rich`) | 6 / 13 (46 %) | 25 h |
| **Combinaison WMO + pluie > 5 mm** | **12 / 13 (92 %)** | **24 h** |

La sonde production actuelle **détecte 12 orages sur 13 avec 24 heures de préavis médian**. C'est un résultat solide, obtenu en 90 secondes d'entraînement CPU, sur un modèle compact `d_model = 64`, sans GPU.

Les trois leçons du parcours :

1. Une bonne **AUC** ne suffit pas. Il faut regarder les probabilités absolues et le mapping vers l'alerte.
2. Une belle **démo sur un cas particulier** ne suffit pas. Il faut un panel diversifié tiré objectivement.
3. Une bonne **architecture** ne suffit pas non plus. La **qualité des labels** est souvent le vrai facteur limitant, avant le compute.

## Ce qui reste à faire

**Court terme** : les configurations GPU sont prêtes (`d_model = 128`, 6 couches, 50 000 pas). Le fetch ERA5 en cours (13 points, 15 ans, ~2340 requêtes) alimentera un pré-entraînement massif, avec l'objectif d'atteindre 12/13 détections ROUGE (pas seulement ORANGE) et un préavis > 6 heures fiables sur toutes les catégories d'orages.

**Moyen terme** : intégrer un capteur BLE porté (options identifiées : **Calypso Portable Mini** ultrasonique pour le vent, driver Python ouvert, plus **RuuviTag** pour température, humidité, pression). L'inférence tourne alors sur le téléphone, sans jamais interroger d'API publique. La règle produit est claire : **la mesure physique locale, pas la position GPS**.

**Long terme** : porter le modèle sur mobile via ONNX ou executorch, écrire une application native, intégrer la réception BLE des capteurs. C'est le lot 3 du PRD initial.

## Un carnet, un dépôt, une invitation

Le carnet complet est publié sous [maudet.cloud/taranis](/taranis/). Le dépôt est **libre**, licence AGPL-3.0. Toute contribution qui **améliore la qualité des labels** (Blitzortung foudre, radar Météo-France), **teste un régime climatique différent** (DOM-TOM, montagnes plus hautes, climat cévenol) ou **porte le modèle sur du hardware réel** est particulièrement bienvenue.

Un projet pédagogique n'a pas vocation à rester un joli PDF. Il doit **circuler**, être **critiqué**, et servir de **base à d'autres**. C'est pour ça que tout est documenté, testé, et prêt à être rejoué en quatre minutes sur n'importe quel portable Python 3.12.

---

*Michel-Marie Maudet · juillet 2026 · Le code source complet est disponible dans le [dépôt Taranis](https://github.com/mmaudet/taranis). Le carnet pédagogique complet est publié sur [maudet.cloud/taranis](/taranis/).*
