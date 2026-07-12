# PRD Taranis, système souverain d'alerte météo par JEPA

Version 0.2
Auteur : Michel-Marie Maudet
Destinataire : Claude Code (agent de développement)
Licence cible : AGPL-3.0
Statut : brouillon pour implémentation

> Note de nommage : « Taranis » (dieu gaulois du tonnerre) est un titre de travail.
> Collision connue avec l'émetteur RC FrSky Taranis. Alternatives : Bélénos, Meteia,
> Gwall (breton, « tempête »). À trancher avant le premier commit public.

---

## 1. Intention et contexte

Taranis est un projet à la fois pédagogique et de recherche appliquée. Objectif double :

1. Comprendre et maîtriser l'architecture JEPA (Joint Embedding Predictive Architecture) appliquée aux séries temporelles, via TS-JEPA.
2. Produire une brique logicielle souveraine capable d'anticiper un événement météorologique dangereux (orage convectif principalement) à partir d'un flux de capteurs local, en montagne.

Le projet assume dès le départ une posture honnête : un capteur ponctuel donne un signal local et à court préavis, pas une prévision spatiale. La cible réaliste est le nowcasting et l'alerte précoce, jamais une garantie. Voir section 11.

## 2. Vision produit, le dispositif Taranis

La cible finale est un dispositif portable connecté en Bluetooth, pensé pour la montagne, qui récupère en temps réel les informations météorologiques locales (température, pression atmosphérique, vitesse du vent, taux d'humidité, et dérivées) afin de prémunir l'utilisateur des orages et autres effets météorologiques indésirables ou dangereux.

Chaîne complète :

- **Capteur portable** : boîtier léger et autonome porté par le randonneur ou fixé au sac. Base matérielle de référence : microcontrôleur ESP32 (BLE intégré), capteur BME280 ou BME688 (température, pression, humidité), petit anémomètre pour la vitesse du vent. Batterie et boîtier résistant.
- **Liaison Bluetooth Low Energy** : le capteur expose un service GATT diffusant les mesures à cadence régulière vers le smartphone.
- **Application mobile** : reçoit le flux BLE, alimente le modèle TS-JEPA (inférence embarquée ou via endpoint souverain), et affiche l'état météorologique local, le régime détecté et le niveau d'alerte.
- **Fonction d'alerte** : anticiper l'apparition d'un orage ou d'un régime dangereux et prévenir l'utilisateur suffisamment tôt pour agir, avec un niveau d'alerte clair (vert, orange, rouge).

Le présent dépôt couvre le cerveau logiciel (données, modèle, évaluation, API d'inférence) et la spécification d'ingestion BLE. Le firmware capteur et l'application mobile sont spécifiés ici mais peuvent être développés dans des dépôts dédiés (voir lots 3 et 4).

## 3. Décision d'architecture (ADR-001)

### Contexte

Le cœur prédictif s'appuie sur TS-JEPA (Ennadir et al., 2025, arXiv:2509.25449), première adaptation systématique de JEPA aux séries temporelles : apprentissage de représentation auto-supervisé, découpage en patches, fort taux de masquage, transformers légers.

### Décision

TS-JEPA est retenu comme unique architecture JEPA du projet. Un prototype fonctionnel existe déjà (`baselines/tsjepa_min.py`).

Justification :
- Simple, documenté, déjà implémenté et entraîné.
- Sample-efficient, ce qui compte vu la rareté attendue des événements orageux locaux.
- Le pré-entraînement auto-supervisé sur beaucoup de données non labellisées, puis un fine-tuning léger sur les rares événements locaux, est le bon schéma pour ce problème.

### Baseline honnête

Le projet conserve en permanence une baseline physique (M0) fondée sur la tendance barométrique. Tant qu'un modèle JEPA ne bat pas cette baseline, on le documente sans se raconter d'histoire. La baseline reste dans le dépôt et dans chaque rapport d'évaluation.

## 4. Objectifs et non-objectifs

### Objectifs

- Pipeline de données reproductible : synthétique (déjà disponible), puis réel (Météo-France données ouvertes, Netatmo, réanalyse ERA5 sur un point).
- Étiquetage des événements dangereux à partir d'une source foudre ou radar (Blitzortung, réseau ouvert).
- Deux modèles comparables sur le même harnais : M0 baseline physique, M1 TS-JEPA.
- Évaluation robuste au déséquilibre (AUC, F1, précision, rappel) et orientée sécurité (lead time, calibration, courbe précision-rappel).
- Inférence légère, exécutable sur CPU puis sur GPU souverain (OVH L40S, DGX Spark GB10), et suffisamment compacte pour viser l'embarqué mobile.
- Spécification d'ingestion Bluetooth (service GATT, format des trames) permettant de brancher le capteur portable.

### Non-objectifs (hors périmètre de la v0.2)

- Prévision spatiale ou assimilation de données radar/satellite (réservé à une évolution de fusion).
- Fabrication matérielle du capteur (le PRD fournit la spécification de référence, pas la production).

## 5. Formulation de la tâche

- Un flux multivarié `x` de forme `(T, V)`, V canaux (pression, température, humidité, vent, plus dérivées comme la tendance de pression).
- Découpage en fenêtres de longueur `Tw`.
- Tâche : à partir de la fenêtre courante, prédire l'apparition d'un orage dans un horizon `H` suivant la fenêtre.
- Métrique de sécurité additionnelle : lead time, temps entre l'alerte et l'onset réel.

Le pré-entraînement JEPA est auto-supervisé (aucun label). Les labels ne servent qu'à la sonde aval et à l'évaluation.

## 6. Architecture technique

### 6.1 Pipeline de données (`taranis/data/`)

- `synthetic.py` : générateur météo synthétique avec régimes pré-orage (déjà prototypé). Sert aux tests et à la CI.
- `meteofrance.py` : chargeur des données horaires ouvertes Météo-France (stations). Normalisation par canal.
- `netatmo.py` : chargeur optionnel du réseau de stations personnelles (API), pour densifier le pré-entraînement.
- `era5.py` : extraction d'un point de réanalyse ERA5 (via cdsapi), pour le pré-entraînement massif.
- `labels_lightning.py` : étiquetage des fenêtres « danger » à partir d'archives foudre (Blitzortung) ou radar.
- `windows.py` : fenêtrage, split chronologique, construction des exemples et labels.

Contrainte : split strictement chronologique (jamais aléatoire sur séries temporelles). Pré-entraînement sans label ; aval avec seuil de détection choisi sur validation puis figé sur test.

### 6.2 Ingestion capteur (`taranis/ingest/`)

- `ble.py` : client BLE qui s'abonne au service GATT du capteur, décode les trames (température, pression, humidité, vent), horodate et normalise le pas de temps.
- `gatt_spec.md` : spécification du service GATT (UUID, caractéristiques, format binaire des mesures, cadence). Sert de contrat entre firmware et application.
- `buffer.py` : fenêtre glissante en mémoire alimentant l'inférence en continu.

### 6.3 Modèles (`taranis/models/`)

**M0 baseline physique (`baseline_physics.py`)**
Régression logistique ou gradient boosting sur features dérivées : tendance de pression sur 1h et 3h, humidité moyenne, vent max, amplitude thermique. Baseline à battre, présente en permanence.

**M1 TS-JEPA (`tsjepa.py`)**
Reprise et industrialisation du prototype existant :
- PatchEmbed : découpage de la fenêtre en patches, projection linéaire, embeddings de position.
- Encodeur contexte : transformer léger (online, avec gradient).
- Encodeur cible : copie EMA de l'encodeur contexte, stop-gradient.
- Prédicteur : prédit les embeddings cible aux positions masquées, à partir du contexte et des positions.
- Masquage par blocs (style I-JEPA en 1D) : blocs de patches cachés (cible), le reste en contexte.
- Perte SmoothL1 en espace latent, cible normalisée.
- Anti-collapse par asymétrie (prédicteur online seulement) plus cible EMA et stop-gradient. Monitoring de l'écart-type des embeddings.
- Sonde aval : encodeur gelé, mean-pooling, tête linéaire pour la classification « danger ».

### 6.4 Évaluation (`taranis/eval/`)

- Métriques : AUC (principale, robuste au déséquilibre), F1, précision, rappel au seuil figé.
- Sécurité : distribution du lead time, courbe précision-rappel, diagramme de fiabilité (calibration).
- Étude de généralité : pré-entraînement in-domain vs cross-domain (stations différentes exclues de l'entraînement).

## 7. Stack et contraintes

- Langage : Python 3.12, PyTorch. Gestion d'environnement : uv.
- Reproductibilité : seed fixée, configs YAML versionnées, algorithmes déterministes quand possible.
- Matériel : CPU-first pour dev et CI (petites configs), puis GPU souverain (OVH L40S, DGX Spark GB10) pour l'entraînement réel.
- Embarqué : viser un modèle assez compact pour tourner sur mobile (quantification, ONNX ou exécuteur léger).
- Licence : AGPL-3.0. Dépendances de licence compatible.
- Souveraineté des données : sources ouvertes (Météo-France, ERA5, Blitzortung) et hébergement sur infrastructure maîtrisée. Aucune dépendance à une API propriétaire non substituable.
- Pas de tiret cadratin dans la documentation générée. Vocabulaire : « lot » et non « phase ».

## 8. Structure du dépôt

```
taranis/
  taranis/
    data/        synthetic.py meteofrance.py netatmo.py era5.py labels_lightning.py windows.py
    ingest/      ble.py gatt_spec.md buffer.py
    models/      baseline_physics.py tsjepa.py
    eval/        metrics.py probe.py calibration.py
    infer/       api.py            # endpoint d'inférence (FastAPI), sortie régime + probabilité
    cli.py       # train / eval / predict
  configs/       tsjepa.yaml baseline.yaml data.yaml
  tests/         test_data.py test_tsjepa.py test_collapse.py test_ingest.py
  notebooks/     01_explo.ipynb
  data/          .gitignore (données non versionnées)
  firmware/      README.md   # spécification capteur ESP32 (dépôt dédié possible)
  app/           README.md   # spécification application mobile (dépôt dédié possible)
  pyproject.toml
  README.md
  PRD_taranis.md
```

## 9. Interfaces

CLI unique `taranis` :
- `taranis train --model {baseline,tsjepa} --config configs/<f>.yaml`
- `taranis eval --model <m> --ckpt <path> --split test`
- `taranis predict --ckpt <path> --input <csv|ble>` renvoie `{regime, proba, lead_time_est}`

Format de données d'entrée : CSV ou parquet avec colonnes `timestamp, pressure, temp, humidity, wind` au minimum, pas de temps régulier documenté dans la config. En mode `ble`, l'entrée provient du client BLE (`taranis/ingest/ble.py`).

API d'inférence (`infer/api.py`) : endpoint POST `/predict` acceptant une fenêtre, renvoyant le régime dominant, sa probabilité, un niveau d'alerte discret (vert, orange, rouge) et le disclaimer de sécurité.

## 10. Lots et livrables

### Lot 0, socle (Definition of Done)
- Dépôt initialisé, pyproject, uv, CI qui lance les tests sur données synthétiques.
- `synthetic.py` et `windows.py` opérationnels, tests verts.
- Critère d'acceptation : `taranis train --model baseline` tourne de bout en bout sur données synthétiques.

### Lot 1, TS-JEPA et baseline sur données réelles
- Chargeur Météo-France opérationnel, un point réel exploitable.
- Étiquetage foudre branché.
- M0 et M1 entraînés et évalués, tableau comparatif AUC/F1 produit.
- `test_collapse.py` vérifie que l'écart-type des embeddings reste au-dessus d'un seuil.
- Critère d'acceptation : rapport d'éval reproductible, M1 documenté, M0 comme référence. Il est acceptable et attendu que M0 domine à ce stade.

### Lot 2, inférence temps réel et niveaux d'alerte
- Client BLE et spécification GATT (`ingest/`).
- Fenêtre glissante alimentant l'inférence en continu.
- API d'inférence avec niveaux d'alerte vert, orange, rouge et disclaimer.
- Critère d'acceptation : à partir d'un flux simulé ou réel, l'API produit un niveau d'alerte cohérent en continu.

### Lot 3, capteur portable
- Spécification firmware ESP32 : lecture BME280/BME688 et anémomètre, service GATT, cadence, gestion batterie.
- Prototype firmware de référence et test de bout en bout capteur vers application.
- Critère d'acceptation : un capteur physique diffuse des mesures lisibles par le client BLE.

### Lot 4, application mobile
- Application recevant le flux BLE, appelant le modèle, affichant état météo, régime et alerte.
- Design issu du brief UX/UI dédié.
- Critère d'acceptation : parcours complet appairage, lecture, alerte, sur un terrain de test.

## 11. Sécurité et responsabilité

Taranis est une aide à la décision, jamais une garantie. En montagne, un faux négatif peut être mortel. Règles de conception :

- L'interface affiche toujours un disclaimer explicite : l'outil ne remplace pas les bulletins officiels ni le jugement de l'utilisateur.
- Optimiser en priorité le rappel sur la classe dangereuse, en assumant des faux positifs, et rendre ce compromis explicite et configurable.
- Ne jamais présenter une absence d'alerte comme une assurance de sécurité.
- Documenter la latence et le préavis typique mesuré, sans le surestimer.

## 12. Risques et mitigations

| Risque | Mitigation |
|---|---|
| Données locales trop rares en événements orageux | Pré-entraînement sur multi-stations et ERA5, fine-tuning local |
| Sur-confiance de l'utilisateur | Disclaimer systématique, priorité au rappel, niveaux d'alerte prudents |
| Le point unique ne voit pas la dynamique spatiale | Assumer le périmètre nowcasting, préparer la fusion en évolution future |
| Collapse de représentation | Test automatique sur l'écart-type et le rang de covariance des embeddings |
| Autonomie et robustesse du capteur en montagne | Cadence de mesure raisonnable, boîtier résistant, gestion d'énergie soignée |
| Perte de liaison BLE | Bufferisation locale, dégradation gracieuse, alerte de perte de signal |

## 13. Instructions pour Claude Code

- Travailler lot par lot, commits atomiques, un test avant chaque brique.
- Ne pas court-circuiter la baseline physique : elle reste dans le dépôt et dans chaque rapport.
- Surveiller le collapse en continu (écart-type et rang de covariance des embeddings), en faire un test de non-régression.
- Ne pas sur-optimiser les hyperparamètres avant que le pipeline complet ne tourne.
- Documenter chaque écart d'implémentation dans un fichier `NOTES_impl.md`.
- Le contrat BLE (`gatt_spec.md`) doit être stable et versionné dès le lot 2, car firmware et application en dépendent.

## 14. Hyperparamètres de référence (TS-JEPA, point de départ)

Issus du prototype fonctionnel, à affiner sur données réelles.

- Fenêtre `Tw` = 96 pas, patch `L` = 8, soit 12 patches par fenêtre. Horizon `H` = 48 pas.
- Dimension latente `D` = 96.
- Encodeur contexte et cible : transformer 3 couches, 4 têtes, feedforward 2·D, activation GELU.
- Prédicteur : transformer 2 couches, 4 têtes.
- Masquage : 2 blocs de 3 patches non chevauchants (fort taux de masquage, cohérent avec TS-JEPA).
- EMA decay 0.996, stop-gradient sur la cible, LayerNorm sur les cibles.
- Optimiseur AdamW, lr 1e-3, weight decay 1e-4, perte SmoothL1 en espace latent.
- Sonde aval : encodeur gelé, mean-pooling, tête linéaire, métrique AUC.

## 15. Références

- Ennadir, Golkar, Sarra. Joint Embeddings Go Temporal (TS-JEPA). arXiv:2509.25449, 2025.
- Assran et al. I-JEPA, Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023.
- Bardes et al. V-JEPA, Revisiting Feature Prediction for Learning Visual Representations from Video. TMLR 2024.
