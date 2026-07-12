# Le plan pédagogique

Le fil rouge du parcours suit sept étapes. Chaque étape a une raison d'être et prépare la suivante. On construit d'abord ce qui est concret, on introduit les abstractions au moment où elles deviennent inévitables.

## Pourquoi cet ordre

La tentation naturelle serait de foncer sur JEPA. C'est une mauvaise idée pédagogique. On ne peut pas juger si une architecture apprend « quelque chose d'utile » sans avoir mesuré ce qu'un modèle simple fait déjà. On commence donc par regarder les données, puis par construire une baseline honnête. Cette baseline sert de référence pour toute la suite. Si un jour un modèle sophistiqué ne la bat pas, on saura pourquoi et on ne se racontera pas d'histoire.

## Les sept étapes

### 1. Le problème
On génère un signal capteur synthétique et on regarde à quoi ressemble un pré-orage dans la pression, l'humidité, le vent, la température. Sans modèle, juste avec les yeux. Objectif : identifier une signature reconnaissable qui justifie l'existence du projet.

**Livrables** : `taranis/data/synthetic.py`, [étape 1 du carnet](01-le-probleme.md).

### 1 bis. Le dataset d'entraînement
On documente précisément d'où viennent les données (synthétiques dans cette phase), quelle est leur volumétrie (365 jours simulés, 52 000 fenêtres), comment on découpe en train, val, test dans l'ordre chronologique, et comment on normalise. C'est le préalable indispensable pour croire aux métriques qui suivent.

**Livrables** : `scripts/prepare_dataset.py`, [étape 1 bis du carnet](01b-le-dataset.md).

### 1 ter. Brancher les données réelles
En parallèle du synthétique, on charge de vraies données SYNOP Météo-France via l'API Opendatasoft. Quatre stations du sud-est de la France, sur 5 ans, à pas de 3 heures. On documente les différences, on adapte fenêtrage et labels, on entraîne la baseline sur les deux. Objectif : préparer une comparaison **sim-to-real** honnête.

**Livrables** : `taranis/data/meteofrance.py`, `scripts/prepare_real_dataset.py`, `scripts/train_baseline_real.py`, [étape 1 ter du carnet](01c-donnees-reelles.md).

### 2. Une baseline honnête
On construit **M0**, un modèle physique simple qui prédit un orage à partir de features intuitives : tendance de pression, humidité moyenne, vent max. Régression logistique ou gradient boosting, rien de plus. On mesure AUC, précision, rappel. Cette baseline reste dans le dépôt pour toujours.

**Livrables** : `taranis/models/baseline_physics.py`, étape 2 du carnet.

### 3. Pourquoi apprendre des représentations
On explicite les limites de M0. Que fait-il quand la signature n'est pas franche ? Quand deux régimes se ressemblent ? On introduit l'idée que le modèle pourrait apprendre à représenter les données sans qu'on lui dise quoi chercher. C'est le pas vers l'auto-supervisé.

**Livrable** : étape 3 du carnet, principalement conceptuelle.

### 4. JEPA en une image
On explique l'idée cœur de JEPA en une phrase et un schéma. On prédit dans l'espace des représentations, pas dans l'espace des pixels ou des mesures brutes. On motive ce choix et on annonce le piège associé, le collapse.

**Livrable** : étape 4 du carnet, avec un schéma commenté.

### 5. TS-JEPA brique par brique
On construit les composants un par un, chacun testable isolément avant l'assemblage :
- PatchEmbed, découpage de la fenêtre en patches et projection,
- encodeur contexte, transformer léger,
- encodeur cible, copie EMA de l'encodeur contexte, stop-gradient,
- prédicteur, transformer qui prédit les embeddings cibles aux positions masquées,
- stratégie de masquage par blocs,
- perte SmoothL1 en espace latent.

**Livrables** : `taranis/models/tsjepa.py`, étapes 5a à 5f du carnet.

### 6. L'assemblage, l'entraînement, la surveillance du collapse
On assemble, on entraîne, on trace la perte. On instrumente le training loop pour mesurer en continu l'écart-type et le rang de covariance des embeddings. C'est notre garde-fou contre le collapse, le piège classique de JEPA.

**Livrables** : boucle d'entraînement, `test_collapse.py`, étape 6 du carnet.

### 7. La sonde aval, la comparaison, la conclusion honnête
On gèle chaque encodeur, on ajoute une tête linéaire, on entraîne cette tête sur la tâche d'alerte, on mesure. On compare M0 et M1 sur les trois datasets. On teste le **transfert sim-to-real** dans les deux sens. On rédige le tableau croisé et un verdict lisible. Si M1 ne bat pas M0, on le dit.

**Livrables** : `taranis/eval/probe.py`, `scripts/evaluate_all.py`, [étape 7 du carnet](07-verdict.md).

### 8. Le passage à l'échelle
On sort du régime CPU pédagogique. On récupère 2.7 millions de records SYNOP sur 62 stations et 16 ans, on ajoute la réanalyse ERA5 pour le pré-entraînement massif, et on prépare toutes les configs GPU pour un vrai entraînement à l'échelle recherche.

**Livrables** : `scripts/fetch_meteofrance_full.py`, `scripts/fetch_era5.py`, `taranis/data/era5.py`, `configs/tsjepa_era5_gpu.yaml`, `configs/tsjepa_synop_full_gpu.yaml`, [étape 8 du carnet](08-passage-a-l-echelle.md).

## Ce qui vient après

Les étapes futures sortent du carnet actuel : étiquetage foudre Blitzortung, fusion multi-source avec radar et satellite, puis ingestion temps réel BLE et capteur portable ESP32. Le PRD initial les décrit dans ses lots 3 et 4.
