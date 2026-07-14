# Étape 19, passage produit, la PWA on-device

## Le décor

À la fin du chapitre 18, un choix produit était acté : le capteur portable ne fournit que 3 canaux (P, T, HR), le modèle HGB-3ch-Tw8 tourne dessus avec AUC 0.7734, et TS-JEPA-3ch reste dans le carnet comme démonstration pédagogique. Reste une question évidente : **comment un randonneur utilise cela sur le terrain** ?

Ce chapitre documente le passage du prototype fastAPI à une vraie **Progressive Web App** installable, avec inférence 100 % locale dans le navigateur du téléphone. C'est là que le vrai travail commence, et où les enseignements les plus utiles se sont révélés.

## Le contrat produit

Trois principes non négociables issus du carnet :

1. **Le capteur est le point de vérité.** L'appli lit un capteur BLE local, jamais une API cloud qui devine à partir de la position GPS
2. **Aucune donnée utilisateur ne quitte le téléphone** hors opt-in explicite (Open-Meteo, Nominatim)
3. **Souveraineté logicielle** : code AGPL-3.0, données Etalab, hébergement propre, aucune Cloud Function payante

Ces trois principes ont dicté toutes les décisions techniques qui suivent.

## Le choix architectural, PWA vs app native

Trois options étaient sur la table :

| Option | Effort | Distribution | Souveraineté |
|---|---|---|---|
| **PWA** | 1 x | pas de store, URL directe | totale |
| Android natif Kotlin | 3 x | Play Store + APK direct | totale |
| Flutter hybride | 2 x | 2 stores | dépendance framework |

La PWA gagne trois fois : effort développeur minimal, distribution sans intermédiaire (le randonneur tape l'URL, la PWA s'installe, point), et souveraineté totale parce que rien ne dépend d'un app store américain. **Le seul coût** : Web Bluetooth n'est pas supporté sur iOS Safari, ce qui force un contournement (le navigateur alternatif Bluefy) pour les utilisateurs Apple.

Décision : **PWA**, avec fallback documenté sur iOS.

## Export des modèles vers le navigateur

Le premier vrai chantier : faire tourner HGB et TS-JEPA **dans le navigateur du téléphone**, sans serveur.

### HGB, JSON compact plutôt qu'ONNX

L'approche naturelle serait d'exporter HGB vers ONNX via `skl2onnx`. J'ai passé trois heures dessus. Le converter officiel a un bug avec les attributs booléens dans les arbres `HistGradientBoostingClassifier` de scikit-learn 1.9 : `TypeError: Field onnx.AttributeProto.ints: Expected an int, got a boolean`. Monkey-patcher `onnx.helper.make_node` pour coercer les listes booléennes en int64 aide sur une partie des attributs mais pas sur ceux passés en `numpy.ndarray[bool]`.

Pivot : sérialiser directement les 200 arbres du modèle en JSON compact. Chaque arbre a 5 tableaux (`feature`, `threshold`, `left`, `right`, `value`), l'évaluateur JavaScript tient en 20 lignes :

```js
function walkTree(tree, features) {
    let node = 0;
    while (tree.feature[node] >= 0) {
        const f = tree.feature[node];
        node = features[f] <= tree.threshold[node]
            ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
}

export function predictHGB(bundle, features) {
    let score = bundle.base_score;
    for (const tree of bundle.trees) score += walkTree(tree, features);
    return 1.0 / (1.0 + Math.exp(-score));
}
```

Résultat : **438 KB de modèle** (200 arbres, 12 200 nœuds), parité stricte avec sklearn (`max_diff = 0.00e+00` sur 200 fenêtres test), aucune dépendance externe, aucun runtime ONNX à charger.

### TS-JEPA, ONNX via torch.onnx.export

TS-JEPA est un transformer PyTorch. `torch.onnx.export` avec `dynamo=False` réussit du premier coup une fois `onnxscript` installé. J'encapsule l'encoder + le mean-pool + la sonde LogReg + la sigmoïde dans une seule classe `TaranisJEPA`, ce qui donne un seul graphe ONNX qui prend directement `(B, Tw=32, 3)` et retourne la probabilité.

Résultat : **476 KB ONNX**, parité PyTorch vs ONNX Runtime `4.47e-07`, chargement `onnxruntime-web` depuis jsDelivr, inférence ~5 ms sur Pixel 10.

Les deux moteurs coexistent dans `models/`, l'utilisateur bascule à chaud dans les réglages.

## Le squelette PWA

Un service worker cache-first (`sw.js`) charge la coquille (HTML, CSS, JS, modèles, icônes) à l'installation. Après ça, l'appli fonctionne 100 % hors ligne. IndexedDB stocke un buffer glissant des mesures capteur des dernières 24 h.

Point crucial : le service worker versionne son cache (`taranis-YYYYMMDD-HHMMSS`) et un `controllerchange` listener force un reload automatique quand une nouvelle version prend le contrôle. Sans ça, les phones restent bloqués sur du code vieux de plusieurs jours.

Structure finale de l'app :

```
Utilisateur (téléphone offline-capable)
    │
    │  HTTPS Let's Encrypt via Caddy
    ▼
┌──────────────────────────────────────────────┐
│  PWA statique servie par Caddy               │
│  taranis/infer/static/                       │
│  ├─ index.html                                │
│  ├─ css/  js/  models/  icons/                │
│  ├─ sw.js       (service worker)             │
│  └─ IndexedDB   (buffer 24 h capteur)        │
└──────┬───────────────────────────┬───────────┘
       ▼                           ▼
┌──────────────┐            ┌─────────────────┐
│  HGB-3ch     │            │  TS-JEPA-3ch    │
│  438 KB JSON │            │  476 KB ONNX    │
│  JS 20 lignes│            │  onnxruntime-web │
└──────────────┘            └─────────────────┘
```

## Les quatre écrans

Le design est venu d'un import [Claude Design](https://claude.ai/design). Quatre écrans :

- **Accueil** : bague circulaire d'alerte VERT/ORANGE/ROUGE, nom de localité, probabilité de prédiction, préavis
- **Live** : hero pression avec courbe 6 h + tendance, cartes T et HR, bloc contexte régional Open-Meteo (vent, rafales, pluie prévue, POP, CAPE)
- **Historique** : courbe pression 6 h colorée par taux de chute (vert / ambre / rouge), crosshair interactif au tap, liste d'événements auto-détectés
- **Capteur** : anneau pulsant, bouton coupler, statistiques batterie et RSSI

Palette « sobre montagne » (dark #0A0E15, thunder #71E3FF, feux tricolores VERT/AMBER/ROUGE), typographie Space Grotesk + IBM Plex Mono, isobares en fond pour rappeler la sémantique météo.

## Décision produit, Open-Meteo par défaut

Sans le RuuviTag encore présent, l'appli doit fonctionner immédiatement pour la première visite. Décision : **Open-Meteo devient la source par défaut**, capteur BLE en option quand couplé, mock en dernier choix pour développeurs.

Open-Meteo fournit `pressure_msl`, `temperature_2m`, `relative_humidity_2m` par point GPS, gratuit et sans clé. Le buffer est rempli avec `past_hours=100`, refresh toutes les 30 minutes. Quand le capteur BLE est couplé, il devient prioritaire et Open-Meteo passe en contexte régional (vent, précipitations, CAPE).

## Internationalisation en cinq langues

Français, anglais, espagnol, italien, allemand. 95 clés par langue, détection automatique via `navigator.language` au premier boot, sélecteur explicite dans les réglages. Les strings dynamiques (statut d'alerte, régime, notes) passent par une fonction `t(key)` avec fallback silencieux vers l'anglais.

Ce n'est pas glamour, mais c'est ce qui fait la différence entre une démo et un produit utilisable au-delà des frontières.

## Déploiement HTTPS souverain

La PWA est un dossier statique servi par **Caddy 2** sur infrastructure OVH souveraine. Certificat Let's Encrypt automatique, sous-domaine `taranis.maudet.cloud`. Le déploiement est un `docker compose restart caddy` : instantané, atomique, avec bind mount du dossier statique.

Un piège documenté ici et dans le blog du carnet : **le bind mount se pinne à l'inode du fichier**. Éditer le Caddyfile change parfois l'inode, le container voit encore l'ancien contenu, un reload de config ne suffit pas. La cible `make deploy` fait donc un vrai `docker compose restart` à chaque déploiement.

Politique cache HTTP réglée pour l'itération rapide :
- `sw.js`, `*.html`, `js/*`, `css/*` : **`Cache-Control: no-cache`**
- `models/`, `icons/` : `max-age=3600`

Cette politique était initialement `max-age=3600` sur tout. Résultat : le bouton « Recharger » de l'app purgeait le Service Worker mais Chrome servait le JavaScript depuis le cache HTTP pendant encore une heure. **Une bonne semaine passée à comprendre pourquoi les fixes ne se propageaient pas sur le Pixel de test**. La bascule vers `no-cache` sur les fichiers de code a résolu le problème d'un coup.

## Audit end-to-end

Un script `scripts/pwa_e2e_audit.sh` vérifie 31 assertions en une passe :

1. Fichiers disque locaux cohérents
2. Fichiers servis via HTTPS identiques
3. Cache-Control par type de fichier
4. Prédiction Node de bout en bout sur Open-Meteo Noja

`make audit` déclenche la suite. Elle a servi à démasquer plusieurs régressions silencieuses : bind mount stale, Caddyfile pas rechargé, i18n manquante, `data-taranis-*` hooks retirés par un refactor. **La règle du carnet vaut ici aussi : si on ne le mesure pas automatiquement, on ne le tient pas.**

## Ce que ce chapitre apporte

Le training du modèle Taranis est fini depuis le chapitre 18. Ce chapitre décrit ce qu'il a fallu bâtir **autour** du modèle pour qu'il serve un utilisateur réel :

- Export ONNX + JSON avec parité stricte
- Service worker + Cache API + IndexedDB
- Cache HTTP contrôlé côté serveur
- Bind mount inode-aware
- Web Bluetooth + fallback Bluefy iOS
- Ré-actualisation GPS auto
- Reverse geocoding Nominatim
- i18n cinq langues
- Coexistence capteur + Open-Meteo
- Audit E2E scripté

**Vingt journées de développement contre quelques heures de training.** Cette proportion mérite d'être documentée honnêtement pour toute personne qui suit ce carnet.

## Reproduire

```bash
# Export des modèles
uv run python scripts/export_hgb_tw8_json.py    # HGB 438 KB
uv run python scripts/export_tsjepa_onnx.py     # TS-JEPA 476 KB

# Servir la PWA localement
cd taranis/infer/static && python3 -m http.server 8899
# → http://127.0.0.1:8899

# Déployer sur taranis.maudet.cloud
make deploy    # bump SW + docker compose restart caddy
make audit     # 31 checks, tout doit être vert
```
