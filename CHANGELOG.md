# Changelog

Tous les changements notables sont documentés ici. Le format suit
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) et le versioning
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ingestion Web Bluetooth Ruuvi v5 réelle (parseur RAWv2, path
  `requestLEScan` + fallback GATT NUS, live UI update)
- Chapitre 19 du carnet, passage produit PWA on-device
- Chapitre 20 du carnet, bugs terrain altitude et harnais
- CHANGELOG.md (ce fichier)
- Community health files : `SECURITY.md`, templates issue/PR

### Changed
- README.md complet aux standards communautaires (badges, screenshots,
  architecture, feuille de route)

## [0.4.0] — 2026-07-14

### Added
- Reverse geocoding Nominatim avec cache localStorage
- Auto-refresh GPS configurable (Manuel / 10 min / 30 min / 1 h,
  défaut 10 min pour usage randonnée), seuil 300 m
- Version chip visible dans la topbar pour diagnostic à distance
- Bouton « Recharger l'app » dans les réglages (purge Cache API + SW)
- Audit E2E scripté (`scripts/pwa_e2e_audit.sh`, 31 checks)
- Article de fond publié sur blog.maudet.cloud

### Changed
- Source Open-Meteo utilise désormais `pressure_msl` (Mean Sea Level)
  au lieu de `surface_pressure`, ce qui rend la prédiction indépendante
  de l'altitude du point de grille
- HGB tourne sur Tw=8 (24 h de lookback) au lieu de Tw=32 (96 h),
  AUC 0.7734 vs 0.7734 (+0.25 pts sur l'ancien), buffer capteur 4x plus
  léger, aligné sur usage randonneur réaliste
- Nav bottom en `position: fixed` avec safe-area-inset-bottom,
  fonctionne sur foldable et sur toutes tailles de viewport mobile
- Cache-Control durci sur `js/`, `css/`, `sw.js`, `*.html` en
  `no-cache` pour propagation instantanée des fixes

### Fixed
- Label position GPS n'affiche plus « Chamonix » quand l'utilisateur
  est ailleurs (nouvelle clé i18n `home.gps_position`)
- Buffer purgé au changement de localisation (fix du prédictogramme
  Chamonix persistant à Noja)
- Docker bind mount `docker compose restart caddy` au lieu de `reload`
  pour prendre en compte les changements de Caddyfile

## [0.3.0] — 2026-07-13

### Added
- PWA installable complète, 4 écrans (Home, Live, History, Sensor)
- Export ONNX du modèle TS-JEPA-3ch (476 KB, parité PyTorch 4.47e-07)
- Export JSON compact du modèle HGB-3ch (438 KB, parité sklearn 0.00e+00)
- Sélecteur de moteur HGB / TS-JEPA dans les réglages
- i18n cinq langues (FR / EN / ES / IT / DE) avec détection navigateur
- Panneau contexte régional Open-Meteo (vent, rafales, pluie, CAPE, POP)
- Crosshair interactif sur les graphes pression Live et Historique
- Géolocalisation opt-in via `navigator.geolocation`
- Déploiement HTTPS sur `taranis.maudet.cloud` via Caddy + Let's Encrypt
- Chapitres 17 (HGB vs TS-JEPA) et 18 (sonde capteur 3 canaux) du carnet

### Changed
- Design PWA refondu depuis import Claude Design, palette « sobre
  montagne » (Space Grotesk + IBM Plex Sans + Mono, isobares en fond)

## [0.2.0] — 2026-07-11

### Added
- TS-JEPA-3ch entraîné et évalué (AUC 0.7361, LOO 14 stations, cross-régime)
- Baseline HGB-3ch entraînée et évaluée (AUC 0.7734)
- Sonde LogReg par-dessus les embeddings TS-JEPA
- Bootstrap AUC IC 95 %, distribution lead time, leave-one-station-out
- Chapitres 15 (évaluation rigoureuse) et 16 (recalibration seuils) du carnet
- Serveur autonome FastAPI + mockup mobile HTML
- Publication carnet MkDocs sur blog.maudet.cloud/taranis/

### Changed
- Recalibration ORANGE / ROUGE : seuils recall 0.70 / 0.30 au lieu de
  0.80 / 0.40, meilleur ratio signal / bruit (fausses alarmes 43 % → 32 %)

## [0.1.0] — 2026-07-01

### Added
- Squelette projet, licence AGPL-3.0-or-later
- Générateur synthétique de signal orage
- Baseline physique (`BaselinePhysics`, 10 features + LogReg)
- Architecture TS-JEPA de base (`TSJEPA`, encoder + prédicteur + EMA target)
- Boucle d'entraînement JEPA avec surveillance du collapse
- Dataset SYNOP Météo-France (15 ans × 62 stations)
- Codes WMO de temps significatif pour labels d'orage
- Chapitres 1 à 14 du carnet pédagogique

---

## Convention de versioning

- **MAJOR** : rupture d'API publique (format modèle, format buffer,
  URL de la PWA, signature Python)
- **MINOR** : ajout de fonctionnalité rétro-compatible (nouveau moteur,
  nouvelle langue i18n, nouveau chapitre)
- **PATCH** : correction de bug, refactor sans changement d'interface

Les versions sont taguées `vX.Y.Z` sur GitHub et alignent le CACHE_VERSION
du service worker de la PWA au format `taranis-YYYYMMDD-HHMMSS`.
