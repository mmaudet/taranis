# Security Policy

Merci de prendre le temps de lire ce document si vous avez trouvé une
vulnérabilité dans Taranis.

## Versions supportées

Le projet est en développement actif et itératif. **Seule la branche
`main`** reçoit les correctifs de sécurité. Les modèles pré-entraînés
distribués dans `taranis/infer/static/models/` sont considérés comme
faisant partie de cette branche.

| Version | Supportée |
|---|---|
| `main` (HEAD)  | Oui |
| Anciens commits | Non |

## Périmètre couvert

Sont dans le périmètre :

- **Code Python** dans `taranis/` (entraînement, évaluation, export)
- **PWA** dans `taranis/infer/static/` (JS, HTML, CSS, service worker)
- **Modèles servis** (`hgb_3ch_tw8.json`, `tsjepa_3ch.onnx`), s'ils
  contiennent des données personnelles ou permettent une exfiltration
- **Instance publique** [taranis.maudet.cloud](https://taranis.maudet.cloud)
  hébergée par l'auteur

Ne sont **pas** dans le périmètre :

- Les vulnérabilités des dépendances tierces (Open-Meteo, Nominatim,
  cdn.jsdelivr.net) qui ont leurs propres processus de sécurité
- Les problèmes de compatibilité navigateur ou d'ergonomie qui ne
  compromettent pas la confidentialité ou l'intégrité

## Signaler une vulnérabilité

**Ne pas ouvrir d'issue publique** pour une vulnérabilité qui n'est
pas déjà connue.

Deux voies possibles, dans l'ordre de préférence :

1. **GitHub Security Advisories** (privé) sur
   [github.com/mmaudet/taranis/security/advisories/new](https://github.com/mmaudet/taranis/security/advisories/new).
   C'est le canal recommandé.
2. **Email** à `security@maudet.cloud` avec description précise
   (impact, reproductible, versions concernées, environnement).

Merci de fournir dans votre signalement :

- Description claire de la vulnérabilité
- Étapes de reproduction ou proof-of-concept
- Impact potentiel (confidentialité, intégrité, disponibilité)
- Version ou commit hash concerné
- Toute idée de mitigation si vous en avez

## Délai de réponse

- Accusé de réception sous **72 heures ouvrables** en semaine
- Évaluation initiale sous **7 jours calendaires**
- Correctif ou plan d'action sous **30 jours** pour les vulnérabilités
  critiques ou majeures

Le projet est développé sur temps personnel, ces délais sont indicatifs.
Merci de votre patience et de votre indulgence.

## Divulgation coordonnée

Je préfère la **divulgation responsable** :

- Publication publique après qu'un correctif soit déployé
- Crédit du rapporteur dans la note de release (avec accord)
- Fenêtre standard de **90 jours** avant divulgation même sans correctif,
  pour respecter les usages industriels

## Zone sûre

Toute recherche de sécurité menée de bonne foi, dans le respect de ce
document, ne fera l'objet d'aucune poursuite judiciaire ou technique
de la part de l'auteur. Les activités suivantes ne sont pas
considérées comme malveillantes :

- Tests d'intrusion sur l'instance publique tant qu'ils n'affectent
  pas la disponibilité pour les autres utilisateurs
- Analyse statique et dynamique du code, du service worker, des modèles
- Ingénierie inverse des fichiers modèles (`hgb_3ch_tw8.json`,
  `tsjepa_3ch.onnx`), leur licence AGPL le permet explicitement
