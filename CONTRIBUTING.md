# Contribuer à Taranis

Merci de l'intérêt porté au projet. Taranis est un carnet pédagogique et une base de recherche autour de JEPA appliqué aux séries temporelles météorologiques. Les contributions sont bienvenues, sous quelques principes.

## Périmètre du dépôt actuel

Le dépôt couvre le **cerveau logiciel** : génération et chargement de données, entraînement de TS-JEPA, sondes aval, évaluation croisée. La partie matérielle (capteur ESP32, application mobile, ingestion BLE) est reportée à des dépôts dédiés (lots 3 et 4 du PRD).

Voir le [PRD](PRD_taranis.md) pour la vision d'ensemble et le [carnet pédagogique](docs/00-plan.md) pour la progression thématique.

## Ce qui est welcome

- **Corrections** dans le code, les tests, la documentation.
- **Nouvelles sources de données** ouvertes et souveraines (Météo-France 6-minutes, DWD, ECA&D, etc.).
- **Étiquettes plus fines** (Blitzortung, réseau national foudre, radar).
- **Améliorations de l'architecture TS-JEPA** documentées et testées.
- **Portage GPU** de la boucle d'entraînement (mixed precision, DataLoader multi-workers).
- **Reproduction sur de nouvelles zones géographiques** avec analyse comparée.

## Style et conventions

Les conventions du projet sont documentées dans le PRD, section 7 et 13. En résumé :

- Rédaction en français, sans tiret cadratin.
- Vocabulaire « lot » (pas « phase »).
- Un test avant chaque brique de code publiée.
- Pas de commits qui bypassent la baseline physique dans les évaluations.
- Documenter tout écart d'implémentation.

## Développement local

```bash
uv sync --extra dev --extra docs
uv run pytest             # 31 tests actuellement
uv run ruff check .       # style
uv run mkdocs serve       # site pédagogique local
```

## Ouvrir une pull request

1. Créer une branche depuis `main`, nommée `feat/<sujet>`, `fix/<sujet>` ou `docs/<sujet>`.
2. Commits atomiques, un sujet par commit, message clair en anglais ou français.
3. La CI doit passer (`pytest`, `ruff check`).
4. Décrire la motivation dans la PR, référencer les métriques concernées, éviter les micro-optimisations qui ne changent pas de scénario testé.

## Signalement de bug ou de discussion

Ouvrir une issue avec :

- version de Python et OS,
- commande exacte lancée,
- extrait pertinent des logs,
- comportement observé vs attendu.

## Licence

Toute contribution est réputée soumise à la licence AGPL-3.0-or-later du projet. En contribuant, tu acceptes que ton code soit distribué sous cette licence.
