<!-- Merci de contribuer à Taranis. Remplissez ce template le mieux possible. -->

## Résumé

<!-- 1 à 3 phrases : quoi, pourquoi, comment. -->

## Type de changement

<!-- Cochez la case appropriée. -->

- [ ] Correction de bug (`fix`)
- [ ] Nouvelle fonctionnalité (`feat`)
- [ ] Refactor sans changement de comportement (`refactor`)
- [ ] Documentation, carnet, README (`docs`)
- [ ] Tests uniquement (`test`)
- [ ] Chore, tooling, CI (`chore`)

## Zone touchée

- [ ] Modèle / entraînement (`taranis/models/`, `taranis/train/`)
- [ ] Évaluation (`taranis/eval/`, `scripts/eval_*.py`)
- [ ] Données (`taranis/data/`, `scripts/prepare_*.py`)
- [ ] PWA (`taranis/infer/static/`)
- [ ] Documentation (`docs/`)
- [ ] Infra (Makefile, Caddyfile, Docker, CI)

## Issue liée

<!-- Fixes #XX, Closes #YY, Related to #ZZ. Enlever si non applicable. -->

## Comment tester

<!-- Étapes précises que le mainteneur peut suivre. -->

```bash
# exemple
uv run pytest tests/test_ma_feature.py
```

## Impact utilisateur

<!--
Y a-t-il un changement visible pour l'utilisateur final de la PWA
(nouvelle langue, nouveau réglage, changement de prédiction) ou pour
le lecteur du carnet ? Décrire.
-->

## Checklist

- [ ] Mon code respecte le style du dépôt (ruff clean, PEP 8)
- [ ] J'ai ajouté ou mis à jour les tests si nécessaire (`pytest` passe)
- [ ] J'ai mis à jour la documentation si nécessaire
- [ ] Les commentaires de code sont en anglais, la doc utilisateur en français
- [ ] Aucun tiret cadratin dans le code, la doc, ou le message de commit
- [ ] Aucun secret, aucune clé, aucune donnée personnelle n'est commit
- [ ] Le message de commit suit les conventions (`feat:`, `fix:`, `docs:`, etc.)

## Notes pour le mainteneur

<!--
Choses à savoir avant la revue : contraintes de temps, dépendances,
choix architecturaux qui méritent discussion.
-->
