# Taranis

**Construire pas à pas un modèle capable d'anticiper un orage à partir d'un capteur météo local.**

Ce site est le carnet d'apprentissage du projet Taranis. Il vise deux choses à la fois : comprendre l'architecture **JEPA** (Joint Embedding Predictive Architecture) appliquée aux séries temporelles, et produire un outil utile. Vous trouverez ici la chaîne complète, de la génération d'un signal capteur synthétique jusqu'à un modèle qui apprend à repérer une signature pré-orageuse.

## À qui s'adresse ce carnet

Un ingénieur ou une ingénieure qui code en Python, connaît les bases du machine learning supervisé, mais n'a jamais mis les mains dans l'auto-supervisé, les transformers ou JEPA. Chaque nouvelle notion est introduite quand elle devient utile, jamais avant.

## La promesse

À la fin du parcours, vous aurez :

- un pipeline de données reproductible, synthétique puis réel,
- une baseline physique honnête, à battre,
- un modèle **TS-JEPA** entraîné, disséqué, comparé à la baseline,
- un test qui surveille le piège classique de JEPA, le collapse de représentation.

Chaque brique est motivée avant d'être codée. On n'écrit pas de ligne de PyTorch sans avoir dit à quoi elle sert.

## Ce que ce carnet n'est pas

Ce n'est pas un cours théorique complet sur les transformers. Ce n'est pas non plus une implémentation optimisée pour la production. On préfère la lisibilité aux hyperparamètres pointus tant que le pipeline n'a pas roulé de bout en bout.

## Avertissement de sécurité

Un capteur ponctuel donne un signal local et à court préavis, jamais une prévision spatiale. La cible réaliste du projet est le **nowcasting** et l'alerte précoce, **jamais une garantie**. En montagne, un faux négatif peut être mortel. L'outil ne remplace pas les bulletins officiels ni le jugement de l'utilisateur.

## Par où commencer

Suivez [Le plan pédagogique](00-plan.md), puis attaquez [l'étape 1](01-le-probleme.md).

## Licence

AGPL-3.0-or-later.
