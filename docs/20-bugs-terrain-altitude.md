# Étape 20, la validation terrain qui révèle les vrais bugs

## Le décor

Le chapitre 19 documente la mise en place de la PWA. À ce stade tout marche « sur le papier » : les modèles sont exportés, le harnais tient, l'audit E2E est vert. Reste ce qui compte vraiment, **la validation en conditions réelles**.

Ce chapitre raconte la série de bugs découverts pendant un trajet en voiture à travers la Cantabrie, entre Meruelo, Noja et Hazas de Cesto, un après-midi de juillet 2026. Trois villages voisins, moins de dix kilomètres les séparent, tous sous une même masse d'air stable. La PWA à mesuré, sur les mêmes données Open-Meteo :

- Meruelo (43.44°N, 3.57°W, altitude 22 m) : **45 % de probabilité, VERT**
- Noja (43.48°N, 3.53°W, altitude 5 m) : **50 % de probabilité, VERT**
- Hazas de Cesto (43.42°N, 3.60°W, altitude 256 m) : **81 % de probabilité, ROUGE**

Trois villages voisins qui vivent la même météo. Trente-six points de probabilité d'écart. Le modèle a hurlé au loup pour un col qui n'existait pas. Voici pourquoi, et comment le fixer.

## Bug 1, l'altitude déguisée en signal d'orage

Open-Meteo renvoie `surface_pressure` par défaut, la pression **à la surface** du point de grille. Cette valeur dépend directement de l'altitude géographique :

| Endroit | Grille altitude | `surface_pressure` | `pressure_msl` |
|---|---|---|---|
| Meruelo | 22 m | 1013 hPa | 1015.6 hPa |
| Hazas de Cesto | **256 m** | **986 hPa** | 1015.9 hPa |

Les deux villages ont une pression réduite au niveau de la mer quasi identique (1015.6 vs 1015.9), c'est-à-dire **la même météo réelle**. Mais leur pression de surface diffère de 27 hPa à cause de la géographie.

Le modèle HGB, entraîné sur les stations SYNOP françaises qui reportent `pres` (station pressure, à l'altitude de la station), a appris que « pression basse = risque orageux ». Il ne fait pas la différence entre une baisse de pression due à une dépression synoptique et une baisse due au relief local. Un point de grille en colline devient orageux dès qu'il existe.

**Fix : basculer la source Open-Meteo sur `pressure_msl`.** Trois lignes de JavaScript. Le modèle reçoit maintenant la pression réduite au niveau de la mer, qui reflète le vrai signal synoptique et est **indépendante de l'altitude géographique**. Résultat sur les mêmes coordonnées :

| Endroit | Nouvelle prédiction |
|---|---|
| Meruelo | 38 % VERT |
| Noja | 36 % VERT |
| Hazas de Cesto | **48 %** VERT |
| Chamonix (vérification, altitude 1035 m) | 15 % VERT |

Cohérence géographique restaurée. Tous les hameaux d'une même masse d'air voient maintenant la même prédiction, avec de petites variations dues aux nuances de vent et humidité locales.

## Ce que cela nous apprend sur le modèle

Le modèle **n'a pas été entraîné sur `pressure_msl`**. Il a été entraîné sur `pres` SYNOP, qui est station pressure. Le passage en MSL introduit un décalage vs distribution d'entraînement. En pratique, ce décalage est faible pour la France métropolitaine (la plupart des stations SYNOP sont en plaine ou côtières, donc `pres` ≈ `pmer`) mais il existe.

Deux options :

1. **Réentraîner sur `pmer` SYNOP** (pression réduite au niveau de la mer, disponible dans le CSV Météo-France). Techniquement propre. Nécessite une passe de préparation dataset.
2. **Utiliser `pressure_msl` en inférence** avec le modèle existant. Léger biais résiduel, mais bien plus faible que le biais actuel de l'altitude géographique brute.

J'ai choisi 2 pour l'itération courante, en documentant le compromis. Le futur TS-JEPA enrichi (chapitre 21) sera entraîné sur MSL dès le départ pour éviter le sujet.

## Bug 2, le buffer qui suit le premier utilisateur

Le premier bug identifié ce même jour tient à un problème d'orchestration. Quand l'application boot avec l'utilisateur par défaut (Chamonix), Open-Meteo fetche les 100 h de Chamonix. L'utilisateur tape ensuite « Localiser » et passe à Noja. La position affichée devient bien Noja, mais **le buffer contient toujours les 100 h de Chamonix**.

Résultat : le modèle prédit un orage à Noja sur des données de Chamonix. Chamonix, à 1035 m d'altitude, retourne `surface_pressure` autour de 890 hPa. Vu de HGB, c'est très bas donc très à risque. 64 % de probabilité, ORANGE.

**Fix : le geolocation button déclenche un `activateDataSource("openmeteo")` complet.** Le buffer est purgé, un nouveau fetch Open-Meteo est effectué pour les vraies coordonnées, la prédiction est recalculée. Le tout en une seconde après le tap sur « Localiser ».

Cette découverte a donné lieu à un principe général :

> **Toute mutation de position doit invalider le buffer et déclencher un refetch complet.** Sans quoi le modèle prédit sur des données qui ne correspondent plus à l'utilisateur.

Idem pour l'auto-refresh GPS toutes les 10 minutes (setting configurable, cf. chapitre 19). Le seuil est fixé à 300 m de déplacement effectif pour éviter le refresh sur dérive GPS.

## Bug 3, le cache HTTP qui bloque les fixes

Ce bug fut le plus long à comprendre. Chaque fois qu'un fix était déployé côté serveur, le téléphone continuait à afficher l'ancien comportement, même après :

- Tap sur « Recharger l'app » dans les réglages (purge Cache API + désenregistre SW)
- Fermeture et réouverture de l'onglet Chrome
- Vider les données de navigation dans Chrome

Le badge de version en haut à droite affichait bien la nouvelle version, mais le comportement métier était inchangé.

Deux causes en cascade :

**Cause A**, le bind mount inode. J'éditais le Caddyfile pour changer les headers. Docker's bind mount se pinne à l'inode du fichier au moment du mount, et une édition qui recrée le fichier (write + rename atomique) change l'inode. Le container voit encore l'ancien Caddyfile. `docker compose exec caddy caddy reload` re-parse le fichier mais depuis le pointeur d'inode obsolète.

Fix : `docker compose restart caddy` à chaque déploiement, jamais `reload`. Documenté dans le Makefile parent du blog qui avait le même piège.

**Cause B**, le Cache-Control trop généreux. Ma config initiale était `max-age=3600` sur `js/`, `css/`, `models/`, `icons/`. Une heure de cache HTTP. Le service worker purge son Cache API, se ré-enregistre, mais Chrome sert le JavaScript depuis le cache HTTP pendant une heure sans même le redemander au serveur. Le badge version indique la nouvelle SW, le code s'exécute avec l'ancien JS.

Fix : `Cache-Control: no-cache` sur tout ce qui est code actif :

```
@sw path /sw.js
header @sw Cache-Control "no-cache"

@html path *.html /
header @html Cache-Control "no-cache"

@code path_regexp ^/(css|js)/.+
header @code Cache-Control "no-cache"

@heavy path_regexp ^/(models|icons)/.+
header @heavy Cache-Control "public, max-age=3600"
```

Les modèles ONNX et les icônes changent rarement et sont lourds, ils restent cachés. Le code doit toujours revalider.

## Autre découverte, la nav coupée sur foldable

Sur un Pixel 10 Pro Fold (aspect ratio 21:9 en portrait), la barre de navigation basse (Accueil / Live / Historique / Capteur) tombait sous le viewport visible de Chrome mobile. La cause : mon CSS utilisait `min-height: 100vh` sur les écrans, qui compte la hauteur totale de l'écran **sans retirer l'UI navigateur** (address bar, gesture handle).

Fix : basculer sur `min-height: 100dvh` (dynamic viewport height, standard CSS depuis 2022) et surtout `.nav { position: fixed; bottom: 0 }` pour ancrer la nav au viewport visible en permanence. Padding-bottom sur les screens = `76px + env(safe-area-inset-bottom) + 24px` pour laisser respirer le dernier élément au-dessus de la nav.

Cette classe de bugs n'apparaît jamais en développement desktop. Elle apparaît la première fois qu'un utilisateur ouvre l'app sur son téléphone en mode portrait. **Testing sur foldable est essentiel** pour tout produit visant la mobilité en 2026.

## L'importance de la version chip

Pour diagnostiquer si l'utilisateur est sur la version fraîche ou pas, j'ai ajouté une petite pastille bleu-cyan dans la topbar qui affiche le `CACHE_VERSION` en cours. Format `taranis-YYYYMMDD-HHMMSS`. Bumpé automatiquement à chaque `make deploy` via un script Makefile.

Cette pastille a économisé des heures de débogage à distance. Il suffit à l'utilisateur de lire les chiffres, et je sais immédiatement s'il est sur une version encore piégée par le cache HTTP ou pas.

**Enseignement produit** : quand une équipe débogue à distance, **rendre la version courante visible à l'utilisateur** est un investissement à ROI immédiat.

## Ce que le chapitre apporte

L'entraînement produit un objet mathématique. Le déploiement produit une expérience utilisateur. Entre les deux, il y a une couche de plomberie qui, mal réglée, peut faire dire au modèle des choses complètement fausses **sans que le modèle soit fautif**. Trois exemples ici :

- Un randonneur à Hazas de Cesto voyait ROUGE parce que Open-Meteo lui envoyait de la pression d'altitude déguisée en signal orageux
- Un randonneur qui change de position voyait la même prédiction que sa position précédente parce que le buffer ne se rafraîchissait pas
- Un randonneur qui rechargeait la page continuait à voir l'ancien code parce que Chrome mettait en cache un `Cache-Control: max-age=3600` sur le JS

Aucun de ces trois bugs n'est un problème de modèle. **Tous les trois font dire au modèle n'importe quoi.**

C'est le corollaire de l'enseignement du chapitre 17 sur la valeur du harnais : **un bon modèle mal servi produit de fausses alertes qu'aucun benchmark ne verra**. La validation terrain, en conditions réelles, avec un utilisateur qui bouge, reste le seul moyen d'attraper cette classe de bugs.

## Reproduire le diagnostic

Le script `scripts/test_noja_flicker.mjs` a été écrit précisément pour reproduire le premier bug en Node, sans avoir besoin de retourner à Meruelo. Il fetche Open-Meteo pour deux paires de coordonnées voisines et compare la sortie HGB. Si les probabilités divergent de plus de 5 points, le test échoue.

```bash
node scripts/test_noja_flicker.mjs
```

C'est un test de non-régression pour la « stabilité géographique » de la prédiction. Il tourne dans `make audit` maintenant.
