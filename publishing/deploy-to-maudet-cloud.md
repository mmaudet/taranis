# Publier Taranis sur maudet.cloud

Deux livrables complémentaires à publier :

1. **Un article de blog** dans la catégorie `Analyse`, résumé de la démarche.
2. **Le carnet MkDocs complet** en sous-chemin `/taranis/` du site principal.

Le blog fonctionne sous **Astro** (générateur statique), déployé sur son propre hébergement. La procédure ci-dessous suppose que tu maintiens le dépôt du blog séparément et que tu peux y pousser des changements.

## 1. Article de blog

Le fichier `publishing/taranis-article-blog.md` est prêt à copier dans le dossier des articles de ton blog Astro (typiquement `src/content/blog/` ou `src/pages/blog/`).

**Étapes** :

```bash
# Dans le dépôt du blog Astro
cp <chemin-taranis>/publishing/taranis-article-blog.md \
   src/content/blog/2026-07-13-taranis-jepa-nowcasting-orage-souverain.md
```

Vérifier que le frontmatter YAML colle bien à ta configuration Astro (clés `title`, `pubDate`, `category`, `tags`, `slug`, `image`). Sinon adapter à la marge.

**Une image de couverture** est référencée par `image: "/images/taranis/step1_zoom_orage.png"`. Copier depuis le dépôt Taranis :

```bash
mkdir -p public/images/taranis
cp <chemin-taranis>/docs/assets/step1_zoom_orage.png \
   public/images/taranis/
```

## 2. Carnet MkDocs en sous-chemin /taranis/

Le carnet est un site MkDocs Material static. On le **build** dans le dépôt Taranis, on **copie** le résultat dans le dossier `public/taranis/` du dépôt blog. Astro sert alors automatiquement ce sous-chemin.

**Étape manuelle** :

```bash
# Dans le dépôt Taranis
uv sync --extra docs
uv run mkdocs build --site-dir site

# Copie dans le dépôt blog
cp -r site/* <chemin-blog>/public/taranis/
```

**Étape automatisée** (à préférer). Le script `publishing/sync-mkdocs.sh` fait la même chose avec quelques garde-fous (nettoyage, vérification, commit git optionnel).

## 3. Ajouter l'entrée « Taranis » (ou « JEPA ») au menu

Le menu principal du blog est défini dans un composant ou un fichier de configuration Astro. Chercher le fichier qui liste `Articles`, `Tags`, `Newsletter`, `À propos`, `Rechercher`. C'est probablement `src/components/Header.astro`, `src/layouts/Layout.astro`, ou `src/config/nav.ts`.

**Ajouter une entrée** du type :

```astro
<a href="/taranis/">Taranis</a>
```

ou dans une configuration TypeScript :

```typescript
export const navItems = [
  { label: "Articles", href: "/blog" },
  { label: "Tags", href: "/tags" },
  { label: "Newsletter", href: "/newsletter" },
  { label: "Taranis", href: "/taranis/" },  // nouveau
  { label: "À propos", href: "/about" },
  { label: "Rechercher", href: "/recherche" },
];
```

Le nom « JEPA » plutôt que « Taranis » est envisageable si tu veux positionner l'entrée sur l'architecture générale plutôt que sur ce projet particulier. Question éditoriale à trancher, techniquement les deux marchent.

## 4. Automatisation (facultative)

Si tu veux que la mise à jour du carnet soit synchronisée automatiquement à chaque tag Git dans le dépôt Taranis, utiliser le workflow GitHub Actions ci-après. Il :

1. Build le site MkDocs.
2. Ouvre une PR sur le dépôt blog pour mettre à jour `public/taranis/`.

C'est le pattern classique de « docs deploy on tag », zéro maintenance manuelle. Le workflow est dans `publishing/gha-mkdocs-to-blog.yml`.

## Résumé, trois commandes

Une fois tout branché :

```bash
# 1. Nouvel article
cp publishing/taranis-article-blog.md <blog>/src/content/blog/
cp docs/assets/step1_zoom_orage.png <blog>/public/images/taranis/

# 2. Mise à jour du carnet
./publishing/sync-mkdocs.sh <blog>

# 3. Publier
cd <blog> && git add . && git commit -m "feat: publish Taranis article and carnet" && git push
```

Le déploiement Astro se déclenche automatiquement à la réception du push (Cloudflare Pages, Netlify, ou ce que tu utilises).
