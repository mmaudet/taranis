# Dossier publishing/

Tout ce qu'il faut pour publier Taranis sur maudet.cloud, en deux formats :

- Un **article de blog** (catégorie Analyse).
- Le **carnet MkDocs complet** en sous-chemin `/taranis/`.

## Fichiers du dossier

- `taranis-article-blog.md` : article prêt à copier dans `src/content/blog/` du dépôt Astro.
- `deploy-to-maudet-cloud.md` : guide de déploiement pas à pas.
- `sync-mkdocs.sh` : script shell qui build le carnet et le pousse dans le dépôt blog local.
- `gha-mkdocs-to-blog.yml` : workflow GitHub Actions pour automatiser la synchro à chaque tag.

## Publication rapide, une fois

```bash
# 1. Article
cp publishing/taranis-article-blog.md ~/work/blog.maudet.cloud/src/content/blog/2026-07-13-taranis-jepa.md
cp docs/assets/step1_zoom_orage.png ~/work/blog.maudet.cloud/public/images/taranis/

# 2. Carnet
./publishing/sync-mkdocs.sh ~/work/blog.maudet.cloud

# 3. Menu : ajouter <a href="/taranis/">Taranis</a> ou une entrée équivalente
#    dans la config de navigation Astro.

# 4. Publier
cd ~/work/blog.maudet.cloud
git add . && git commit -m "feat: publish Taranis carnet and article" && git push
```

## Automatisation

Installer `gha-mkdocs-to-blog.yml` dans `.github/workflows/publish-mkdocs.yml` du dépôt Taranis, configurer les secrets `BLOG_REPO` et `BLOG_REPO_TOKEN`. À chaque tag `v*`, le carnet est reconstruit et poussé automatiquement dans le blog.
