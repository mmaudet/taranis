#!/usr/bin/env bash
# Synchronise le build MkDocs de Taranis dans le dépôt blog Astro sous public/taranis/.
#
# Usage :
#   ./publishing/sync-mkdocs.sh /chemin/vers/depot/blog
#
# Ce que fait le script :
#   1. Build MkDocs en strict mode (échoue sur warning).
#   2. Nettoie public/taranis/ dans le dépôt blog cible.
#   3. Copie le résultat.
#   4. Affiche un git diff résumé du blog.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <chemin-depot-blog>"
    echo "Exemple : $0 ~/work/blog.maudet.cloud"
    exit 1
fi

BLOG_ROOT="$1"
if [ ! -d "$BLOG_ROOT" ]; then
    echo "Erreur : $BLOG_ROOT n'est pas un dossier."
    exit 1
fi
if [ ! -d "$BLOG_ROOT/public" ]; then
    echo "Erreur : $BLOG_ROOT/public/ absent. Est-ce bien un dépôt Astro ?"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARANIS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Taranis root : $TARANIS_ROOT"
echo "Blog target  : $BLOG_ROOT"
echo

# 1. Build MkDocs en strict mode
cd "$TARANIS_ROOT"
echo "== Build MkDocs =="
uv sync --extra docs > /dev/null
uv run mkdocs build --strict --site-dir site

# 2. Nettoyage cible
TARGET="$BLOG_ROOT/public/taranis"
echo
echo "== Nettoyage $TARGET =="
rm -rf "$TARGET"
mkdir -p "$TARGET"

# 3. Copie
echo
echo "== Copie du site =="
cp -r site/* "$TARGET/"
N_FILES=$(find "$TARGET" -type f | wc -l)
SIZE=$(du -sh "$TARGET" | cut -f1)
echo "Copié : $N_FILES fichiers, $SIZE"

# 4. Résumé git
echo
echo "== git status côté blog =="
cd "$BLOG_ROOT"
git status --short public/taranis/ | head -20
echo
echo "Pour publier :"
echo "  cd $BLOG_ROOT"
echo "  git add public/taranis/"
echo "  git commit -m 'docs(taranis): update carnet'"
echo "  git push"
