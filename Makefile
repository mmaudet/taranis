# Taranis PWA deploy helpers.
#
# The static PWA files under taranis/infer/static are bind-mounted read-only
# into the blog.maudet.cloud Caddy container. Any edit is instantly served,
# no rebuild step. The only thing "deploy" does is bump the service worker
# cache version and restart Caddy so a stale mount can pick up new files.

BLOG_DIR := /home/mmaudet/work/blog.maudet.cloud
STATIC_DIR := taranis/infer/static

.PHONY: help deploy caddy-reload verify pwa-check bump-sw

help:
	@echo "Taranis PWA · deploy targets:"
	@echo "  make deploy        Bump SW cache + restart Caddy (fastest)"
	@echo "  make caddy-reload  Reload Caddy config without container recreate"
	@echo "  make verify        curl the taranis.maudet.cloud endpoints"
	@echo "  make pwa-check     List all critical PWA assets locally"
	@echo "  make bump-sw       Bump the SW CACHE_VERSION timestamp"

# Increment the SW cache version so clients auto-reload on next visit.
bump-sw:
	@ts=$$(date +%Y%m%d-%H%M%S); \
	 sed -i "s/const CACHE_VERSION = \".*\"/const CACHE_VERSION = \"taranis-$$ts\"/" \
	     $(STATIC_DIR)/sw.js; \
	 grep CACHE_VERSION $(STATIC_DIR)/sw.js

caddy-reload:
	@cd $(BLOG_DIR) && docker compose exec caddy caddy reload \
	    --config /etc/caddy/Caddyfile 2>&1 | tail -3

deploy: bump-sw caddy-reload
	@echo "✓ SW bumped, Caddy reloaded. taranis.maudet.cloud serves fresh files."

verify:
	@echo "== HTTPS via Host header (bypasses DNS mismatch) =="
	@curl -sk --resolve taranis.maudet.cloud:443:127.0.0.1 \
	    -o /dev/null -w "  index.html      %{http_code}\n" \
	    https://taranis.maudet.cloud/index.html || true
	@curl -sk --resolve taranis.maudet.cloud:443:127.0.0.1 \
	    -o /dev/null -w "  hgb_3ch_tw8.json %{http_code}\n" \
	    https://taranis.maudet.cloud/models/hgb_3ch_tw8.json || true
	@curl -sk --resolve taranis.maudet.cloud:443:127.0.0.1 \
	    -o /dev/null -w "  tsjepa_3ch.onnx  %{http_code}\n" \
	    https://taranis.maudet.cloud/models/tsjepa_3ch.onnx || true
	@echo ""
	@echo "== Public DNS state =="
	@host taranis.maudet.cloud | head -1
	@echo ""
	@echo "Server IP: $$(hostname -I | awk '{print $$1}')"
	@echo "If DNS != server IP, update the A record."

pwa-check:
	@echo "PWA assets in $(STATIC_DIR):"
	@ls -la $(STATIC_DIR)/*.html $(STATIC_DIR)/*.webmanifest \
	    $(STATIC_DIR)/*.js $(STATIC_DIR)/js/*.js $(STATIC_DIR)/css/*.css \
	    $(STATIC_DIR)/models/* $(STATIC_DIR)/icons/* 2>/dev/null
