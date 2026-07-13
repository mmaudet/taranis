#!/usr/bin/env bash
# End-to-end audit of every layer between the code and the phone.
# Fails loudly on any regression. Run: bash scripts/pwa_e2e_audit.sh

set -u

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
NC=$'\033[0m'

pass=0; fail=0
check() {
    local label="$1"; shift
    if "$@"; then
        echo "  ${GREEN}✓${NC} $label"
        pass=$((pass + 1))
    else
        echo "  ${RED}✗${NC} $label"
        fail=$((fail + 1))
    fi
}

# --- LAYER 1: local files on disk ----------------------------------------
echo "== 1. Local disk files =="

STATIC=/home/mmaudet/work/taranis/taranis/infer/static
LOC_APP=$STATIC/js/app.js
LOC_CSS=$STATIC/css/app.css
LOC_HTML=$STATIC/index.html
LOC_I18N=$STATIC/js/i18n.js
LOC_SW=$STATIC/sw.js

check "app.js: home.gps_position at both sites" \
    bash -c "test \$(grep -c 'home.gps_position' $LOC_APP) -eq 2"
check "app.js: no #place-name -> home.default_place" \
    bash -c "! grep '#place-name.*home.default_place' $LOC_APP"
check "app.js: installCrosshair uses overlay id" \
    bash -c "grep -q 'live-chart-overlay' $LOC_APP"
check "app.js: pointerdown + touchstart handlers present" \
    bash -c "grep -q 'pointerdown' $LOC_APP && grep -q 'touchstart' $LOC_APP"
check "css: no overflow:hidden on hero-pressure" \
    bash -c "! grep -Pzo '(?s)\.hero-pressure \{[^}]*overflow:\s*hidden' $LOC_CSS >/dev/null"
check "css: chart-overlay class present" \
    bash -c "grep -q '\.chart-overlay' $LOC_CSS"
check "css: place-detail nowrap + ellipsis" \
    bash -c "grep -q 'white-space: nowrap' $LOC_CSS && grep -q 'text-overflow: ellipsis' $LOC_CSS"
check "html: 2x chart-overlay div" \
    bash -c "test \$(grep -c 'chart-overlay' $LOC_HTML) -ge 2"
check "html: sensor group headers present" \
    bash -c "grep -q 'sensor-group-header' $LOC_HTML && grep -q 'context-header' $LOC_HTML"
check "i18n: gps_position in 5 langs" \
    bash -c "test \$(grep -c 'home.gps_position' $LOC_I18N) -eq 5"

# --- LAYER 2: served bundle over HTTPS ------------------------------------
echo ""
echo "== 2. HTTPS served bundle (via correct IP) =="

BASE="https://taranis.maudet.cloud"
RESOLVE="--resolve taranis.maudet.cloud:443:85.10.205.244"

FETCH="curl -sk $RESOLVE $BASE"

reach() { test "$(curl -sk $RESOLVE -o /dev/null -w '%{http_code}' $BASE/$1)" = 200; }
body() { curl -sk $RESOLVE "$BASE/$1"; }
hdrs() { curl -sIk $RESOLVE "$BASE/$1"; }

check "index.html reachable" reach index.html
check "sw.js reachable" reach sw.js
check "manifest reachable" reach manifest.webmanifest
check "app.js reachable" reach js/app.js
check "i18n.js reachable" reach js/i18n.js
check "hgb_3ch_tw8.json reachable" reach models/hgb_3ch_tw8.json
check "tsjepa_3ch.onnx reachable" reach models/tsjepa_3ch.onnx

echo ""
served_app=$(body js/app.js)
served_i18n=$(body js/i18n.js)
served_index=$(body index.html)
served_css=$(body css/app.css)

check "served app.js has home.gps_position x2" \
    bash -c "echo \"\$1\" | grep -c 'home.gps_position' | grep -q '^2$'" _ "$served_app"
check "served app.js has no #place-name -> default_place" \
    bash -c "! echo \"\$1\" | grep -q '#place-name.*home.default_place'" _ "$served_app"
check "served app.js has installCrosshair" \
    bash -c "echo \"\$1\" | grep -q 'installCrosshair'" _ "$served_app"
check "served app.js has chart-overlay id" \
    bash -c "echo \"\$1\" | grep -q 'chart-overlay'" _ "$served_app"
check "served index.html has 2 chart-overlay div" \
    bash -c "test \$(echo \"\$1\" | grep -c 'chart-overlay') -ge 2" _ "$served_index"
check "served i18n.js has gps_position x5" \
    bash -c "test \$(echo \"\$1\" | grep -c 'home.gps_position') -eq 5" _ "$served_i18n"
check "served css: no overflow:hidden on hero-pressure block" \
    bash -c "! echo \"\$1\" | tr -d '\n' | grep -Po '\\.hero-pressure\\s*\\{[^}]*overflow:\\s*hidden' >/dev/null" _ "$served_css"

echo ""
echo "== 3. Cache-Control headers (the usual suspect) =="

hdr_of() { hdrs "$1" | grep -i "cache-control:" | head -1 | tr -d '\r'; }
echo "  sw.js         → $(hdr_of sw.js)"
echo "  index.html    → $(hdr_of index.html)"
echo "  js/app.js     → $(hdr_of js/app.js)"
echo "  css/app.css   → $(hdr_of css/app.css)"
echo "  models/*.json → $(hdr_of models/hgb_3ch_tw8.json)"

echo ""
served_sw_hdr=$(hdrs sw.js | grep -i cache-control)
served_html_hdr=$(hdrs index.html | grep -i cache-control)
served_appjs_hdr=$(hdrs js/app.js | grep -i cache-control)
served_appcss_hdr=$(hdrs css/app.css | grep -i cache-control)
check "sw.js Cache-Control is no-cache" \
    bash -c "echo \"\$1\" | grep -qi 'no-cache'" _ "$served_sw_hdr"
check "index.html Cache-Control is no-cache" \
    bash -c "echo \"\$1\" | grep -qi 'no-cache'" _ "$served_html_hdr"
check "js/app.js Cache-Control is no-cache (so reload gets fresh code)" \
    bash -c "echo \"\$1\" | grep -qi 'no-cache'" _ "$served_appjs_hdr"
check "css/app.css Cache-Control is no-cache" \
    bash -c "echo \"\$1\" | grep -qi 'no-cache'" _ "$served_appcss_hdr"

# --- LAYER 4: end-to-end prediction on Noja ------------------------------
echo ""
echo "== 4. End-to-end Node simulation on Noja =="
cd /home/mmaudet/work/taranis
node scripts/test_noja_flicker.mjs 2>&1 | grep -E "^Open-Meteo only, HGB-Tw8" | head -1

# --- LAYER 5: SW version -------------------------------------------------
echo ""
echo "== 5. SW version served =="
sw_ver=$(body sw.js | grep -oE 'taranis-[0-9]+-[0-9]+' | head -1)
echo "  ${GREEN}CACHE_VERSION on server: ${sw_ver}${NC}"

# --- LAYER 6: mask stale HTTP cache risk --------------------------------
echo ""
echo "== 6. Verdict =="
echo "  passed: $pass    failed: $fail"

if [ $fail -gt 0 ]; then
    echo "${RED}Some checks failed. Fix before shipping.${NC}"
    exit 1
fi
echo "${GREEN}All server-side checks pass. If the phone still sees old code, the culprit is browser HTTP cache — see next fix.${NC}"
