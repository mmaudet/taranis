#!/usr/bin/env python3
"""Rich analytics for the Taranis PWA from Caddy JSON access logs.

Souverain by construction: no third-party API call, no client-side
tracker. All lookups (country, device, browser, OS) are done locally
on the server from headers already logged by Caddy.

Usage:
    python scripts/pwa_stats.py            # last 24h
    python scripts/pwa_stats.py --today    # since 00:00 UTC today
    python scripts/pwa_stats.py --all      # up to 30 days on disk
    python scripts/pwa_stats.py --since 2026-07-14T14:00:00
    python scripts/pwa_stats.py --exclude-bots
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_LOCAL = Path("/home/mmaudet/work/blog.maudet.cloud/data/caddy/logs/taranis-access.log")
LOG_CONTAINER = "/data/logs/taranis-access.log"
CONTAINER = "blog-maudet-caddy"
GEOIP_DB = Path("/home/mmaudet/work/taranis/data/geoip/dbip-country-lite.mmdb")


def read_log_lines():
    """Read the log via local mount if permitted, else via docker exec."""
    if LOG_LOCAL.exists():
        try:
            return LOG_LOCAL.read_text(errors="replace").splitlines()
        except PermissionError:
            pass
    try:
        out = subprocess.check_output(
            ["docker", "exec", CONTAINER, "cat", LOG_CONTAINER],
            stderr=subprocess.DEVNULL,
        )
        return out.decode(errors="replace").splitlines()
    except Exception as e:
        print(f"Cannot read access log: {e}", file=sys.stderr)
        sys.exit(1)


def parse_since(arg_since: str | None, today: bool, all_: bool) -> datetime:
    if arg_since:
        return datetime.fromisoformat(arg_since).astimezone(timezone.utc)
    if today:
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    if all_:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(hours=24)


# ---------------------------------------------------------------------------
# GeoIP
# ---------------------------------------------------------------------------

_GEO_READER = None
_COUNTRY_NAMES = {
    "FR": "France", "ES": "Espagne", "IT": "Italie", "DE": "Allemagne",
    "GB": "Royaume-Uni", "UK": "Royaume-Uni", "US": "États-Unis",
    "CA": "Canada", "CH": "Suisse", "BE": "Belgique", "NL": "Pays-Bas",
    "PT": "Portugal", "AT": "Autriche", "SE": "Suède", "NO": "Norvège",
    "DK": "Danemark", "FI": "Finlande", "IE": "Irlande", "PL": "Pologne",
    "CZ": "Tchéquie", "IN": "Inde", "BR": "Brésil", "MX": "Mexique",
    "AR": "Argentine", "AU": "Australie", "JP": "Japon", "CN": "Chine",
    "KR": "Corée du Sud", "TW": "Taïwan", "SG": "Singapour", "MA": "Maroc",
    "TN": "Tunisie", "DZ": "Algérie", "SN": "Sénégal", "CI": "Côte d'Ivoire",
    "IL": "Israël", "TR": "Turquie", "RU": "Russie", "UA": "Ukraine",
    "GR": "Grèce", "RO": "Roumanie", "BG": "Bulgarie", "HU": "Hongrie",
    "IS": "Islande", "LU": "Luxembourg", "MC": "Monaco", "AD": "Andorre",
    "LI": "Liechtenstein", "SM": "Saint-Marin", "VA": "Vatican",
    "MT": "Malte", "CY": "Chypre", "EE": "Estonie", "LV": "Lettonie",
    "LT": "Lituanie", "SI": "Slovénie", "SK": "Slovaquie", "HR": "Croatie",
    "BA": "Bosnie", "RS": "Serbie", "ME": "Monténégro", "MK": "Macédoine",
    "AL": "Albanie", "XK": "Kosovo", "MD": "Moldavie", "BY": "Biélorussie",
    "GE": "Géorgie", "AM": "Arménie", "AZ": "Azerbaïdjan", "TH": "Thaïlande",
    "VN": "Vietnam", "PH": "Philippines", "ID": "Indonésie", "MY": "Malaisie",
    "NZ": "Nouvelle-Zélande", "ZA": "Afrique du Sud", "EG": "Égypte",
}


def geo_country(ip: str) -> tuple[str | None, str | None]:
    """Return (iso_code, name_fr) for the given IPv4/IPv6 address."""
    global _GEO_READER
    if not ip:
        return None, None
    if _GEO_READER is None:
        if not GEOIP_DB.exists():
            return None, None
        try:
            import maxminddb
            _GEO_READER = maxminddb.open_database(str(GEOIP_DB))
        except Exception:
            return None, None
    try:
        rec = _GEO_READER.get(ip)
    except Exception:
        return None, None
    if not rec:
        return None, None
    iso = (rec.get("country", {}) or {}).get("iso_code")
    if not iso:
        return None, None
    return iso, _COUNTRY_NAMES.get(iso, iso)


# ---------------------------------------------------------------------------
# User-Agent parsing (device / OS / browser) — minimal, no external lib
# ---------------------------------------------------------------------------

_BOT_PATTERNS = re.compile(
    r"(bot|crawl|spider|slurp|facebookexternalhit|linkedinbot|twitterbot|"
    r"whatsapp|telegram|discord|slackbot|preview|scan|check|monitor|pingdom|"
    r"headless|python-requests|curl|wget|go-http-client|httpx)",
    re.IGNORECASE,
)


def parse_ua(ua: str) -> dict:
    """Return {device, os, browser, is_bot} from a User-Agent string."""
    if not ua or ua == "?":
        return {"device": "?", "os": "?", "browser": "?", "is_bot": False}

    is_bot = bool(_BOT_PATTERNS.search(ua))

    if re.search(r"iPad|Tablet|Kindle", ua, re.I):
        device = "tablette"
    elif re.search(r"Mobile|Android|iPhone|iPod|BlackBerry|IEMobile|Opera Mini", ua, re.I):
        device = "mobile"
    else:
        device = "desktop"

    if re.search(r"Windows NT", ua):
        os_name = "Windows"
    elif re.search(r"Mac OS X|Macintosh", ua):
        os_name = "macOS"
    elif re.search(r"Android", ua, re.I):
        os_name = "Android"
    elif re.search(r"iPhone|iPad|iPod|iOS", ua, re.I):
        os_name = "iOS"
    elif re.search(r"Linux", ua):
        os_name = "Linux"
    else:
        os_name = "?"

    if re.search(r"Edg/", ua):
        browser = "Edge"
    elif re.search(r"OPR/|Opera", ua, re.I):
        browser = "Opera"
    elif re.search(r"Firefox/", ua):
        browser = "Firefox"
    elif re.search(r"Chrome/", ua) and not re.search(r"Chromium", ua):
        browser = "Chrome"
    elif re.search(r"Safari/", ua) and re.search(r"Version/", ua):
        browser = "Safari"
    elif re.search(r"curl/", ua, re.I):
        browser = "curl"
    else:
        browser = "?"

    return {"device": device, "os": os_name, "browser": browser, "is_bot": is_bot}


# ---------------------------------------------------------------------------
# Referrer categorisation
# ---------------------------------------------------------------------------


def parse_referrer(ref: str) -> str:
    if not ref:
        return "direct"
    ref = ref.lower()
    if "blog.maudet.cloud" in ref:
        return "blog article"
    if "linkedin" in ref:
        return "LinkedIn"
    if "twitter.com" in ref or "x.com/" in ref or "t.co/" in ref:
        return "X / Twitter"
    if "facebook" in ref or "fb.com" in ref:
        return "Facebook"
    if "reddit" in ref:
        return "Reddit"
    if "bsky" in ref or "bluesky" in ref:
        return "Bluesky"
    if "mastodon" in ref or "framapiaf" in ref or "piaille" in ref:
        return "Mastodon"
    if "hackernews" in ref or "news.ycombinator" in ref:
        return "Hacker News"
    if "google" in ref:
        return "Google"
    if "duckduckgo" in ref or "ddg" in ref:
        return "DuckDuckGo"
    if "bing" in ref:
        return "Bing"
    if "github" in ref:
        return "GitHub"
    if "taranis.maudet.cloud" in ref:
        return "self (interne PWA)"
    return "other referrer"


def parse_lang(header: str) -> str:
    if not header:
        return "?"
    first = header.split(",")[0].split(";")[0].strip()
    if not first:
        return "?"
    return first.split("-")[0].lower()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--all", action="store_true", dest="all_")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO8601 UTC, e.g. 2026-07-14T14:00:00")
    ap.add_argument("--exclude-bots", action="store_true",
                    help="Skip requests identified as bots / crawlers")
    args = ap.parse_args()

    since = parse_since(args.since, args.today, args.all_)
    label = f"since {since.isoformat()} UTC"

    lines = read_log_lines()
    if not lines:
        print("Log empty.")
        sys.exit(0)

    total = 0
    bots_skipped = 0
    ips = Counter()
    countries = Counter()
    devices = Counter()
    oses = Counter()
    browsers = Counter()
    referrers = Counter()
    languages = Counter()
    paths = Counter()
    status = Counter()
    hours = Counter()
    sessions: dict[tuple, list[datetime]] = defaultdict(list)

    funnel_index = set()
    funnel_model = set()
    funnel_manifest = set()
    funnel_sw = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = rec.get("ts")
        if not ts:
            continue
        try:
            when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when < since:
            continue

        req = rec.get("request", {})
        headers = req.get("headers", {})
        ua = (headers.get("User-Agent") or ["?"])[0]
        info = parse_ua(ua)
        if args.exclude_bots and info["is_bot"]:
            bots_skipped += 1
            continue

        total += 1
        ip = req.get("client_ip") or req.get("remote_ip") or "?"
        uri = req.get("uri", "?")
        st = rec.get("status", 0)
        ref = (headers.get("Referer") or [""])[0]
        lang = parse_lang((headers.get("Accept-Language") or [""])[0])

        ips[ip] += 1
        iso, name = geo_country(ip)
        countries[name or "?"] += 1
        devices[info["device"]] += 1
        oses[info["os"]] += 1
        browsers[info["browser"]] += 1
        referrers[parse_referrer(ref)] += 1
        languages[lang] += 1
        paths[uri] += 1
        status[st] += 1
        hours[when.strftime("%Y-%m-%d %H:00")] += 1

        session_key = (ip, ua)
        sessions[session_key].append(when)
        if uri in ("/", "/index.html"):
            funnel_index.add(session_key)
        if uri.startswith("/models/"):
            funnel_model.add(session_key)
        if uri == "/manifest.webmanifest":
            funnel_manifest.add(session_key)
        if uri == "/sw.js":
            funnel_sw.add(session_key)

    print(f"== Taranis PWA access stats ({label}) ==\n")
    if total == 0:
        print("No requests in that window.")
        if bots_skipped:
            print(f"({bots_skipped} bot requests filtered out)")
        return

    print(f"Total requests    : {total}")
    print(f"Unique client IPs : {len(ips)}")
    print(f"Distinct sessions : {len(sessions)}  (unique IP + UA)")
    if bots_skipped:
        print(f"Bots filtered     : {bots_skipped}")
    print()

    durations = []
    for ts_list in sessions.values():
        if len(ts_list) >= 2:
            durations.append((max(ts_list) - min(ts_list)).total_seconds())
    if durations:
        durations.sort()
        median = durations[len(durations) // 2]
        p90 = durations[int(len(durations) * 0.9)]
        max_ = max(durations)
        print(f"Session length    : median {median:.0f} s   p90 {p90:.0f} s   max {max_:.0f} s")
        print(f"Multi-request sessions: {len(durations)} / {len(sessions)}")
        print()

    if any(k != "?" for k in countries):
        print("-- Pays d'origine (via DB-IP country-lite MMDB) --")
        for name, n in countries.most_common(15):
            print(f"  {n:5d}  {name}")
        print()

    print("-- Device --")
    for k, n in devices.most_common():
        print(f"  {n:5d}  {k}")
    print()

    print("-- OS --")
    for k, n in oses.most_common(6):
        print(f"  {n:5d}  {k}")
    print()

    print("-- Browser --")
    for k, n in browsers.most_common(6):
        print(f"  {n:5d}  {k}")
    print()

    print("-- Referrer --")
    for k, n in referrers.most_common(10):
        print(f"  {n:5d}  {k}")
    print()

    print("-- Langue (Accept-Language) --")
    for k, n in languages.most_common(8):
        print(f"  {n:5d}  {k}")
    print()

    n_index = len(funnel_index)
    n_sw = len(funnel_sw)
    n_manifest = len(funnel_manifest)
    n_model = len(funnel_model)
    total_sessions = len(sessions)
    print("-- Engagement funnel --")

    def bar(n, denom):
        pct = (100 * n / max(1, denom))
        return f"{n:4d} ({pct:5.1f} %)"

    print(f"  Sessions total          : {total_sessions}")
    print(f"  Landed on index         : {bar(n_index, total_sessions)}")
    print(f"  Fetched service worker  : {bar(n_sw, total_sessions)}")
    print(f"  Fetched manifest        : {bar(n_manifest, total_sessions)}  <- candidat install PWA")
    print(f"  Fetched a model file    : {bar(n_model, total_sessions)}  <- utilisateur engagé")
    print()

    print("-- Top paths --")
    for path, n in paths.most_common(15):
        print(f"  {n:5d}  {path}")
    print()

    print("-- Status codes --")
    for st, n in sorted(status.items()):
        print(f"  {st}: {n}")
    print()

    print("-- Requests per hour (UTC) --")
    for h in sorted(hours):
        bar_str = "█" * min(60, hours[h])
        print(f"  {h}   {hours[h]:5d}  {bar_str}")


if __name__ == "__main__":
    main()
