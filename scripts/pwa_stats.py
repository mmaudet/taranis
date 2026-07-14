#!/usr/bin/env python3
"""Summarise Taranis PWA access logs from Caddy JSON output.

Usage:
    python scripts/pwa_stats.py            # last 24h
    python scripts/pwa_stats.py --today    # since 00:00 UTC today
    python scripts/pwa_stats.py --all      # everything on disk
    python scripts/pwa_stats.py --since 2026-07-14T14:00:00  # custom
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_LOCAL = Path("/home/mmaudet/work/blog.maudet.cloud/data/caddy/logs/taranis-access.log")
LOG_CONTAINER = "/data/logs/taranis-access.log"
CONTAINER = "blog-maudet-caddy"


def read_log_lines():
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


def parse_since(arg_since, today, all_):
    if arg_since:
        return datetime.fromisoformat(arg_since).astimezone(timezone.utc)
    if today:
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    if all_:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(hours=24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--all", action="store_true", dest="all_")
    ap.add_argument("--since", type=str, default=None,
                    help="ISO8601 UTC timestamp, e.g. 2026-07-14T14:00:00")
    args = ap.parse_args()

    since = parse_since(args.since, args.today, args.all_)
    label = f"since {since.isoformat()} UTC"

    lines = read_log_lines()
    if not lines:
        print("No lines in access log.")
        sys.exit(0)

    total = 0
    ips = Counter()
    ua = Counter()
    paths = Counter()
    status = Counter()
    hours = Counter()
    sessions = set()

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
        total += 1
        req = rec.get("request", {})
        client_ip = req.get("client_ip") or req.get("remote_ip") or "?"
        ua_str = (req.get("headers", {}).get("User-Agent") or ["?"])[0]
        uri = req.get("uri", "?")
        st = rec.get("status", 0)
        ips[client_ip] += 1
        ua[ua_str] += 1
        paths[uri] += 1
        status[st] += 1
        hours[when.strftime("%Y-%m-%d %H:00")] += 1
        sessions.add((client_ip, ua_str))

    print(f"== Taranis PWA access stats ({label}) ==\n")
    if total == 0:
        print("No requests in that window.")
        return
    print(f"Total requests    : {total}")
    print(f"Unique client IPs : {len(ips)}")
    print(f"Unique user agents: {len(ua)}")
    print(f"Distinct sessions : {len(sessions)}  (unique IP + UA)")
    print()

    print("-- Top paths --")
    for path, n in paths.most_common(15):
        print(f"  {n:5d}  {path}")
    print()

    print("-- Status codes --")
    for st, n in sorted(status.items()):
        print(f"  {st}: {n}")
    print()

    print("-- Top user agents (truncated) --")
    for ua_str, n in ua.most_common(6):
        print(f"  {n:5d}  {ua_str[:80]}")
    print()

    print("-- Requests per hour (UTC) --")
    for h in sorted(hours):
        bar = "█" * min(60, hours[h])
        print(f"  {h}   {hours[h]:5d}  {bar}")


if __name__ == "__main__":
    main()
