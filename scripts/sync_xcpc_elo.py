#!/usr/bin/env python3
"""
scripts/sync_xcpc_elo.py - Official XCPC-Elo Data Synchronizer & Baseline Generator.

Fetches precalculated Elo ratings from the authoritative XCPC-Elo project
(https://zzzzzzyt.github.io/xcpc-elo/), extracts organization-level and player-level
Elo baselines, and generates `data/external_elo_map.json` and USTS player references.

Key features:
  - Fetches and parses `window.__ELO_DATA__` from live URL or local cache file.
  - Aggregates organization baseline Elo (top-3 varsity power mean).
  - Extracts individual player ratings for fine-grained team anchoring.
  - Can cross-reference Suzhou University of Science and Technology (苏州科技大学)
    historical contestants and export a template roster.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import urllib.request
from typing import Any

DEFAULT_ELO_DATA_URL = "https://zzzzzzyt.github.io/xcpc-elo/data.js"
DEFAULT_OUTPUT_MAP = "data/external_elo_map.json"
DEFAULT_OUTPUT_USTS = "data/users_xcpc_history.csv"


def fetch_xcpc_elo_data(
    source_url: str = DEFAULT_ELO_DATA_URL,
    local_cache_path: Path | str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Fetch and decode XCPC-Elo data payload.
    If local_cache_path is provided and exists, reads from cache instead.
    """
    if local_cache_path:
        cache_p = Path(local_cache_path)
        if cache_p.is_file():
            print(f"[SYNC] Reading XCPC-Elo data from local cache: {cache_p}")
            with cache_p.open("r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                if content.startswith("window.__ELO_DATA__ = "):
                    prefix = "window.__ELO_DATA__ = "
                    json_str = content[len(prefix):].rstrip(";\n ")
                    return json.loads(json_str)
                return json.loads(content)

    print(f"[SYNC] Fetching XCPC-Elo data from {source_url} (timeout={timeout}s)...")
    req = urllib.request.Request(
        source_url,
        headers={"User-Agent": "USTS-Rating-Engine/2.0 (+https://github.com/usts-rating)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw_bytes = response.read()

    text = raw_bytes.decode("utf-8")
    prefix = "window.__ELO_DATA__ = "
    if text.startswith(prefix):
        json_str = text[len(prefix):].rstrip(";\n ")
    else:
        json_str = text

    data = json.loads(json_str)
    print(f"[SYNC] Successfully parsed XCPC-Elo data. Total players: {len(data.get('players', []))}")
    return data


def extract_organization_elo(
    elo_data: dict[str, Any],
    top_k: int = 3,
) -> dict[str, int]:
    """
    Aggregate university/organization Elo ratings.
    For each organization, takes the arithmetic mean of its top-k players' latest ratings,
    which represents the university's varsity team baseline power.
    """
    players = elo_data.get("players", [])
    org_ratings: dict[str, list[int]] = {}

    for p in players:
        org = str(p.get("organization") or "").strip()
        if not org:
            continue
        hist = p.get("history", [])
        if not hist:
            continue
        # In xcpc-elo format, hist[-1][3] is latest Elo rating
        latest_r = hist[-1][3]
        if latest_r is None or not isinstance(latest_r, (int, float)):
            continue

        if org not in org_ratings:
            org_ratings[org] = []
        org_ratings[org].append(int(round(latest_r)))

    org_summary: dict[str, int] = {}
    for org, ratings in org_ratings.items():
        ratings.sort(reverse=True)
        top_slice = ratings[:top_k]
        avg_rating = sum(top_slice) / len(top_slice)
        org_summary[org] = int(round(avg_rating))

    return org_summary


def extract_player_elo(
    elo_data: dict[str, Any],
) -> dict[str, int]:
    """
    Extract player-level Elo ratings:
    Returns dict mapping "PlayerName" and "Org::PlayerName" to latest Elo rating.
    """
    players = elo_data.get("players", [])
    player_map: dict[str, int] = {}

    for p in players:
        name = str(p.get("name") or "").strip()
        org = str(p.get("organization") or "").strip()
        if not name:
            continue
        hist = p.get("history", [])
        if not hist:
            continue
        latest_r = hist[-1][3]
        if latest_r is None or not isinstance(latest_r, (int, float)):
            continue

        val = int(round(latest_r))
        if org:
            player_map[f"{org}::{name}"] = val
        if name not in player_map or val > player_map[name]:
            player_map[name] = val

    return player_map


def extract_usts_members(
    elo_data: dict[str, Any],
    school_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Extract all historical players from Suzhou University of Science and Technology.
    """
    keywords = school_keywords or ["苏州科技大学"]
    players = elo_data.get("players", [])
    usts_list: list[dict[str, Any]] = []

    for p in players:
        org = str(p.get("organization") or "").strip()
        if not any(kw in org for kw in keywords):
            continue

        hist = p.get("history", [])
        latest_r = hist[-1][3] if hist else None
        max_r = max((h[3] for h in hist if h[3] is not None), default=latest_r)

        usts_list.append({
            "name": p.get("name", ""),
            "organization": org,
            "latest_elo": latest_r,
            "max_elo": max_r,
            "contest_count": len(hist),
        })

    usts_list.sort(key=lambda x: -(x["latest_elo"] or 0))
    return usts_list


def sync_xcpc_elo(
    source_url: str = DEFAULT_ELO_DATA_URL,
    local_cache: Path | str | None = None,
    output_map_path: Path | str = DEFAULT_OUTPUT_MAP,
    output_usts_path: Path | str = DEFAULT_OUTPUT_USTS,
    sync_usts_roster: bool = True,
) -> dict[str, Any]:
    """
    Full sync pipeline: fetches data, extracts organization/player maps, and writes to disk.
    """
    elo_data = fetch_xcpc_elo_data(source_url, local_cache)
    org_map = extract_organization_elo(elo_data)
    player_map = extract_player_elo(elo_data)
    usts_members = extract_usts_members(elo_data)

    out_map_file = Path(output_map_path)
    out_map_file.parent.mkdir(parents=True, exist_ok=True)

    combined_map: dict[str, Any] = {
        "metadata": {
            "source": source_url,
            "generated_at": elo_data.get("generatedAt"),
            "total_organizations": len(org_map),
            "total_players": len(player_map),
        },
        "organizations": org_map,
        "players": player_map,
    }

    with out_map_file.open("w", encoding="utf-8") as f:
        json.dump(combined_map, f, ensure_ascii=False, indent=2)
    print(f"[SYNC] Exported external Elo map to: {out_map_file}")
    print(f"       Total organizations: {len(org_map)}, Total players: {len(player_map)}")

    if sync_usts_roster and usts_members:
        out_usts_file = Path(output_usts_path)
        out_usts_file.parent.mkdir(parents=True, exist_ok=True)
        with out_usts_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "organization", "latest_external_elo", "max_external_elo", "contest_count"])
            for m in usts_members:
                writer.writerow([
                    m["name"],
                    m["organization"],
                    m["latest_elo"],
                    m["max_elo"],
                    m["contest_count"],
                ])
        print(f"[SYNC] Exported USTS historical players ({len(usts_members)}) to: {out_usts_file}")

    return combined_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync official XCPC-Elo data into USTS Rating System.")
    parser.add_argument("--source", type=str, default=DEFAULT_ELO_DATA_URL, help="URL or path to XCPC-Elo data.js")
    parser.add_argument("--cache", type=str, default=None, help="Local cache path to read from / write to")
    parser.add_argument("--output-map", type=str, default=DEFAULT_OUTPUT_MAP, help="Path for external_elo_map.json")
    parser.add_argument("--output-usts", type=str, default=DEFAULT_OUTPUT_USTS, help="Path for users_xcpc_history.csv")
    parser.add_argument("--no-usts", action="store_true", help="Skip exporting USTS historical players")
    args = parser.parse_args()

    sync_xcpc_elo(
        source_url=args.source,
        local_cache=args.cache,
        output_map_path=args.output_map,
        output_usts_path=args.output_usts,
        sync_usts_roster=not args.no_usts,
    )


if __name__ == "__main__":
    main()
