#!/usr/bin/env python3
"""
scripts/import_srk_batch.py - Batch Importer for algoUX Standard Ranklists (.srk.json).

Recursively discovers and imports official XCPC / CCPC / Provincial contest ranklists
from algoux/srk-collection into USTS standard contest JSON format, auto-identifying
USTS teams and anchoring external teams against the official XCPC-Elo baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

from scripts.import_contest import import_contest, DEFAULT_SCHOOL_KEYWORDS
from scripts.validate_data import validate_contest_dict, validate_users_csv


def slugify(text: str) -> str:
    """Create filesystem-safe slug from title or filename."""
    s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:40] if s else "contest"


def batch_import_srk(
    source_dir: Path | str,
    output_dir: Path | str = "data/contests",
    users_path: Path | str = "data/users.csv",
    external_elo_map_path: Path | str = "data/external_elo_map.json",
    school_keywords: list[str] | None = None,
    default_external_elo: int = 1500,
    force_overwrite: bool = False,
) -> list[Path]:
    """
    Scan source_dir for *.srk.json files, parse and convert each into standard contest JSON.
    Returns list of successfully generated contest files.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        raise NotADirectoryError(f"Source directory not found: {src}")

    # Find all .srk.json files
    srk_files = sorted(list(src.rglob("*.srk.json")) + list(src.rglob("*.srk")))
    # Also support .json files that match SRK structure if explicit
    if not srk_files:
        srk_files = sorted(src.glob("*.json"))

    if not srk_files:
        print(f"[BATCH-IMPORT] No .srk.json files found under: {src}")
        return []

    print(f"[BATCH-IMPORT] Found {len(srk_files)} SRK files to process from {src}")

    keywords = school_keywords or DEFAULT_SCHOOL_KEYWORDS
    imported_files: list[Path] = []
    success_count = 0
    skipped_count = 0

    for file_path in srk_files:
        try:
            # Let import_contest parse the srk metadata and determine start_time
            # Temporary dry-run output path
            stem = file_path.stem.replace(".srk", "")
            target_slug = slugify(stem)

            # We create an initial target filename based on file stem
            target_file = out / f"{target_slug}.json"
            if target_file.is_file() and not force_overwrite:
                print(f"[BATCH-IMPORT] Skipping existing contest: {target_file.name}")
                skipped_count += 1
                continue

            result = import_contest(
                input_path=file_path,
                output_path=target_file,
                fmt="srk",
                contest_type="EXTERNAL_XCPC",
                school_keywords=keywords,
                default_external_elo=default_external_elo,
                users_path=users_path,
                external_elo_file=external_elo_map_path if Path(external_elo_map_path).is_file() else None,
                external_elo_mode="rank",
            )

            # Check if start_time allows a better canonical YYYY_MM_DD filename
            st = result.get("start_time", "")
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", st)
            if m:
                canonical_name = f"{m.group(1)}_{m.group(2)}_{m.group(3)}_{target_slug}.json"
                canonical_file = out / canonical_name
                if canonical_file != target_file:
                    target_file.replace(canonical_file)
                    target_file = canonical_file

            imported_files.append(target_file)
            success_count += 1
            school_teams = sum(1 for t in result.get("standings", []) if t.get("is_school"))
            print(f"[BATCH-IMPORT] Successfully imported: {target_file.name} (Total: {len(result.get('standings', []))}, School: {school_teams})")

        except Exception as e:
            print(f"[BATCH-IMPORT] Error processing {file_path.name}: {e}", file=sys.stderr)

    print(f"[BATCH-IMPORT] Finished. Imported: {success_count}, Skipped: {skipped_count}, Total: {len(srk_files)}")
    return imported_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch import .srk.json ranklists into USTS Rating System.")
    parser.add_argument("source_dir", type=str, help="Directory containing .srk.json ranklists")
    parser.add_argument("--output-dir", type=str, default="data/contests", help="Target contest directory (default: data/contests)")
    parser.add_argument("--users", type=str, default="data/users.csv", help="Path to users.csv")
    parser.add_argument("--elo-map", type=str, default="data/external_elo_map.json", help="Path to external_elo_map.json")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing contest files")
    args = parser.parse_args()

    batch_import_srk(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        users_path=args.users,
        external_elo_map_path=args.elo_map,
        force_overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
