#!/usr/bin/env python3
"""
scripts/import_srk_batch.py - Batch Importer for algoUX Standard Ranklists (.srk.json).

Recursively discovers and imports official XCPC / CCPC / Provincial contest ranklists
from algoux/srk-collection into USTS standard contest JSON format, auto-identifying
USTS teams and anchoring external teams against the official XCPC-Elo baseline.
"""

from __future__ import annotations

import argparse
from hashlib import blake2s
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

if __package__:
    from .import_contest import import_contest, infer_start_time, read_srk
    from .validate_data import parse_contest_datetime
else:
    from import_contest import import_contest, infer_start_time, read_srk
    from validate_data import parse_contest_datetime


def slugify(text: str) -> str:
    """Create a slug that matches the contest filename schema."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not text.isascii():
        suffix = blake2s(text.encode("utf-8"), digest_size=4).hexdigest()
        return f"{(slug or 'contest')[:31]}_{suffix}"
    return slug[:40] or "contest"


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

    out.mkdir(parents=True, exist_ok=True)
    imported_files: list[Path] = []
    skipped_count = 0
    failed_count = 0
    with TemporaryDirectory(dir=out) as staging_dir:
        for file_path in srk_files:
            slug = slugify(file_path.stem.removesuffix(".srk"))
            staged = Path(staging_dir) / f"{slug}.json"
            try:
                _, meta = read_srk(file_path)
                start_time = meta.get("start_time") or infer_start_time(slug)
                start = parse_contest_datetime(start_time)
                if start is None:
                    raise ValueError(f"Invalid start_time: {start_time}")
                target = out / f"{start:%Y_%m_%d}_{slug}.json"
                if target.is_file() and not force_overwrite:
                    print(f"[BATCH-IMPORT] Skipping existing contest: {target.name}")
                    skipped_count += 1
                    continue

                result = import_contest(
                    input_path=file_path,
                    output_path=staged,
                    fmt="srk",
                    contest_type="EXTERNAL_XCPC",
                    start_time=start_time,
                    school_keywords=school_keywords,
                    default_external_elo=default_external_elo,
                    users_path=users_path,
                    external_elo_file=external_elo_map_path,
                    external_elo_mode="rank",
                )
                if force_overwrite:
                    staged.replace(target)
                else:
                    try:
                        # Publish the completed file only if the destination is absent.
                        target.hardlink_to(staged)
                    except FileExistsError:
                        print(f"[BATCH-IMPORT] Skipping existing contest: {target.name}")
                        skipped_count += 1
                        continue
                imported_files.append(target)
                standings = result["standings"]
                school_teams = sum(1 for team in standings if team["is_school"])
                print(
                    f"[BATCH-IMPORT] Successfully imported: {target.name} "
                    f"(Total: {len(standings)}, School: {school_teams})"
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                print(f"[BATCH-IMPORT] Error processing {file_path.name}: {exc}", file=sys.stderr)
                failed_count += 1
            finally:
                staged.unlink(missing_ok=True)

    print(f"[BATCH-IMPORT] Finished. Imported: {len(imported_files)}, "
          f"Skipped: {skipped_count}, Failed: {failed_count}, Total: {len(srk_files)}")
    if failed_count:
        raise ValueError(f"{failed_count} ranklist import(s) failed.")
    return imported_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch import .srk.json ranklists into USTS Rating System.")
    parser.add_argument("source_dir", type=str, help="Directory containing .srk.json ranklists")
    parser.add_argument("--output-dir", type=str, default="data/contests", help="Target contest directory (default: data/contests)")
    parser.add_argument("--users", type=str, default="data/users.csv", help="Path to users.csv")
    parser.add_argument("--elo-map", type=str, default="data/external_elo_map.json", help="Path to external_elo_map.json")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing contest files")
    args = parser.parse_args()

    try:
        batch_import_srk(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            users_path=args.users,
            external_elo_map_path=args.elo_map,
            force_overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"[BATCH-IMPORT] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
