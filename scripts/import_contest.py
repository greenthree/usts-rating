#!/usr/bin/env python3
"""
scripts/import_contest.py - Multi-Format Contest Standings Importer.

Converts scoreboard exports from Generic CSV, DOMjudge API JSON, or PTA (拼题A)
into standard USTS contest JSON format (data/contests/YYYY_MM_DD_<slug>.json),
and automatically validates the output against validate_data.py rules.

Supported formats:
  - csv: Generic CSV with headers in English or Chinese.
  - domjudge: DOMjudge scoreboard JSON (with optional teams.json).
  - pta: PTA rankings JSON or CSV export.
  - srk: algoUX Standard Ranklist JSON (.srk.json) from algoux/srk-collection.
  - auto: Automatically detects the format based on file content.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

# Import validation logic directly from validate_data.py
try:
    from scripts.validate_data import (
        STUDENT_ID_PATTERN,
        VALID_CONTEST_TYPES,
        parse_bool,
        parse_contest_datetime,
        validate_contest_dict,
        validate_users_csv,
    )
except ImportError:
    # If invoked directly when scripts/ is current dir
    from validate_data import (
        STUDENT_ID_PATTERN,
        VALID_CONTEST_TYPES,
        parse_bool,
        parse_contest_datetime,
        validate_contest_dict,
        validate_users_csv,
    )


DEFAULT_SCHOOL_KEYWORDS = ["苏州科技大学", "USTS", "苏科大", "苏州科大"]
KNOWN_COACH_NAMES = {"程成", "景国良", "教练", "coach", "trainer"}


def split_members(raw: Any) -> list[str]:
    """Extract list of student IDs from string or list."""
    if isinstance(raw, list):
        return [str(m).strip() for m in raw if str(m).strip()]
    if not isinstance(raw, str):
        return []
    # Split by comma, semicolon, slash, pipe, or whitespace
    parts = re.split(r"[,;/\s|]+", raw.strip())
    return [p for p in parts if p]


def is_school_team(
    team_name: str,
    affiliation: str,
    members: list[str],
    school_keywords: list[str],
    registered_student_ids: set[str] | None = None,
    contest_type: str = "INTERNAL_SCHOOL",
) -> bool:
    """Determine whether a team belongs to USTS."""
    if contest_type == "INTERNAL_SCHOOL":
        return True

    text_to_check = f"{team_name} {affiliation}".lower()
    for kw in school_keywords:
        if kw.lower() in text_to_check:
            return True

    if registered_student_ids and members:
        # If any member is a known registered student, consider it school
        if any(m in registered_student_ids for m in members):
            return True

    return False


# ==========================================
# Format Importers
# ==========================================

def parse_generic_csv(
    file_path: Path,
    school_keywords: list[str],
    registered_student_ids: set[str] | None,
    contest_type: str,
    default_external_elo: int = 1500,
) -> list[dict[str, Any]]:
    """Parse Generic CSV scoreboard."""
    standings: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty or headerless CSV file: {file_path}")

        # Normalize header keys to lowercase stripped
        field_map: dict[str, str] = {}
        for fn in reader.fieldnames:
            norm = fn.strip().lower()
            field_map[norm] = fn

        def get_val(row: dict[str, str], *aliases: str) -> str:
            for a in aliases:
                norm_a = a.strip().lower()
                if norm_a in field_map:
                    val = row.get(field_map[norm_a], "")
                    if val is not None and str(val).strip() != "":
                        return str(val).strip()
            return ""

        for row_idx, row in enumerate(reader, start=1):
            if not row or all(v.strip() == "" for v in row.values() if isinstance(v, str)):
                continue

            team_name = get_val(row, "team_name", "team", "name", "队名", "队伍", "队伍名称", "选手")
            if not team_name:
                team_name = f"Team_{row_idx}"

            raw_members = get_val(row, "members", "member", "student_id", "student_ids", "队员", "学号", "队员学号")
            members = split_members(raw_members)

            affiliation = get_val(row, "affiliation", "organization", "school", "school_name", "学校", "机构", "单位")

            # is_school
            is_school_raw = get_val(row, "is_school", "school_team", "是否本校", "本校")
            if is_school_raw:
                parsed_bool = parse_bool(is_school_raw)
                if parsed_bool is None:
                    is_school = is_school_team(
                        team_name, affiliation, members, school_keywords,
                        registered_student_ids, contest_type
                    )
                else:
                    is_school = parsed_bool
            else:
                is_school = is_school_team(
                    team_name, affiliation, members, school_keywords,
                    registered_student_ids, contest_type
                )

            # solved
            solved_raw = get_val(row, "solved", "num_solved", "score", "题数", "解题数", "通过数", "得分")
            try:
                solved = int(float(solved_raw)) if solved_raw else 0
            except ValueError:
                solved = 0

            # penalty
            penalty_raw = get_val(row, "penalty", "total_time", "time", "罚时", "总用时", "耗时")
            try:
                penalty = int(float(penalty_raw)) if penalty_raw else 0
            except ValueError:
                penalty = 0

            # rank
            rank_raw = get_val(row, "rank", "ranking", "排名", "名次")
            rank: int | None = None
            if rank_raw:
                try:
                    rank = int(rank_raw)
                except ValueError:
                    pass

            # external_elo
            elo_raw = get_val(row, "external_elo", "elo", "外部elo", "外部积分")
            external_elo: int | None = None
            if not is_school:
                if elo_raw:
                    try:
                        external_elo = int(elo_raw)
                    except ValueError:
                        external_elo = default_external_elo
                else:
                    external_elo = default_external_elo

            # is_star
            star_raw = get_val(row, "is_star", "star", "打星", "是否打星")
            is_star = parse_bool(star_raw) if star_raw else False
            if is_star is None:
                is_star = False
            if team_name.startswith("*"):
                is_star = True

            standings.append({
                "team_name": team_name,
                "is_school": is_school,
                "members": members if is_school else [],
                "external_elo": external_elo,
                "solved": solved,
                "penalty": penalty,
                "rank": rank,
                "is_star": is_star,
            })

    return standings


def parse_domjudge(
    scoreboard_file: Path,
    teams_file: Path | None,
    school_keywords: list[str],
    registered_student_ids: set[str] | None,
    contest_type: str,
    default_external_elo: int = 1500,
) -> list[dict[str, Any]]:
    """Parse DOMjudge API scoreboard JSON (and optional teams.json)."""
    with scoreboard_file.open("r", encoding="utf-8-sig") as f:
        sb_data = json.load(f)

    # Load teams metadata if available
    teams_meta: dict[str, dict[str, Any]] = {}
    if teams_file and teams_file.is_file():
        with teams_file.open("r", encoding="utf-8-sig") as f:
            tm_data = json.load(f)
            if isinstance(tm_data, list):
                for t in tm_data:
                    tid = str(t.get("id", ""))
                    teams_meta[tid] = t

    # Handle rows
    if isinstance(sb_data, dict) and "rows" in sb_data:
        rows = sb_data["rows"]
    elif isinstance(sb_data, list):
        rows = sb_data
    else:
        raise ValueError(f"Unrecognized DOMjudge scoreboard JSON structure in {scoreboard_file}")

    standings: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        team_id = str(row.get("team_id", f"team_{idx}"))
        team_info = teams_meta.get(team_id, {})

        team_name = (
            team_info.get("name")
            or team_info.get("display_name")
            or row.get("team_name")
            or f"Team {team_id}"
        )
        affiliation = team_info.get("affiliation", "")
        members_raw = team_info.get("members", "")
        members = split_members(members_raw)

        is_school = is_school_team(
            team_name, affiliation, members, school_keywords,
            registered_student_ids, contest_type
        )

        score_obj = row.get("score", {})
        solved = score_obj.get("num_solved", 0)
        total_time = score_obj.get("total_time", 0)

        # DOMjudge total_time is in minutes or seconds
        # If total_time > 100000, likely seconds, convert to minutes
        if isinstance(total_time, (int, float)) and total_time > 100000:
            penalty = int(total_time // 60)
        else:
            penalty = int(total_time)

        rank = row.get("rank")
        is_star = bool(team_info.get("is_star", False) or team_name.startswith("*"))
        external_elo = None if is_school else default_external_elo

        standings.append({
            "team_name": team_name,
            "is_school": is_school,
            "members": members if is_school else [],
            "external_elo": external_elo,
            "solved": int(solved),
            "penalty": penalty,
            "rank": int(rank) if rank is not None else None,
            "is_star": is_star,
        })

    return standings


def parse_pta(
    file_path: Path,
    school_keywords: list[str],
    registered_student_ids: set[str] | None,
    contest_type: str,
    default_external_elo: int = 1500,
) -> list[dict[str, Any]]:
    """Parse PTA (拼题A) export (JSON or CSV)."""
    # Try reading as JSON first
    is_json = False
    try:
        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
            is_json = True
    except Exception:
        is_json = False

    if is_json:
        rankings = data.get("commonRankings", [])
        if not rankings and isinstance(data, list):
            rankings = data

        standings: list[dict[str, Any]] = []
        for idx, item in enumerate(rankings, start=1):
            user_obj = item.get("user", {})
            student_user = user_obj.get("studentUser", {})

            student_num = str(student_user.get("studentNumber", "")).strip()
            name = (
                student_user.get("name")
                or user_obj.get("name")
                or user_obj.get("nickname")
                or f"Contestant_{idx}"
            )
            org = user_obj.get("organization", "")

            members = [student_num] if student_num else []
            is_school = is_school_team(
                name, org, members, school_keywords,
                registered_student_ids, contest_type
            )

            score = int(item.get("score", 0))
            time_consuming = int(item.get("timeConsuming", 0))
            ranking = item.get("ranking")

            standings.append({
                "team_name": name,
                "is_school": is_school,
                "members": members if is_school else [],
                "external_elo": None if is_school else default_external_elo,
                "solved": score,
                "penalty": time_consuming,
                "rank": int(ranking) if ranking is not None else None,
                "is_star": False,
            })
        return standings
    else:
        # Fallback to Generic CSV parsing for PTA CSV
        return parse_generic_csv(
            file_path, school_keywords, registered_student_ids,
            contest_type, default_external_elo
        )


def read_srk(file_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with file_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("SRK ranklist must be a JSON object.")

    meta: dict[str, str] = {}
    contest = data.get("contest")
    if isinstance(contest, dict):
        title = contest.get("title")
        if isinstance(title, dict):
            title = title.get("zh-CN") or title.get("fallback") or title.get("en")
        if isinstance(title, str):
            meta["title"] = title

        start_time = contest.get("startAt")
        if isinstance(start_time, str):
            meta["start_time"] = start_time

    return data, meta


def parse_srk(
    file_path: Path,
    school_keywords: list[str],
    registered_student_ids: set[str] | None,
    name_to_id: dict[str, str] | None,
    contest_type: str,
    default_external_elo: int = 1500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse algoUX standard ranklist (.srk.json) format."""
    data, meta = read_srk(file_path)
    rows = data.get("rows", [])
    standings: list[dict[str, Any]] = []
    seen_names: dict[str, int] = {}

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        user = row.get("user", {})
        raw_team_name = user.get("name") or f"Team_{idx}"
        if isinstance(raw_team_name, dict):
            raw_team_name = raw_team_name.get("zh-CN") or raw_team_name.get("fallback") or raw_team_name.get("en") or f"Team_{idx}"
        raw_team_name = str(raw_team_name).strip()

        organization = user.get("organization") or ""
        if isinstance(organization, dict):
            organization = organization.get("zh-CN") or organization.get("fallback") or organization.get("en") or ""
        organization = str(organization).strip()

        official = user.get("official", True)
        is_star = not official

        # Extract members, ignoring coaches
        raw_members = user.get("teamMembers", [])
        member_names: list[str] = []
        if isinstance(raw_members, list):
            for m in raw_members:
                if isinstance(m, dict):
                    role = str(m.get("role", "")).lower()
                    if role in ("coach", "trainer", "instructor", "教练"):
                        continue
                    m_val = m.get("name", "")
                    if isinstance(m_val, dict):
                        m_name = str(m_val.get("zh-CN") or m_val.get("fallback") or m_val.get("en") or "").strip()
                    else:
                        m_name = str(m_val).strip()
                    if m_name and m_name not in KNOWN_COACH_NAMES:
                        member_names.append(m_name)
                elif isinstance(m, str) and m.strip():
                    name_clean = m.strip()
                    if name_clean not in KNOWN_COACH_NAMES:
                        member_names.append(name_clean)

        # Determine is_school
        is_school = False
        if contest_type == "INTERNAL_SCHOOL":
            is_school = True
        else:
            text_check = f"{raw_team_name} {organization}".lower()
            if any(kw.lower() in text_check for kw in school_keywords):
                is_school = True
            elif not organization.strip() and name_to_id and any(m in name_to_id for m in member_names):
                is_school = True
            elif not organization.strip() and registered_student_ids and any(m in registered_student_ids for m in member_names):
                is_school = True

        # Resolved members
        resolved_members: list[str] = []
        if is_school:
            unresolved_members: list[str] = []
            for m_name in member_names:
                if registered_student_ids is not None and m_name in registered_student_ids:
                    resolved_members.append(m_name)
                elif name_to_id and m_name in name_to_id:
                    resolved_members.append(name_to_id[m_name])
                elif registered_student_ids is None and STUDENT_ID_PATTERN.fullmatch(m_name):
                    resolved_members.append(m_name)
                else:
                    unresolved_members.append(m_name)

            if unresolved_members:
                raise ValueError(
                    f"School team '{raw_team_name}' has members missing from the roster: "
                    f"{', '.join(unresolved_members)}."
                )
            if not member_names:
                raise ValueError(f"School team '{raw_team_name}' has no participant names or IDs.")

        # Disambiguate duplicate team names by adding organization if necessary
        team_name = raw_team_name
        if team_name in seen_names:
            seen_names[team_name] += 1
            if organization:
                team_name = f"{raw_team_name} ({organization} #{seen_names[raw_team_name]})"
            else:
                team_name = f"{raw_team_name} #{seen_names[raw_team_name]}"
        else:
            seen_names[team_name] = 1

        # Score & Penalty
        score_obj = row.get("score", {})
        solved = score_obj.get("value", 0)
        time_spec = score_obj.get("time", 0)

        if isinstance(time_spec, list) and len(time_spec) >= 1:
            val = time_spec[0]
            unit = time_spec[1] if len(time_spec) > 1 else "s"
            if unit == "s":
                penalty = int(val // 60)
            elif unit == "ms":
                penalty = int(val // 60000)
            else:
                penalty = int(val)
        elif isinstance(time_spec, (int, float)):
            if time_spec > 100000:
                penalty = int(time_spec // 60)
            else:
                penalty = int(time_spec)
        else:
            penalty = 0

        external_elo = None if is_school else default_external_elo

        standings.append({
            "team_name": team_name,
            "organization": organization,
            "is_school": is_school,
            "members": resolved_members if is_school else [],
            "external_elo": external_elo,
            "solved": int(solved),
            "penalty": int(penalty),
            "rank": idx,
            "is_star": bool(is_star),
        })

    return standings, meta


def detect_format(file_path: Path) -> str:
    """Auto-detect format of input file."""
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        try:
            with file_path.open("r", encoding="utf-8-sig") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    if "commonRankings" in content:
                        return "pta"
                    if ("type" in content and "problems" in content and "rows" in content) or (
                        "contest" in content and "rows" in content and any("user" in r for r in content.get("rows", [])[:5])
                    ):
                        return "srk"
                    if "rows" in content:
                        return "domjudge"
                if isinstance(content, list) and content and "score" in content[0]:
                    return "domjudge"
        except Exception:
            pass
        return "domjudge"
    return "csv"


def infer_start_time(contest_id: str) -> str:
    match = re.match(r"^(\d{4})_(\d{2})_(\d{2})", contest_id)
    if match:
        return f"{match[1]}-{match[2]}-{match[3]}T13:00:00+08:00"
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def import_contest(
    input_path: str | Path,
    output_path: str | Path,
    fmt: str = "auto",
    teams_path: str | Path | None = None,
    contest_id: str | None = None,
    title: str | None = None,
    contest_type: str = "INTERNAL_SCHOOL",
    start_time: str | None = None,
    school_keywords: list[str] | None = None,
    default_external_elo: int = 1500,
    users_path: str | Path = "data/users.csv",
    skip_validation: bool = False,
    type: str | None = None,
    external_elo_mode: str = "fixed",
    external_elo_file: str | Path | None = None,
) -> dict[str, Any]:
    """Execute contest import workflow and produce validated contest JSON."""
    if type is not None:
        contest_type = type

    in_file = Path(input_path)
    out_file = Path(output_path)
    t_file = Path(teams_path) if teams_path else None

    if not in_file.is_file():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    if contest_type not in VALID_CONTEST_TYPES:
        raise ValueError(f"Invalid contest_type: {contest_type}. Must be one of {VALID_CONTEST_TYPES}")

    # Load users for referential integrity & school matching
    registered_ids: set[str] | None = None
    name_to_id: dict[str, str] = {}
    u_path = Path(users_path)
    if u_path.is_file():
        users_dict, _ = validate_users_csv(u_path)
        registered_ids = set(users_dict.keys())
        for sid, u in users_dict.items():
            name_to_id[u["name"]] = sid

    keywords = school_keywords or DEFAULT_SCHOOL_KEYWORDS

    # Determine format
    if fmt == "auto":
        fmt = detect_format(in_file)

    meta: dict[str, Any] = {}

    # Parse standings
    if fmt == "csv":
        standings = parse_generic_csv(in_file, keywords, registered_ids, contest_type, default_external_elo)
    elif fmt == "domjudge":
        standings = parse_domjudge(in_file, t_file, keywords, registered_ids, contest_type, default_external_elo)
    elif fmt == "pta":
        standings = parse_pta(in_file, keywords, registered_ids, contest_type, default_external_elo)
    elif fmt in ("srk", "srk.json"):
        standings, meta = parse_srk(in_file, keywords, registered_ids, name_to_id, contest_type, default_external_elo)
        if not title and meta.get("title"):
            title = meta["title"]
        if not start_time and meta.get("start_time"):
            start_time = meta["start_time"]
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Re-sort standings by (solved desc, penalty asc) and assign nominal ranks
    standings.sort(key=lambda t: (-t["solved"], t["penalty"]))
    for rank_idx, t in enumerate(standings, start=1):
        t["rank"] = rank_idx

    # Apply external Elo anchoring strategy
    elo_map: dict[str, int] = {}
    target_ef = external_elo_file or ("data/external_elo_map.json" if Path("data/external_elo_map.json").is_file() else None)
    if target_ef:
        ef_path = Path(target_ef)
        if ef_path.is_file():
            try:
                with ef_path.open("r", encoding="utf-8-sig") as f:
                    raw_map = json.load(f)
                    if isinstance(raw_map, dict):
                        if "organizations" in raw_map and isinstance(raw_map["organizations"], dict):
                            # Structured external_elo_map.json format
                            for k, v in raw_map["organizations"].items():
                                elo_map[str(k).strip()] = int(round(v))
                        else:
                            # Flat {org_or_team: elo} format
                            for k, v in raw_map.items():
                                if isinstance(v, (int, float)):
                                    elo_map[str(k).strip()] = int(round(v))
            except Exception as e:
                print(f"Warning: Failed to load external elo map from {ef_path}: {e}", file=sys.stderr)

    n_teams = len(standings)
    for t in standings:
        if t["is_school"]:
            continue

        matched_elo = elo_map.get(t.get("organization", ""))
        if matched_elo is None:
            matched_elo = next(
                (score for key, score in elo_map.items() if key and key in t["team_name"]),
                None,
            )

        if matched_elo is not None:
            t["external_elo"] = matched_elo
        elif external_elo_mode == "rank" and n_teams > 1:
            rank_idx = t["rank"]
            pct = (rank_idx - 1) / max(1, n_teams - 1)
            # Standard ICPC tiered anchor: Gold: 2700-2300, Silver: 2300-1800, Bronze: 1800-1400, Iron: 1400-1100
            if pct <= 0.10:
                est = 2700.0 - (pct / 0.10) * 400.0
            elif pct <= 0.30:
                est = 2300.0 - ((pct - 0.10) / 0.20) * 500.0
            elif pct <= 0.60:
                est = 1800.0 - ((pct - 0.30) / 0.30) * 400.0
            else:
                est = 1400.0 - ((pct - 0.60) / 0.40) * 300.0
            t["external_elo"] = int(round(est))

    # Infer contest_id if not provided
    if not contest_id:
        contest_id = out_file.stem

    if not start_time:
        start_time = infer_start_time(contest_id)

    if not title:
        title = f"Contest {contest_id}"

    contest_dict: dict[str, Any] = {
        "contest_id": contest_id,
        "title": title,
        "contest_type": contest_type,
        "start_time": start_time,
        "standings": standings,
    }

    # Validate output schema
    if not skip_validation:
        errors = validate_contest_dict(
            contest=contest_dict,
            valid_student_ids=registered_ids,
            filename=out_file.name,
        )
        if errors:
            err_msg = "\n  - ".join(errors)
            raise ValueError(f"Generated contest data failed validation:\n  - {err_msg}")

    # Ensure parent directory exists and write JSON
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(contest_dict, f, indent=2, ensure_ascii=False)

    return contest_dict


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for import_contest.py."""
    parser = argparse.ArgumentParser(
        description="Multi-format scoreboard importer for USTS Rating System."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to raw input file (CSV, DOMjudge JSON, PTA)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to target JSON output file",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["auto", "csv", "domjudge", "pta", "srk"],
        default="auto",
        help="Input format (default: auto)",
    )
    parser.add_argument(
        "--teams", "-t",
        default=None,
        help="Optional teams metadata file (e.g. DOMjudge teams.json)",
    )
    parser.add_argument(
        "--contest-id",
        default=None,
        help="Contest ID (default: inferred from output filename)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Contest title",
    )
    parser.add_argument(
        "--type",
        choices=["INTERNAL_SCHOOL", "EXTERNAL_XCPC"],
        default="INTERNAL_SCHOOL",
        help="Contest type (default: INTERNAL_SCHOOL)",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="ISO-8601 start datetime (default: inferred from contest date)",
    )
    parser.add_argument(
        "--school-keywords",
        default="苏州科技大学,USTS,苏科大,苏州科大",
        help="Comma-separated keywords to identify school teams",
    )
    parser.add_argument(
        "--default-external-elo",
        type=int,
        default=1500,
        help="Default anchor Elo for external teams (default: 1500)",
    )
    parser.add_argument(
        "--users",
        default="data/users.csv",
        help="Path to users.csv file for referential verification (default: data/users.csv)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema validation before writing output",
    )
    parser.add_argument(
        "--external-elo-mode",
        choices=["fixed", "rank"],
        default="fixed",
        help="External team Elo anchoring mode: 'fixed' (uses default-external-elo) or 'rank' (dynamic percentile anchoring 1100-2700 based on standings rank). Defaults to fixed.",
    )
    parser.add_argument(
        "--external-elo-file",
        default=None,
        help="Optional JSON file mapping university/team names to external Elo ratings",
    )

    args = parser.parse_args(argv)

    keywords = [k.strip() for k in args.school_keywords.split(",") if k.strip()]

    try:
        result = import_contest(
            input_path=args.input,
            output_path=args.output,
            fmt=args.format,
            teams_path=args.teams,
            contest_id=args.contest_id,
            title=args.title,
            contest_type=args.type,
            start_time=args.start_time,
            school_keywords=keywords,
            default_external_elo=args.default_external_elo,
            users_path=args.users,
            skip_validation=args.skip_validation,
            external_elo_mode=args.external_elo_mode,
            external_elo_file=args.external_elo_file,
        )
        print(f"[SUCCESS] Successfully imported contest '{result['contest_id']}' with "
              f"{len(result['standings'])} teams -> {args.output}")
        return 0
    except Exception as e:
        print(f"[ERROR] Import failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
