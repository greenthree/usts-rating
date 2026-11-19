#!/usr/bin/env python3
"""
scripts/validate_data.py - Data Validation Tool for USTS Rating System.

Enforces 15 strict validation checks on data/users.csv and data/contests/*.json
using pure Python standard library.

CLI:
    python scripts/validate_data.py [--users PATH] [--contests PATH] [--strict]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

# Allowed enum values
VALID_CONTEST_TYPES = {"EXTERNAL_XCPC", "INTERNAL_SCHOOL"}

# Regex patterns
STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,32}$")
CONTEST_FILENAME_PATTERN = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_([a-z0-9_]+)\.json$")

# Required CSV headers in exact order, with an optional membership column.
EXPECTED_USER_HEADERS = [
    "student_id",
    "name",
    "grade",
    "initial_external_elo",
    "is_active",
]
USER_HEADERS_WITH_MEMBERSHIP = [*EXPECTED_USER_HEADERS, "is_team_member"]


class ValidationError(Exception):
    """Custom exception representing a validation error."""


def parse_bool(value: Any) -> bool | None:
    """Parse boolean from string or numeric representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (1, 1.0):
            return True
        if value in (0, 0.0):
            return False
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "t", "是", "y"):
            return True
        if v in ("false", "0", "no", "f", "否", "n"):
            return False
    return None


def validate_users_csv(users_path: str | Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate data/users.csv according to rules 1-8.

    Returns:
        (users_dict, list_of_errors)
    """
    path = Path(users_path)
    errors: list[str] = []
    users: dict[str, dict[str, Any]] = {}

    # Rule 1: FILE_EXISTENCE
    if not path.is_file():
        errors.append(f"[FILE_EXISTENCE] Roster file not found or not a regular file: {path}")
        return users, errors

    try:
        # Use utf-8-sig to automatically strip BOM if present
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append(f"[HEADER_MATCH] Roster file is completely empty: {path}")
                return users, errors

            # Rule 2: HEADER_MATCH
            clean_header = [h.strip() for h in header]
            if clean_header not in (EXPECTED_USER_HEADERS, USER_HEADERS_WITH_MEMBERSHIP):
                errors.append(
                    f"[HEADER_MATCH] Invalid CSV header: {clean_header}. "
                    f"Expected exact order: {EXPECTED_USER_HEADERS}, optionally followed by is_team_member."
                )
                return users, errors

            has_membership = clean_header == USER_HEADERS_WITH_MEMBERSHIP
            for row_idx, row in enumerate(reader, start=2):
                if not row or all(c.strip() == "" for c in row):
                    continue  # Ignore empty lines

                if len(row) != len(clean_header):
                    errors.append(
                        f"[CSV_FORMAT] Row {row_idx}: expected {len(clean_header)} "
                        f"columns, got {len(row)}: {row}"
                    )
                    continue

                student_id = row[0].strip()
                name = row[1].strip()
                grade_str = row[2].strip()
                elo_str = row[3].strip()
                is_active_str = row[4].strip()

                # Rule 3: STUDENT_ID_UNIQUE
                if student_id in users:
                    errors.append(
                        f"[STUDENT_ID_UNIQUE] Row {row_idx}: Duplicate student_id '{student_id}'. "
                        f"Previously seen on row {users[student_id]['row_idx']}."
                    )
                # Rule 4: STUDENT_ID_FORMAT
                elif not STUDENT_ID_PATTERN.match(student_id):
                    errors.append(
                        f"[STUDENT_ID_FORMAT] Row {row_idx}: Invalid student_id '{student_id}'. "
                        "Must match ^[A-Za-z0-9_-]{4,32}$."
                    )

                # Rule 5: NAME_VALID
                if not name or len(name) > 50:
                    errors.append(
                        f"[NAME_VALID] Row {row_idx}: Name '{name}' must be between 1 and 50 chars."
                    )
                elif any(c in name for c in ("\t", "\r", "\n")):
                    errors.append(
                        f"[NAME_VALID] Row {row_idx}: Name contains illegal control characters."
                    )

                # Rule 6: GRADE_VALID
                try:
                    grade = int(grade_str)
                    if not (2010 <= grade <= 2035):
                        errors.append(
                            f"[GRADE_VALID] Row {row_idx}: Grade {grade} out of range [2010, 2035]."
                        )
                except ValueError:
                    errors.append(
                        f"[GRADE_VALID] Row {row_idx}: Grade '{grade_str}' is not a valid integer."
                    )
                    grade = None

                # Rule 7: ELO_VALID
                initial_elo: int | None = None
                if elo_str != "":
                    try:
                        initial_elo = int(elo_str)
                        if not (100 <= initial_elo <= 3500):
                            errors.append(
                                f"[ELO_VALID] Row {row_idx}: initial_external_elo {initial_elo} "
                                "out of range [100, 3500]."
                            )
                    except ValueError:
                        errors.append(
                            f"[ELO_VALID] Row {row_idx}: initial_external_elo '{elo_str}' "
                            "is not a valid integer."
                        )

                # Rule 8: IS_ACTIVE_VALID
                is_active = parse_bool(is_active_str)
                if is_active is None:
                    errors.append(
                        f"[IS_ACTIVE_VALID] Row {row_idx}: is_active '{is_active_str}' "
                        "must be boolean ('true'/'false'/'1'/'0')."
                    )

                is_team_member = parse_bool(row[5]) if has_membership else True
                if is_team_member is None:
                    errors.append(
                        f"[IS_TEAM_MEMBER_VALID] Row {row_idx}: is_team_member '{row[5]}' "
                        "must be boolean ('true'/'false'/'1'/'0')."
                    )

                if student_id not in users:
                    users[student_id] = {
                        "row_idx": row_idx,
                        "student_id": student_id,
                        "name": name,
                        "grade": grade,
                        "initial_external_elo": initial_elo,
                        "is_active": is_active if is_active is not None else True,
                        "is_team_member": is_team_member,
                    }
    except Exception as e:
        errors.append(f"[CSV_READ_ERROR] Error reading {path}: {e}")

    return users, errors


def parse_contest_datetime(date_str: str) -> datetime | None:
    """Parse ISO-8601 or common datetime string into datetime object."""
    if not isinstance(date_str, str):
        return None
    s = date_str.strip()
    try:
        # Python 3.11 fromisoformat supports Z and timezone offsets
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        pass
    # Fallback formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def validate_contest_dict(
    contest: dict[str, Any],
    valid_student_ids: set[str] | None = None,
    filename: str | None = None,
) -> list[str]:
    """Validate contest data dictionary against schema rules 10-15."""
    errors: list[str] = []
    prefix = f"[{filename}] " if filename else ""

    # Rule 12: JSON_SYNTAX_SCHEMA top-level fields
    required_keys = {"contest_id", "title", "contest_type", "start_time", "standings"}
    missing = required_keys - set(contest.keys())
    if missing:
        errors.append(f"{prefix}[JSON_SYNTAX_SCHEMA] Missing required root keys: {sorted(missing)}")
        return errors

    contest_id = contest.get("contest_id")
    title = contest.get("title")
    contest_type = contest.get("contest_type")
    start_time_str = contest.get("start_time")
    standings = contest.get("standings")

    if not isinstance(contest_id, str) or not contest_id.strip():
        errors.append(f"{prefix}[JSON_SYNTAX_SCHEMA] contest_id must be a non-empty string.")

    if not isinstance(title, str) or not title.strip():
        errors.append(f"{prefix}[JSON_SYNTAX_SCHEMA] title must be a non-empty string.")

    # Rule 13: TYPE_ENUM
    if contest_type not in VALID_CONTEST_TYPES:
        errors.append(
            f"{prefix}[TYPE_ENUM] Invalid contest_type '{contest_type}'. "
            f"Allowed types: {sorted(VALID_CONTEST_TYPES)}."
        )

    # Date parsing and Rule 11: DATE_CONSISTENCY
    dt = parse_contest_datetime(str(start_time_str))
    if dt is None:
        errors.append(
            f"{prefix}[JSON_SYNTAX_SCHEMA] start_time '{start_time_str}' is not a valid ISO-8601 datetime."
        )
    elif filename:
        fn_match = CONTEST_FILENAME_PATTERN.match(filename)
        if fn_match:
            fn_year, fn_month, fn_day = int(fn_match.group(1)), int(fn_match.group(2)), int(fn_match.group(3))
            if (dt.year, dt.month, dt.day) != (fn_year, fn_month, fn_day):
                errors.append(
                    f"{prefix}[DATE_CONSISTENCY] Filename date ({fn_year:04d}_{fn_month:02d}_{fn_day:02d}) "
                    f"does not match start_time date ({dt.year:04d}_{dt.month:02d}_{dt.day:02d})."
                )

    # Validate standings list
    if not isinstance(standings, list) or len(standings) == 0:
        errors.append(f"{prefix}[JSON_SYNTAX_SCHEMA] standings must be a non-empty array of teams.")
        return errors

    seen_team_names: set[str] = set()
    students_in_contest: dict[str, str] = {}  # student_id -> team_name

    for idx, team in enumerate(standings, start=1):
        team_prefix = f"{prefix}Team {idx}"
        if not isinstance(team, dict):
            errors.append(f"{team_prefix} [JSON_SYNTAX_SCHEMA] Team entry must be an object.")
            continue

        team_name = team.get("team_name")
        is_school = team.get("is_school")
        members = team.get("members")
        external_elo = team.get("external_elo")
        solved = team.get("solved")
        penalty = team.get("penalty")
        rank = team.get("rank")
        is_star = team.get("is_star", False)

        # Team name check
        if not isinstance(team_name, str) or not team_name.strip():
            errors.append(f"{team_prefix} [JSON_SYNTAX_SCHEMA] team_name must be a non-empty string.")
            team_name = f"<team_{idx}>"
        else:
            team_name = team_name.strip()
            # Rule 15: Unique team name per contest
            if team_name in seen_team_names:
                errors.append(
                    f"{team_prefix} [CONTEST_INTEGRITY] Duplicate team_name '{team_name}' in standings."
                )
            seen_team_names.add(team_name)

        # is_school check
        if not isinstance(is_school, bool):
            errors.append(f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] is_school must be a boolean.")

        # solved check
        if not isinstance(solved, int) or isinstance(solved, bool) or solved < 0:
            errors.append(
                f"{team_prefix} ('{team_name}') [CONTEST_INTEGRITY] solved must be non-negative integer, got {solved}."
            )

        # penalty check
        if not isinstance(penalty, int) or isinstance(penalty, bool) or penalty < 0:
            errors.append(
                f"{team_prefix} ('{team_name}') [CONTEST_INTEGRITY] penalty must be non-negative integer, got {penalty}."
            )

        # rank check (optional)
        if rank is not None:
            if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                errors.append(
                    f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] rank must be positive integer if provided, got {rank}."
                )

        # is_star check (optional)
        if not isinstance(is_star, bool):
            errors.append(
                f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] is_star must be boolean, got {is_star}."
            )

        # external_elo check
        if external_elo is not None:
            if isinstance(external_elo, bool) or not isinstance(external_elo, (int, float)):
                errors.append(
                    f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] external_elo must be number or null, got {external_elo}."
                )
            elif not (100 <= external_elo <= 3500):
                errors.append(
                    f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] external_elo {external_elo} out of range [100, 3500]."
                )

        # members check
        if not isinstance(members, list):
            errors.append(f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] members must be a list of strings.")
            continue

        if is_school:
            if len(members) == 0:
                errors.append(
                    f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] is_school=True team must have at least 1 member."
                )

            seen_in_team: set[str] = set()
            for m in members:
                if not isinstance(m, str) or not m.strip():
                    errors.append(
                        f"{team_prefix} ('{team_name}') [JSON_SYNTAX_SCHEMA] Member ID must be non-empty string, got {m!r}."
                    )
                    continue
                m_str = m.strip()

                # Rule 15: No duplicate members in same team
                if m_str in seen_in_team:
                    errors.append(
                        f"{team_prefix} ('{team_name}') [CONTEST_INTEGRITY] Duplicate member '{m_str}' in same team."
                    )
                seen_in_team.add(m_str)

                # Rule 15: No student in multiple teams
                if m_str in students_in_contest:
                    errors.append(
                        f"{team_prefix} ('{team_name}') [CONTEST_INTEGRITY] Student '{m_str}' is already assigned "
                        f"to team '{students_in_contest[m_str]}'."
                    )
                else:
                    students_in_contest[m_str] = team_name

                # Rule 14: REFERENTIAL_INTEGRITY
                if valid_student_ids is not None and m_str not in valid_student_ids:
                    errors.append(
                        f"{team_prefix} ('{team_name}') [REFERENTIAL_INTEGRITY] Student ID '{m_str}' "
                        "not found in users.csv."
                    )

    return errors


def validate_contest_file(
    contest_path: str | Path,
    valid_student_ids: set[str] | None = None,
) -> list[str]:
    """Validate a single contest JSON file."""
    path = Path(contest_path)
    errors: list[str] = []

    if not path.is_file():
        return [f"[FILE_EXISTENCE] Contest file not found: {path}"]

    # Rule 10: FILENAME_FORMAT
    filename = path.name
    if not CONTEST_FILENAME_PATTERN.match(filename):
        errors.append(
            f"[{filename}] [FILENAME_FORMAT] Filename '{filename}' does not match "
            r"^(\d{4})_(\d{2})_(\d{2})_([a-z0-9_]+)\.json$."
        )

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            contest_data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"[{filename}] [JSON_SYNTAX_SCHEMA] Malformed JSON: {e}")
        return errors
    except Exception as e:
        errors.append(f"[{filename}] [JSON_SYNTAX_SCHEMA] Failed to read JSON file: {e}")
        return errors

    if not isinstance(contest_data, dict):
        errors.append(f"[{filename}] [JSON_SYNTAX_SCHEMA] Root element must be a JSON object.")
        return errors

    file_errors = validate_contest_dict(
        contest=contest_data,
        valid_student_ids=valid_student_ids,
        filename=filename,
    )
    errors.extend(file_errors)
    return errors


def validate_all(
    users_path: str | Path = "data/users.csv",
    contests_path: str | Path = "data/contests",
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Run full validation suite on roster and contest files.

    Returns:
        (is_valid, list_of_errors)
    """
    all_errors: list[str] = []

    # 1. Validate Users CSV
    users_dict, user_errors = validate_users_csv(users_path)
    all_errors.extend(user_errors)
    valid_student_ids: set[str] = set(users_dict.keys())

    # 2. Validate Contests
    c_path = Path(contests_path)
    if c_path.is_file():
        contest_files = [c_path]
    elif c_path.is_dir():
        contest_files = sorted(c_path.glob("*.json"))
        # Rule 9: DIR_NOT_EMPTY
        if not contest_files:
            all_errors.append(f"[DIR_NOT_EMPTY] Contests directory '{contests_path}' contains no .json files.")
    else:
        all_errors.append(f"[FILE_EXISTENCE] Contests path not found: {contests_path}")
        contest_files = []

    for c_file in contest_files:
        c_errors = validate_contest_file(c_file, valid_student_ids=valid_student_ids)
        all_errors.extend(c_errors)

    is_valid = len(all_errors) == 0
    return is_valid, all_errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Strict validator for USTS Rating System data files."
    )
    parser.add_argument(
        "--users",
        default="data/users.csv",
        help="Path to users.csv file (default: data/users.csv)",
    )
    parser.add_argument(
        "--contests",
        default="data/contests",
        help="Path to contests directory or single JSON file (default: data/contests)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict mode (exit non-zero on any issue)",
    )

    args = parser.parse_args(argv)

    print("==================================================")
    print("  USTS Rating System - Data Integrity Validator")
    print("==================================================")
    print(f"Users file:    {args.users}")
    print(f"Contests path: {args.contests}")
    print("--------------------------------------------------")

    is_valid, errors = validate_all(
        users_path=args.users,
        contests_path=args.contests,
        strict=args.strict,
    )

    if is_valid:
        print("[SUCCESS] All data validation checks passed successfully!")
        return 0
    else:
        print(f"[FAILED] Found {len(errors)} validation error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
