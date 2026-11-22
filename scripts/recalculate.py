"""
USTS Rating System - Elo Calculation Engine
NumPy-accelerated multi-player Elo calculation engine generating rating_db.json.
Pure, unit-testable mathematical functions and chronological contest processing pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .validate_data import validate_users_csv
else:
    from validate_data import validate_users_csv


# ==============================================================================
# 1. PURE MATHEMATICAL FUNCTIONS
# ==============================================================================

def compute_team_rating(
    member_ratings: list[float],
    rated_flags: list[bool] | None = None,
    is_school: bool = True,
    external_elo: float | None = None,
    initial_rating: float = 800.0,
) -> float:
    """
    Compute team rating R_team using XCPC-Elo's log-power-mean aggregation:
    R_team = 400 * log10( (1/k) * sum_{i=1}^k 10^(R_i / 400) )
    - For external teams: returns external_elo (default 1500.0).
    - For school teams:
      * Unrated members (history == 0) are excluded so seeds are driven
        by members who have competed.
      * If no member has prior contest history, returns initial_rating (default 800.0).
      * If k == 1, returns the single member's rating.
      * If k >= 2, computes numerically stable log-power-mean.
    """
    if not is_school:
        return float(external_elo) if external_elo is not None else 1500.0

    if rated_flags is not None:
        rated_ratings = [float(r) for r, f in zip(member_ratings, rated_flags) if f]
    else:
        rated_ratings = [float(r) for r in member_ratings]

    if not rated_ratings:
        return initial_rating

    k = len(rated_ratings)
    if k == 1:
        return float(rated_ratings[0])

    r_arr = np.asarray(rated_ratings, dtype=np.float64)
    r_max = np.max(r_arr)
    p = np.exp(np.log(10.0) / 400.0 * (r_arr - r_max))
    return float(r_max + 400.0 * np.log10(np.mean(p)))


def get_display_offset(rated_count: int) -> int:
    """
    Get display rating offset based on accumulated rated contests.
    Newbie protection lasts 3 contests; 3rd contest finishes to show real rating:
    - rated_count <= 0: -1000
    - rated_count == 1: -500
    - rated_count == 2: -250
    - rated_count >= 3: 0 (第三场结束展示真实积分)
    """
    if rated_count <= 0:
        return -1000
    elif rated_count == 1:
        return -500
    elif rated_count == 2:
        return -250
    else:
        return 0


def to_display_rating(true_rating: float, rated_count: int) -> int:
    """
    Convert underlying true rating to display rating:
    display_rating = max(0, round(true_rating + offset))
    """
    offset = get_display_offset(rated_count)
    return max(0, int(round(true_rating + offset)))


def get_tier_and_color(display_rating: int) -> tuple[str, str, str]:
    """
    Determine 7-tier classification, Chinese name, and hex color token:
    - 0 ~ 799:    Newbie / 萌新 (#808080)
    - 800 ~ 999:  Pupil / 进阶 (#008000)
    - 1000 ~ 1199: Specialist / 骨干 (#03A89E)
    - 1200 ~ 1399: Expert / 精英 (#0000FF)
    - 1400 ~ 1599: Candidate Master / 候选大师 (#AA00AA)
    - 1600 ~ 1799: Master / 校队大师 (#FF8C00)
    - >= 1800:     Grandmaster / 传奇宗师 (#FF0000)
    """
    if display_rating < 800:
        return "Newbie", "萌新", "#808080"
    elif display_rating < 1000:
        return "Pupil", "进阶", "#008000"
    elif display_rating < 1200:
        return "Specialist", "骨干", "#03A89E"
    elif display_rating < 1400:
        return "Expert", "精英", "#0000FF"
    elif display_rating < 1600:
        return "Candidate Master", "候选大师", "#AA00AA"
    elif display_rating < 1800:
        return "Master", "校队大师", "#FF8C00"
    else:
        return "Grandmaster", "传奇宗师", "#FF0000"


def compute_ranks(standings: list[dict]) -> np.ndarray:
    """
    Compute 1-based ranks from standings list.
    Teams are sorted primary by solved descending, secondary by penalty ascending.
    Tied teams receive the arithmetic mean of their tied 1-based rank positions.
    Output array preserves the original input order of standings.
    """
    n = len(standings)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    # indexed by original position: (orig_idx, team)
    indexed = sorted(
        enumerate(standings),
        key=lambda item: (-item[1].get("solved", 0), item[1].get("penalty", 0))
    )

    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while (
            j < n
            and indexed[j][1].get("solved", 0) == indexed[i][1].get("solved", 0)
            and indexed[j][1].get("penalty", 0) == indexed[i][1].get("penalty", 0)
        ):
            j += 1
        # 1-based positions from (i + 1) to j
        mean_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            orig_idx = indexed[k][0]
            ranks[orig_idx] = mean_rank
        i = j

    return ranks


def compute_seeds(ratings: np.ndarray) -> np.ndarray:
    """
    Compute expected ranks (Seeds) for each team in ratings:
    P(B > A) = 1 / (1 + 10^((R_A - R_B) / 400))
    Seed_A = 1 + sum_{B != A} P(B > A)
    Optimized with exponential decomposition: 10^((R_A - R_B)/400) = A_i * B_j.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n = len(ratings)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    FACTOR = 0.005756462732485114  # np.log(10.0) / 400.0
    B = np.exp(-ratings * FACTOR)
    A = np.exp(ratings * FACTOR)
    M = A[:, None] * B
    M += 1.0
    np.reciprocal(M, out=M)
    # On diagonal, A_i * B_i == 1.0 -> P_{i, i} == 0.5
    return 1.0 + np.sum(M, axis=1) - 0.5


def compute_perf_ratings(
    ratings: np.ndarray,
    target_ranks: np.ndarray,
    iterations: int = 50,
) -> np.ndarray:
    """
    Vectorized bisection in [-500.0, 3500.0] over 50 iterations to solve for
    performance rating R_perf such that expected rank equals target_rank.
    Optimized with precomputed exponential factor, unique-pair deduplication,
    and zero-allocation in-place buffer.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    target_ranks = np.asarray(target_ranks, dtype=np.float64)
    n = len(ratings)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return ratings.copy()

    FACTOR = 0.005756462732485114  # np.log(10.0) / 400.0
    B = np.exp(-ratings * FACTOR)  # Shape (n,)

    if n > 400:
        # Deduplicate identical (target_rank, rating) pairs (common in large preliminary contests with tied ranks)
        pairs = np.column_stack([target_ranks, ratings])
        u_pairs, inverse_idx = np.unique(pairs, axis=0, return_inverse=True)
        u_n = len(u_pairs)
        low = np.full(u_n, -500.0, dtype=np.float64)
        high = np.full(u_n, 3500.0, dtype=np.float64)
        u_targets = u_pairs[:, 0]
        u_B = np.exp(-u_pairs[:, 1] * FACTOR)
        M = np.empty((u_n, n), dtype=np.float64)

        iter_steps = 35 if iterations >= 35 else iterations
        for _ in range(iter_steps):
            mid = (low + high) * 0.5
            A = np.exp(mid * FACTOR)
            np.multiply(A[:, None], B, out=M)
            M += 1.0
            np.reciprocal(M, out=M)
            diag = 1.0 / (1.0 + A * u_B)
            exp_rank = 1.0 + np.sum(M, axis=1) - diag

            mask = exp_rank > u_targets
            low[mask] = mid[mask]
            high[~mask] = mid[~mask]

        u_perf = (low + high) * 0.5
        return u_perf[inverse_idx]

    # Standard path for smaller contests (n <= 400)
    low = np.full(n, -500.0, dtype=np.float64)
    high = np.full(n, 3500.0, dtype=np.float64)
    M = np.empty((n, n), dtype=np.float64)

    for _ in range(iterations):
        mid = (low + high) * 0.5
        A = np.exp(mid * FACTOR)
        np.multiply(A[:, None], B, out=M)
        M += 1.0
        np.reciprocal(M, out=M)
        diag = 1.0 / (1.0 + A * B)
        exp_rank = 1.0 + np.sum(M, axis=1) - diag

        mask = exp_rank > target_ranks
        low[mask] = mid[mask]
        high[~mask] = mid[~mask]

    return (low + high) * 0.5


def calibrate_internal_school_deltas(deltas: list[float]) -> list[float]:
    """
    For INTERNAL_SCHOOL contests, calibrate deltas by subtracting mean drift:
    mu = sum(deltas) / M
    calibrated_delta = delta - mu
    guaranteeing sum(calibrated_deltas) == 0.
    """
    if not deltas:
        return []
    mu = sum(deltas) / len(deltas)
    return [d - mu for d in deltas]


def extract_member_ids(members: list) -> list[str]:
    """
    Extract clean student ID strings from members list.
    Supports both list of strings ["2023001"] and list of dicts [{"student_id": "2023001"}].
    """
    ids: list[str] = []
    for m in members:
        if isinstance(m, str):
            val = m.strip()
            if val:
                ids.append(val)
        elif isinstance(m, dict):
            sid = m.get("student_id") or m.get("id") or ""
            val = str(sid).strip()
            if val:
                ids.append(val)
    return ids


def parse_iso_datetime(dt_str: str) -> datetime:
    """Parse ISO-8601 string to timezone-aware UTC datetime."""
    dt = datetime.fromisoformat(dt_str.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_user_retired(
    last_contest_time: datetime | None,
    grade: int,
    is_active: bool,
    reference_time: datetime,
) -> bool:
    """
    Determine retirement status according to specifications:
    - is_active == False: unconditionally retired.
    - grade <= 2022: 2022 grade and earlier (graduated) are unconditionally retired.
    - If student has contest history: retired if (reference_time - last_contest_time) > 730 days.
    - If 0 contests: retired if (reference_time - September 1 of grade year) > 730 days.
    """
    if not is_active:
        return True
    if grade <= 2022:
        return True
    cutoff = timedelta(days=730)
    if last_contest_time is not None:
        return (reference_time - last_contest_time) > cutoff
    enrollment_time = datetime(grade, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
    return (reference_time - enrollment_time) > cutoff


# ==============================================================================
# 2. DATA LOADING & ROSTER MANAGEMENT
# ==============================================================================

def load_users(users_csv_path: Path | str) -> dict[str, dict[str, Any]]:
    """
    Load users from users.csv (supporting UTF-8 and UTF-8-SIG).
    Initializes true_rating from initial_external_elo or the default of 800.
    """
    path = Path(users_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Users file not found: {path}")
    roster, errors = validate_users_csv(path)
    if errors:
        raise ValueError(f"Invalid roster {path}:\n" + "\n".join(errors))

    info_path = path.parent / "student_info.json"
    student_info: dict[str, Any] = {}
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                student_info = json.load(f)
        except Exception:
            student_info = {}

    users: dict[str, dict[str, Any]] = {}
    for sid, row in roster.items():
        name = row["name"]
        grade = row["grade"]
        initial_elo = row["initial_external_elo"]
        init_true = float(initial_elo) if initial_elo is not None else 800.0
        is_active = row["is_active"]
        u_meta = student_info.get(name, {})
        initial_display = to_display_rating(init_true, 0)
        users[sid] = {
            "student_id": sid,
            "name": name,
            "grade": grade,
            "class_name": u_meta.get("class_name", ""),
            "pinyin": u_meta.get("pinyin", ""),
            "major": u_meta.get("major", ""),
            "is_active": is_active,
            "is_team_member": row["is_team_member"],
            "is_retired": (not is_active) or (grade <= 2022),
            "true_rating": init_true,
            "display_rating": initial_display,
            "max_display_rating": initial_display,
            "rated_count": 0,
            "tier": "Newbie",
            "color": "#808080",
            "last_contest_date": None,
            "history": [],
            "_last_contest_dt": None,
        }
    return users


def load_contests(contests_dir: Path | str) -> list[dict[str, Any]]:
    """
    Load all contest JSON files from contests_dir and sort chronologically by start_time.
    """
    dir_path = Path(contests_dir)
    if not dir_path.exists():
        return []

    contests: list[tuple[datetime, dict[str, Any]]] = []
    for json_file in sorted(dir_path.glob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            start = parse_iso_datetime(data["start_time"])
            data["_file_path"] = str(json_file)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Failed to load contest {json_file}: {exc}") from exc
        contests.append((start, data))

    contests.sort(key=lambda item: item[0])
    return [data for _, data in contests]


# ==============================================================================
# 3. CONTEST PROCESSING PIPELINE
# ==============================================================================

def process_contest(
    contest: dict[str, Any],
    user_db: dict[str, dict[str, Any]],
    external_db: dict[str, dict[str, Any]] | None = None,
    external_elo_map: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """
    Process a single contest chronologically and update user_db in-place.
    Optionally tracks persistent external teams across contests in external_db.
    Returns the processed contest record suitable for JSON serialization.
    """
    contest_id = contest["contest_id"]
    contest_title = contest.get("title", contest_id)
    contest_type = contest.get("contest_type", "INTERNAL_SCHOOL")
    start_time_str = contest["start_time"]
    start_dt = parse_iso_datetime(start_time_str)

    # Contest weight: 0.80 for INTERNAL_SCHOOL, 1.00 for EXTERNAL_XCPC
    w_contest = 0.80 if contest_type == "INTERNAL_SCHOOL" else 1.00

    raw_standings = contest.get("standings", [])
    if not raw_standings:
        return {
            "contest_id": contest_id,
            "title": contest_title,
            "contest_type": contest_type,
            "start_time": start_time_str,
            "standings": [],
        }

    # Sort standings by (-solved, penalty)
    standings = sorted(
        raw_standings,
        key=lambda t: (-t.get("solved", 0), t.get("penalty", 0))
    )
    n = len(standings)

    # 1. Compute 1-based ranks with tie-breaking arithmetic mean
    ranks = compute_ranks(standings)

    # 2. Synthesize team ratings
    team_ratings = np.zeros(n, dtype=np.float64)
    team_members_list: list[list[str]] = []

    for i, team in enumerate(standings):
        is_school = team.get("is_school", True)
        if not is_school:
            t_key = team.get("team_id") or team.get("team_name") or f"ext_team_{i}"
            if external_db is not None and t_key in external_db:
                # Evolved persistent rating from previous contests
                r_team = float(external_db[t_key]["rating"])
            elif team.get("external_elo") is not None:
                r_team = float(team["external_elo"])
            elif external_elo_map:
                # Lookup baseline by exact team key or organization substring (cached for fast replay)
                if t_key in external_elo_map:
                    r_team = float(external_elo_map[t_key])
                else:
                    if "_elo_cache" not in external_elo_map:
                        external_elo_map["_elo_cache"] = {}
                    cache = external_elo_map["_elo_cache"]
                    if t_key in cache:
                        r_team = cache[t_key]
                    else:
                        org_part = t_key.split(" - ")[0].strip() if " - " in t_key else ""
                        if org_part and org_part in external_elo_map:
                            matched_elo = float(external_elo_map[org_part])
                        else:
                            matched_elo = None
                            for k, v in external_elo_map.items():
                                if k != "_elo_cache" and k in t_key:
                                    matched_elo = float(v)
                                    break
                        r_team = matched_elo if matched_elo is not None else 1500.0
                        cache[t_key] = r_team
            else:
                r_team = compute_team_rating([], is_school=False, external_elo=team.get("external_elo"))
            team_ratings[i] = r_team
            team_members_list.append([])
        else:
            m_ids = extract_member_ids(team.get("members", []))
            team_members_list.append(m_ids)
            member_ratings: list[float] = []
            rated_flags: list[bool] = []
            for sid in m_ids:
                if sid in user_db:
                    member_ratings.append(user_db[sid]["true_rating"])
                    rated_flags.append(len(user_db[sid]["history"]) > 0)
                else:
                    member_ratings.append(800.0)
                    rated_flags.append(False)
            r_team = compute_team_rating(member_ratings, rated_flags=rated_flags, is_school=True)
            team_ratings[i] = r_team

    # 3. Elo expectations and performance bisection
    seeds = compute_seeds(team_ratings)
    target_ranks = np.sqrt(seeds * ranks)
    needed_ratings = compute_perf_ratings(team_ratings, target_ranks, iterations=50)
    perf_ratings = compute_perf_ratings(team_ratings, ranks, iterations=50)

    # 4. Member delta allocation (fully aligned with XCPC-Elo)
    # lr = 0.6 + 0.3 * exp(-history_len / 1.0)
    # raw_delta = w_contest * (needed_ratings[i] - old_true_rating) * lr
    school_member_instances: list[dict[str, Any]] = []
    for i, team in enumerate(standings):
        if team.get("is_school", True):
            for sid in team_members_list[i]:
                old_r = user_db[sid]["true_rating"] if sid in user_db else 800.0
                hist_len = len(user_db[sid]["history"]) if sid in user_db else 0
                lr = 0.6 + 0.3 * np.exp(-hist_len / 1.0)
                raw_delta = w_contest * (needed_ratings[i] - old_r) * lr
                school_member_instances.append({
                    "team_idx": i,
                    "student_id": sid,
                    "raw_delta": float(raw_delta),
                })

    if contest_type == "INTERNAL_SCHOOL" and school_member_instances:
        raw_deltas = [inst["raw_delta"] for inst in school_member_instances]
        calibrated_deltas = calibrate_internal_school_deltas(raw_deltas)
        for inst, c_delta in zip(school_member_instances, calibrated_deltas):
            inst["final_delta"] = c_delta
    else:
        for inst in school_member_instances:
            inst["final_delta"] = inst["raw_delta"]

    # 5. Map (team_idx, sid) -> final_delta
    member_final_delta_map: dict[tuple[int, str], float] = {
        (inst["team_idx"], inst["student_id"]): inst["final_delta"]
        for inst in school_member_instances
    }

    # Compute team deltas (mean member delta for school teams, or (needed - rating)*0.6 for external)
    team_deltas = np.zeros(n, dtype=np.float64)
    for i, team in enumerate(standings):
        if team.get("is_school", True):
            m_sids = team_members_list[i]
            if m_sids:
                team_deltas[i] = float(np.mean([member_final_delta_map.get((i, sid), 0.0) for sid in m_sids]))
            else:
                team_deltas[i] = 0.0
        else:
            team_deltas[i] = float(w_contest * (needed_ratings[i] - team_ratings[i]) * 0.6)

    for i, team in enumerate(standings):
        if team.get("is_school", True):
            for sid in team_members_list[i]:
                if sid not in user_db:
                    user_db[sid] = {
                        "student_id": sid,
                        "name": sid,
                        "grade": 2026,
                        "is_active": True,
                        "is_team_member": False,
                        "is_retired": False,
                        "true_rating": 800.0,
                        "display_rating": to_display_rating(800.0, 0),
                        "max_display_rating": to_display_rating(800.0, 0),
                        "rated_count": 0,
                        "tier": "Newbie",
                        "color": "#808080",
                        "last_contest_date": None,
                        "history": [],
                        "_last_contest_dt": None,
                    }

                user = user_db[sid]
                f_delta = member_final_delta_map.get((i, sid), team_deltas[i])
                new_true = user["true_rating"] + f_delta
                user["true_rating"] = new_true
                user["rated_count"] += 1
                new_display = to_display_rating(new_true, user["rated_count"])
                user["display_rating"] = new_display
                user["max_display_rating"] = max(user["max_display_rating"], new_display)
                user["last_contest_date"] = start_time_str
                user["_last_contest_dt"] = start_dt

                tier_name, _, hex_color = get_tier_and_color(new_display)
                user["tier"] = tier_name
                user["color"] = hex_color

                # Record individual history entry
                history_entry = {
                    "contest_id": contest_id,
                    "contest_title": contest_title,
                    "contest_type": contest_type,
                    "contest_date": start_time_str,
                    "team_name": team.get("team_name", "Team"),
                    "rank": float(ranks[i]),
                    "solved": int(team.get("solved", 0)),
                    "penalty": int(team.get("penalty", 0)),
                    "perf_rating": round(float(perf_ratings[i]), 1),
                    "delta": round(float(f_delta), 1),
                    "true_rating_after": round(float(new_true), 1),
                    "display_rating_after": int(new_display),
                    "rated_count_after": int(user["rated_count"]),
                }
                user["history"].append(history_entry)

    # 6.5 Update external teams in external_db
    if external_db is not None:
        for i, team in enumerate(standings):
            if not team.get("is_school", True):
                t_key = team.get("team_id") or team.get("team_name") or f"ext_team_{i}"
                cur_r = float(team_ratings[i])
                d = float(team_deltas[i])
                new_r = cur_r + d
                if t_key not in external_db:
                    external_db[t_key] = {
                        "team_name": team.get("team_name", t_key),
                        "rating": new_r,
                        "max_rating": new_r,
                        "rated_count": 1,
                        "last_contest": contest_id,
                    }
                else:
                    external_db[t_key]["rating"] = new_r
                    external_db[t_key]["max_rating"] = max(external_db[t_key]["max_rating"], new_r)
                    external_db[t_key]["rated_count"] += 1
                    external_db[t_key]["last_contest"] = contest_id

    # 7. Format standings for output contest record
    processed_standings: list[dict[str, Any]] = []
    for i, team in enumerate(standings):
        t_record = dict(team)
        t_record["rank"] = float(ranks[i])
        t_record["team_rating"] = round(float(team_ratings[i]), 1)
        t_record["seed"] = round(float(seeds[i]), 2)
        t_record["perf_rating"] = round(float(perf_ratings[i]), 1)
        t_record["delta"] = round(float(team_deltas[i]), 1)
        if not team.get("is_school", True):
            t_record["rating_after"] = round(float(team_ratings[i] + team_deltas[i]), 1)
        processed_standings.append(t_record)

    return {
        "contest_id": contest_id,
        "title": contest_title,
        "contest_type": contest_type,
        "start_time": start_time_str,
        "team_count": len(standings),
        "standings": processed_standings,
    }


# ==============================================================================
# 4. FULL RECALCULATION & OUTPUT GENERATION
# ==============================================================================

TIERS_CONFIG = [
    {"name": "Newbie", "cn_name": "萌新", "min_rating": 0, "max_rating": 799, "color": "#808080"},
    {"name": "Pupil", "cn_name": "进阶", "min_rating": 800, "max_rating": 999, "color": "#008000"},
    {"name": "Specialist", "cn_name": "骨干", "min_rating": 1000, "max_rating": 1199, "color": "#03A89E"},
    {"name": "Expert", "cn_name": "精英", "min_rating": 1200, "max_rating": 1399, "color": "#0000FF"},
    {"name": "Candidate Master", "cn_name": "候选大师", "min_rating": 1400, "max_rating": 1599, "color": "#AA00AA"},
    {"name": "Master", "cn_name": "校队大师", "min_rating": 1600, "max_rating": 1799, "color": "#FF8C00"},
    {"name": "Grandmaster", "cn_name": "传奇宗师", "min_rating": 1800, "max_rating": 9999, "color": "#FF0000"},
]


def recalculate_all(
    data_dir: Path | str,
    output_paths: list[Path | str] | None = None,
) -> dict[str, Any]:
    """
    Run full chronological recalculation across users and contests.
    Evaluates 2-year retirement status.
    Generates rating database dictionary and writes to output_paths.
    """
    data_dir = Path(data_dir)
    users_csv = data_dir / "users.csv"
    contests_dir = data_dir / "contests"

    user_db = load_users(users_csv)
    contests = load_contests(contests_dir)

    # Load baseline external Elo map if available (generated from sync_xcpc_elo.py or static)
    external_elo_map: dict[str, float] = {}
    candidate_maps = [
        data_dir / "external_elo_map.json",
        Path("data/external_elo_map.json"),
    ]
    for m_path in candidate_maps:
        if m_path.is_file():
            try:
                with open(m_path, "r", encoding="utf-8-sig") as f:
                    raw_m = json.load(f)
                    if isinstance(raw_m, dict):
                        if "organizations" in raw_m and isinstance(raw_m["organizations"], dict):
                            for k, v in raw_m["organizations"].items():
                                external_elo_map[str(k).strip()] = float(v)
                        else:
                            for k, v in raw_m.items():
                                if isinstance(v, (int, float)):
                                    external_elo_map[str(k).strip()] = float(v)
                break
            except Exception as e:
                print(f"Warning: Failed to load external elo map from {m_path}: {e}", file=sys.stderr)

    external_db: dict[str, dict[str, Any]] = {}
    processed_contests: list[dict[str, Any]] = []
    latest_contest_dt: datetime | None = None

    for c in contests:
        p_c = process_contest(c, user_db, external_db=external_db, external_elo_map=external_elo_map)
        processed_contests.append(p_c)
        c_dt = parse_iso_datetime(c["start_time"])
        if latest_contest_dt is None or c_dt > latest_contest_dt:
            latest_contest_dt = c_dt

    # Reference time for retirement evaluation
    now_utc = datetime.now(timezone.utc)
    ref_time = max(latest_contest_dt, now_utc) if latest_contest_dt else now_utc

    # Use IDs scoped to this database for public roster and standings links.
    participant_ids = {sid: number for number, sid in enumerate(sorted(user_db), start=1)}
    for contest in processed_contests:
        for standing in contest["standings"]:
            members = standing.get("members", []) if standing.get("is_school", True) else []
            standing["members"] = [
                participant_ids[sid] for sid in extract_member_ids(members)
            ]

    # Finalize users: retirement status, tier, clean private fields, round ratings
    final_users: list[dict[str, Any]] = []
    for user in user_db.values():
        user["is_retired"] = is_user_retired(
            user["_last_contest_dt"],
            user["grade"],
            user["is_active"],
            ref_time,
        )
        tier_name, _, hex_color = get_tier_and_color(user["display_rating"])
        user["tier"] = tier_name
        user["color"] = hex_color

        sid = user["student_id"]
        participant_id = participant_ids[sid]
        clean_user = {
            "participant_id": participant_id,
            "name": user["name"] if user["name"] != sid else f"队员 {participant_id}",
            "grade": user["grade"],
            "class_name": user.get("class_name", ""),
            "pinyin": user.get("pinyin", ""),
            "major": user.get("major", ""),
            "is_active": user["is_active"],
            "is_team_member": user["is_team_member"],
            "is_retired": user["is_retired"],
            "true_rating": round(float(user["true_rating"]), 1),
            "display_rating": int(user["display_rating"]),
            "max_display_rating": int(user["max_display_rating"]),
            "rated_count": int(user["rated_count"]),
            "tier": user["tier"],
            "color": user["color"],
            "last_contest_date": user["last_contest_date"],
            "history": user["history"],
        }
        final_users.append(clean_user)

    final_users.sort(key=lambda u: (-u["display_rating"], -u["true_rating"], u["participant_id"]))

    total_users = len(final_users)
    active_users = sum(1 for u in final_users if not u["is_retired"])
    retired_users = sum(1 for u in final_users if u["is_retired"])

    rating_db = {
        "metadata": {
            "generated_at": now_utc.isoformat(),
            "total_users": total_users,
            "active_users": active_users,
            "retired_users": retired_users,
            "total_contests": len(processed_contests),
            "external_teams_tracked": len(external_db),
        },
        "tiers": TIERS_CONFIG,
        "users": final_users,
        "contests": processed_contests,
    }

    summary_contests: list[dict[str, Any]] = []
    for contest in processed_contests:
        summary_contests.append({
            "contest_id": contest["contest_id"],
            "title": contest["title"],
            "contest_type": contest["contest_type"],
            "start_time": contest["start_time"],
            "team_count": len(contest.get("standings", [])),
            "standings": [],
        })

    summary_db = {
        "metadata": {
            "generated_at": now_utc.isoformat(),
            "total_users": total_users,
            "active_users": active_users,
            "retired_users": retired_users,
            "total_contests": len(processed_contests),
            "external_teams_tracked": len(external_db),
        },
        "tiers": TIERS_CONFIG,
        "users": final_users,
        "contests": summary_contests,
    }

    # Default output paths if none provided
    if output_paths is None or len(output_paths) == 0:
        output_paths = [
            Path("web/data/rating_db.json"),
            Path("dist/data/rating_db.json"),
        ]

    for out_p in output_paths:
        target = Path(out_p)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(rating_db, f, ensure_ascii=False, separators=(',', ':'))
        print(f"[RECALCULATE] Successfully exported rating database to: {target}")

        summary_target = target.parent / "summary.json"
        with open(summary_target, "w", encoding="utf-8") as f:
            json.dump(summary_db, f, ensure_ascii=False, separators=(',', ':'))
        print(f"[RECALCULATE] Successfully exported summary database to: {summary_target}")

        contests_dir_out = target.parent / "contests"
        if contests_dir_out.resolve() != contests_dir.resolve():
            contests_dir_out.mkdir(parents=True, exist_ok=True)
            for p_c in processed_contests:
                cid = p_c["contest_id"]
                c_target = contests_dir_out / f"{cid}.json"
                with open(c_target, "w", encoding="utf-8") as f:
                    json.dump(p_c, f, ensure_ascii=False, separators=(',', ':'))
            print(f"[RECALCULATE] Successfully exported {len(processed_contests)} contest details to: {contests_dir_out}")

    return rating_db


def main() -> None:
    parser = argparse.ArgumentParser(description="USTS Rating Engine - Recalculate Rating Database")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory containing users.csv and contests/ (default: 'data')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output path for rating_db.json",
    )
    args = parser.parse_args()

    output_paths = []
    if args.output:
        output_paths.append(Path(args.output))
    # Also write default web and dist paths
    default_web = Path("web/data/rating_db.json")
    default_dist = Path("dist/data/rating_db.json")
    if default_web not in output_paths:
        output_paths.append(default_web)
    if default_dist not in output_paths:
        output_paths.append(default_dist)

    recalculate_all(data_dir=args.data_dir, output_paths=output_paths)


if __name__ == "__main__":
    main()
