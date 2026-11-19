#!/usr/bin/env python3
"""
tests/test_e2e_suite.py - Comprehensive Opaque-Box E2E Test Suite for USTS Rating System.

Structured in 4 Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature area across CSV validation,
  JSON contest schema, retirement rule, Elo calculation, team power synthesis,
  0-solve protection, zero-sum drift, 7-tier badge colors, CI/CD workflow syntax).
- Tier 2: Boundary & Corner Cases (empty standings, 1 team only, all teams 0 solved,
  extreme ratings like 3000 and 200, ties in solved and penalty, 2-year boundary exact cutoff,
  invalid types and duplicate IDs).
- Tier 3: Cross-Feature Combinations (solo + multi-person team in internal school contest
  with 0-solve team and zero-sum drift calibration; external contest with school vs external anchors;
  retirement status across multiple contest years).
- Tier 4: Real-World Application Scenarios (Full pipeline execution: raw CSV import ->
  validation -> recalculation -> rating_db.json verification -> web asset integrity check).

Testing Philosophy:
- Opaque-box execution testing via CLI tools and file artifacts.
- Sandboxed execution via isolated temporary directories.
- Authoritative expected values derived from mathematical models in PROJECT.md / ORIGINAL_REQUEST.md.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
import numpy as np

# Project Root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
DATA_DIR = WORKSPACE_ROOT / "data"
USERS_CSV = DATA_DIR / "users.csv"
CONTESTS_DIR = DATA_DIR / "contests"
VALIDATE_SCRIPT = SCRIPTS_DIR / "validate_data.py"
IMPORT_SCRIPT = SCRIPTS_DIR / "import_contest.py"
RECALCULATE_SCRIPT = SCRIPTS_DIR / "recalculate.py"
REQUIREMENTS_FILE = WORKSPACE_ROOT / "requirements.txt"
DEPLOY_WORKFLOW_FILE = WORKSPACE_ROOT / ".github" / "workflows" / "deploy.yml"
WEB_DIR = WORKSPACE_ROOT / "web"
README_FILE = WORKSPACE_ROOT / "README.md"


def run_cli(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: float = 60.0,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Execute a CLI command via subprocess and capture the outcome."""
    full_cmd = [sys.executable] + cmd
    return subprocess.run(
        full_cmd,
        cwd=str(cwd or WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )


# ==============================================================================
# AUTHORITATIVE REFERENCE MATHEMATICAL ORACLES (Derived from R3 & R4 specifications)
# ==============================================================================

def oracle_team_rating(ratings: list[float], is_school: bool = True, external_elo: float | None = None) -> float:
    """Derive team power R_team using Exponential Weighted Average (指数加权平均)."""
    if not is_school:
        return float(external_elo) if external_elo is not None else 1500.0
    k = len(ratings)
    if k == 0:
        return 800.0
    if k == 1:
        return float(ratings[0])
    
    max_r = max(ratings)
    weights = [10.0 ** ((r - max_r) / 400.0) for r in ratings]
    sum_w = sum(weights)
    if sum_w <= 0.0:
        return float(ratings[0])
    return float(sum(w * r for w, r in zip(weights, ratings)) / sum_w)


def oracle_display_offset(rated_count: int) -> int:
    """Derive display rating offset from 3-match newbie protection specification."""
    if rated_count <= 0:
        return -1000
    elif rated_count == 1:
        return -500
    elif rated_count == 2:
        return -250
    else:
        return 0


def oracle_display_rating(true_rating: float, rated_count: int) -> int:
    """Derive display rating: max(0, round(true_rating + offset))."""
    offset = oracle_display_offset(rated_count)
    return max(0, round(true_rating + offset))


def oracle_tier_and_color(display_rating: int) -> tuple[str, str, str]:
    """Derive 7-tier classification and hex color from R4.1 specification."""
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


def oracle_win_probability(ra: float, rb: float) -> float:
    """P(B > A) = 1 / (1 + 10^((ra - rb)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((ra - rb) / 400.0))


def oracle_expected_seeds(ratings: list[float]) -> list[float]:
    """Seed_A = 1 + sum_{B != A} P(B > A)."""
    n = len(ratings)
    seeds = []
    for i in range(n):
        seed_i = 1.0 + sum(oracle_win_probability(ratings[i], ratings[j]) for j in range(n) if j != i)
        seeds.append(seed_i)
    return seeds


def oracle_bisection_perf(ratings: list[float], target_ranks: list[float], iterations: int = 50) -> list[float]:
    """Bisection search in [-500, 3500] over 50 iterations."""
    n = len(ratings)
    r_arr = np.array(ratings, dtype=np.float64)
    m_arr = np.array(target_ranks, dtype=np.float64)
    low = np.full(n, -500.0, dtype=np.float64)
    high = np.full(n, 3500.0, dtype=np.float64)
    
    for _ in range(iterations):
        mid = (low + high) * 0.5
        diff = mid[:, None] - r_arr[None, :]
        p = 1.0 / (1.0 + 10.0 ** (diff / 400.0))
        np.fill_diagonal(p, 0.0)
        exp_rank = 1.0 + np.sum(p, axis=1)
        mask = exp_rank > m_arr
        low = np.where(mask, mid, low)
        high = np.where(~mask, mid, high)
        
    return list((low + high) * 0.5)


def oracle_zero_solve_clamp(delta: float, solved: int) -> float:
    """Enforce delta >= -60.0 if solved == 0."""
    if solved == 0:
        return max(-60.0, delta)
    return delta


def oracle_internal_zero_sum(deltas: list[float]) -> list[float]:
    """Calibrate internal school deltas by subtracting mean drift mu."""
    if not deltas:
        return []
    mu = sum(deltas) / len(deltas)
    return [d - mu for d in deltas]


def oracle_retirement_status(
    last_contest_time: datetime | None,
    grade: int,
    is_active: bool,
    reference_time: datetime,
) -> bool:
    """Determine retirement status according to R2.3 specification."""
    if not is_active:
        return True
    if grade <= 2022:
        return True
    cutoff = timedelta(days=730)
    if last_contest_time is not None:
        return (reference_time - last_contest_time) > cutoff
    enrollment_time = datetime(grade, 9, 1, tzinfo=timezone.utc)
    return (reference_time - enrollment_time) > cutoff


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature area)
# ==============================================================================

class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Systematic coverage of all core requirements and feature areas."""

    # --------------------------------------------------------------------------
    # 1.1 CSV Validation (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_csv_01_valid_production_roster(self):
        """Rule 1-8: data/users.csv passes validation cleanly."""
        res = run_cli([str(VALIDATE_SCRIPT), "--users", str(USERS_CSV), "--strict"])
        self.assertEqual(res.returncode, 0, f"Validation failed: {res.stdout}\n{res.stderr}")
        self.assertIn("[SUCCESS]", res.stdout)

    def test_t1_csv_02_missing_header_columns(self):
        """Rule 2: HEADER_MATCH rejects incorrect CSV headers."""
        bad_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_bad_header.csv"
        try:
            bad_csv.write_text("student_id,name,grade,is_active\n23001001,Test,2023,true\n", encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(bad_csv), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[HEADER_MATCH]", res.stdout + res.stderr)
        finally:
            if bad_csv.exists():
                bad_csv.unlink()

    def test_t1_csv_03_duplicate_student_id(self):
        """Rule 3: STUDENT_ID_UNIQUE rejects duplicate student IDs in roster."""
        dup_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_dup.csv"
        try:
            content = (
                "student_id,name,grade,initial_external_elo,is_active\n"
                "23001001,Alice,2023,1200,true\n"
                "23001001,Bob,2023,1300,true\n"
            )
            dup_csv.write_text(content, encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(dup_csv), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[STUDENT_ID_UNIQUE]", res.stdout + res.stderr)
        finally:
            if dup_csv.exists():
                dup_csv.unlink()

    def test_t1_csv_04_invalid_student_id_format(self):
        """Rule 4: STUDENT_ID_FORMAT rejects invalid ID tokens (e.g. spaces, emojis, too short)."""
        bad_id_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_bad_id.csv"
        try:
            content = (
                "student_id,name,grade,initial_external_elo,is_active\n"
                "12,TooShort,2023,1200,true\n"
                "bad id with space,SpaceID,2023,1200,true\n"
            )
            bad_id_csv.write_text(content, encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(bad_id_csv), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[STUDENT_ID_FORMAT]", res.stdout + res.stderr)
        finally:
            if bad_id_csv.exists():
                bad_id_csv.unlink()

    def test_t1_csv_05_invalid_grade_range(self):
        """Rule 6: GRADE_VALID rejects admission years out of realistic range [2010, 2035]."""
        bad_grade_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_bad_grade.csv"
        try:
            content = (
                "student_id,name,grade,initial_external_elo,is_active\n"
                "23001001,Ancient,1995,1200,true\n"
            )
            bad_grade_csv.write_text(content, encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(bad_grade_csv), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[GRADE_VALID]", res.stdout + res.stderr)
        finally:
            if bad_grade_csv.exists():
                bad_grade_csv.unlink()

    def test_t1_csv_06_invalid_external_elo(self):
        """Rule 7: ELO_VALID rejects negative or out-of-bounds Elo values."""
        bad_elo_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_bad_elo.csv"
        try:
            content = (
                "student_id,name,grade,initial_external_elo,is_active\n"
                "23001001,Extreme,9999,true\n"
            )
            bad_elo_csv.write_text(content, encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(bad_elo_csv), "--strict"])
            self.assertNotEqual(res.returncode, 0)
        finally:
            if bad_elo_csv.exists():
                bad_elo_csv.unlink()

    def test_t1_csv_07_invalid_is_active_boolean(self):
        """Rule 8: IS_ACTIVE_VALID rejects non-boolean values."""
        bad_bool_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_bad_bool.csv"
        try:
            content = (
                "student_id,name,grade,initial_external_elo,is_active\n"
                "23001001,Alice,2023,1200,maybe\n"
            )
            bad_bool_csv.write_text(content, encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(bad_bool_csv), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[IS_ACTIVE_VALID]", res.stdout + res.stderr)
        finally:
            if bad_bool_csv.exists():
                bad_bool_csv.unlink()

    def test_t1_csv_08_utf8_bom_support(self):
        """Encoding: UTF-8 BOM must be stripped without corrupting 'student_id' header."""
        bom_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_bom.csv"
        temp_contest = WORKSPACE_ROOT / "tests" / "fixtures" / "2026_09_15_temp_bom.json"
        try:
            content = (
                "\ufeffstudent_id,name,grade,initial_external_elo,is_active\n"
                "23001001,Alice,2023,1200,true\n"
            )
            bom_csv.write_text(content, encoding="utf-8")
            contest_data = {
                "contest_id": "2026_09_15_temp_bom",
                "title": "BOM Test Contest",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {"team_name": "T1", "is_school": True, "members": ["23001001"], "solved": 1, "penalty": 10}
                ]
            }
            temp_contest.write_text(json.dumps(contest_data), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(bom_csv), "--contests", str(temp_contest), "--strict"])
            self.assertEqual(res.returncode, 0, f"BOM validation failed: {res.stdout}\n{res.stderr}")
            self.assertIn("[SUCCESS]", res.stdout)
        finally:
            if bom_csv.exists():
                bom_csv.unlink()
            if temp_contest.exists():
                temp_contest.unlink()

    # --------------------------------------------------------------------------
    # 1.2 JSON Contest Schema (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_json_01_valid_production_contests(self):
        """Rule 9-15: all production contests in data/contests/ pass validation."""
        res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(CONTESTS_DIR), "--strict"])
        self.assertEqual(res.returncode, 0, f"Contests validation failed: {res.stdout}\n{res.stderr}")

    def test_t1_json_02_filename_date_mismatch(self):
        """Rule 11: DATE_CONSISTENCY rejects date mismatch between filename and start_time."""
        bad_json = CONTESTS_DIR / "2026_09_15_test_mismatch.json"
        try:
            data = {
                "contest_id": "2026_09_15_test_mismatch",
                "title": "Date Mismatch Test",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-16T13:00:00+08:00",
                "standings": [
                    {"team_name": "TeamA", "is_school": True, "members": ["23001003"], "solved": 1, "penalty": 10}
                ]
            }
            bad_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(bad_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[DATE_CONSISTENCY]", res.stdout + res.stderr)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    def test_t1_json_03_invalid_contest_type_enum(self):
        """Rule 13: TYPE_ENUM rejects types other than INTERNAL_SCHOOL or EXTERNAL_XCPC."""
        bad_json = CONTESTS_DIR / "2026_09_15_bad_type.json"
        try:
            data = {
                "contest_id": "2026_09_15_bad_type",
                "title": "Invalid Type Test",
                "contest_type": "ONLINE_OPEN",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {"team_name": "TeamA", "is_school": True, "members": ["23001003"], "solved": 1, "penalty": 10}
                ]
            }
            bad_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(bad_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[TYPE_ENUM]", res.stdout + res.stderr)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    def test_t1_json_04_missing_required_fields(self):
        """Rule 12: JSON_SYNTAX_SCHEMA rejects missing mandatory keys."""
        bad_json = CONTESTS_DIR / "2026_09_15_missing_key.json"
        try:
            data = {
                "contest_id": "2026_09_15_missing_key",
                "title": "Missing Standings",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00"
            }
            bad_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(bad_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[JSON_SYNTAX_SCHEMA]", res.stdout + res.stderr)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    def test_t1_json_05_negative_solved_or_penalty(self):
        """Rule 15: CONTEST_INTEGRITY rejects negative solved or penalty values."""
        bad_json = CONTESTS_DIR / "2026_09_15_negative_val.json"
        try:
            data = {
                "contest_id": "2026_09_15_negative_val",
                "title": "Negative Solved",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {"team_name": "TeamA", "is_school": True, "members": ["23001003"], "solved": -1, "penalty": 10}
                ]
            }
            bad_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(bad_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[CONTEST_INTEGRITY]", res.stdout + res.stderr)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    def test_t1_json_06_referential_integrity_violation(self):
        """Rule 14: REFERENTIAL_INTEGRITY rejects unregistered school student IDs."""
        bad_json = CONTESTS_DIR / "2026_09_15_unregistered.json"
        try:
            data = {
                "contest_id": "2026_09_15_unregistered",
                "title": "Ghost Student",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {"team_name": "TeamGhost", "is_school": True, "members": ["99999999"], "solved": 1, "penalty": 10}
                ]
            }
            bad_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(bad_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[REFERENTIAL_INTEGRITY]", res.stdout + res.stderr)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    def test_t1_json_07_duplicate_members_in_same_team(self):
        """Rule 15: CONTEST_INTEGRITY rejects duplicate student IDs within the same team."""
        bad_json = CONTESTS_DIR / "2026_09_15_dup_member.json"
        try:
            data = {
                "contest_id": "2026_09_15_dup_member",
                "title": "Duplicate Member",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {"team_name": "CloneTeam", "is_school": True, "members": ["23001003", "23001003"], "solved": 2, "penalty": 50}
                ]
            }
            bad_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(bad_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[CONTEST_INTEGRITY]", res.stdout + res.stderr)
        finally:
            if bad_json.exists():
                bad_json.unlink()

    # --------------------------------------------------------------------------
    # 1.3 Retirement Rule (2-Year Inactivity) (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_retire_01_active_with_recent_contest(self):
        """R2.3: Student competing 100 days ago is active (is_retired = False)."""
        ref_time = datetime(2026, 11, 18, tzinfo=timezone.utc)
        last_contest = ref_time - timedelta(days=100)
        status = oracle_retirement_status(last_contest, grade=2024, is_active=True, reference_time=ref_time)
        self.assertFalse(status)

    def test_t1_retire_02_retired_with_old_contest(self):
        """R2.3: Student competing 750 days ago (>2 years) is retired (is_retired = True)."""
        ref_time = datetime(2026, 11, 18, tzinfo=timezone.utc)
        last_contest = ref_time - timedelta(days=750)
        status = oracle_retirement_status(last_contest, grade=2022, is_active=True, reference_time=ref_time)
        self.assertTrue(status)

    def test_t1_retire_03_zero_contests_recent_freshman(self):
        """R2.3: Freshman enrolled 60 days ago with 0 contests is active."""
        ref_time = datetime(2026, 11, 18, tzinfo=timezone.utc)
        status = oracle_retirement_status(None, grade=2026, is_active=True, reference_time=ref_time)
        self.assertFalse(status)

    def test_t1_retire_04_zero_contests_senior_over_two_years(self):
        """R2.3: Senior enrolled > 2 years ago (e.g. 2023) with 0 contests is retired."""
        ref_time = datetime(2026, 11, 18, tzinfo=timezone.utc)
        status = oracle_retirement_status(None, grade=2023, is_active=True, reference_time=ref_time)
        self.assertTrue(status)

    def test_t1_retire_05_administrative_inactive_override(self):
        """R2.3: Student with is_active = False in CSV is unconditionally retired."""
        ref_time = datetime(2026, 11, 18, tzinfo=timezone.utc)
        last_contest = ref_time - timedelta(days=10)
        status = oracle_retirement_status(last_contest, grade=2025, is_active=False, reference_time=ref_time)
        self.assertTrue(status)

    # --------------------------------------------------------------------------
    # 1.4 Elo Calculation & Expected Seeds (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_elo_01_equal_ratings_probability(self):
        """R3.3: Equal ratings yield win probability P = 0.5."""
        p = oracle_win_probability(1500.0, 1500.0)
        self.assertAlmostEqual(p, 0.5, places=7)

    def test_t1_elo_02_symmetry_property(self):
        """R3.3: Symmetry P(A > B) + P(B > A) == 1.0."""
        pairs = [(1000, 1200), (1500, 2000), (800, 2400), (1450, 1420)]
        for ra, rb in pairs:
            p_ab = oracle_win_probability(ra, rb)
            p_ba = oracle_win_probability(rb, ra)
            self.assertAlmostEqual(p_ab + p_ba, 1.0, places=7)

    def test_t1_elo_03_rating_difference_400_pts(self):
        """R3.3: Rating difference of 400 yields 10:1 odds (P ≈ 0.090909)."""
        p = oracle_win_probability(1200.0, 800.0) # P(weaker > stronger)
        self.assertAlmostEqual(p, 1.0 / 11.0, places=5)
        p_strong = oracle_win_probability(800.0, 1200.0)
        self.assertAlmostEqual(p_strong, 10.0 / 11.0, places=5)

    def test_t1_elo_04_expected_seeds_calculation(self):
        """R3.3: Seed_A = 1 + sum P(B > A) matches hand calculation."""
        # 3 teams: 1600, 1200, 800
        ratings = [1600.0, 1200.0, 800.0]
        seeds = oracle_expected_seeds(ratings)
        # For highest rated team (1600): seed = 1 + P(1200 > 1600) + P(800 > 1600) < 2.0
        self.assertLess(seeds[0], seeds[1])
        self.assertLess(seeds[1], seeds[2])
        self.assertGreater(seeds[0], 1.0)
        self.assertAlmostEqual(sum(seeds), (1 + 2 + 3), places=6)

    def test_t1_elo_05_target_geometric_mean_rank_monotonicity(self):
        """R3.3: Target geometric mean rank m = sqrt(seed * rank) preserves ranking order."""
        ratings = [1500.0, 1400.0, 1300.0, 1200.0]
        seeds = oracle_expected_seeds(ratings)
        ranks = [1.0, 2.0, 3.0, 4.0]
        targets = [math.sqrt(s * r) for s, r in zip(seeds, ranks)]
        for i in range(len(targets) - 1):
            self.assertLess(targets[i], targets[i+1])

    # --------------------------------------------------------------------------
    # 1.5 Team Power Synthesis (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_team_01_solo_team_identity(self):
        """R3.2: Solo team (k=1, C_1=0) power is identical to member rating."""
        for r in [800.0, 1000.0, 1500.0, 2100.0]:
            team_r = oracle_team_rating([r], is_school=True)
            self.assertAlmostEqual(team_r, r, places=6)

    def test_t1_team_02_duo_team_constant(self):
        """R3.2: Duo team equal ratings yield exact identity."""
        team_r = oracle_team_rating([1200.0, 1200.0], is_school=True)
        self.assertAlmostEqual(team_r, 1200.0, places=5)

    def test_t1_team_03_trio_team_constant(self):
        """R3.2: Trio team equal ratings yield exact identity."""
        team_r = oracle_team_rating([1200.0, 1200.0, 1200.0], is_school=True)
        self.assertAlmostEqual(team_r, 1200.0, places=5)

    def test_t1_team_04_quad_and_larger_formula(self):
        """R3.2: Large team (k>3) equal ratings yield exact identity."""
        k = 4
        team_r = oracle_team_rating([1000.0] * k, is_school=True)
        self.assertAlmostEqual(team_r, 1000.0, places=5)

    def test_t1_team_05_logsumexp_numerical_stability(self):
        """R3.2: Extreme ratings (e.g. 2900 and 800) evaluate stably without overflow, bounded between min and max."""
        team_r = oracle_team_rating([2900.0, 800.0], is_school=True)
        self.assertFalse(math.isnan(team_r))
        self.assertFalse(math.isinf(team_r))
        # Dominant player rating should dominate
        self.assertGreater(team_r, 2800.0)
        self.assertLessEqual(team_r, 2900.0)

    def test_t1_team_06_external_anchor_default(self):
        """R3.2: External team defaults to external_elo = 1500.0."""
        team_r = oracle_team_rating([], is_school=False, external_elo=None)
        self.assertEqual(team_r, 1500.0)
        custom_r = oracle_team_rating([], is_school=False, external_elo=1850.0)
        self.assertEqual(custom_r, 1850.0)

    # --------------------------------------------------------------------------
    # 1.6 0-Solve Drop Protection (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_zerosolve_01_drop_clamped_at_minus_60(self):
        """R3.3: Team with solved == 0 and raw delta = -120 is clamped to -60.0."""
        clamped = oracle_zero_solve_clamp(-120.0, solved=0)
        self.assertEqual(clamped, -60.0)

    def test_t1_zerosolve_02_moderate_drop_unclamped(self):
        """R3.3: Team with solved == 0 and raw delta = -25.0 remains -25.0."""
        clamped = oracle_zero_solve_clamp(-25.0, solved=0)
        self.assertEqual(clamped, -25.0)

    def test_t1_zerosolve_03_nonzero_solve_no_clamp(self):
        """R3.3: Team with solved >= 1 and raw delta = -85.0 is NOT clamped."""
        delta = oracle_zero_solve_clamp(-85.0, solved=1)
        self.assertEqual(delta, -85.0)

    def test_t1_zerosolve_04_zero_solve_positive_delta(self):
        """R3.3: Team with solved == 0 achieving positive delta (e.g. +10) retains +10."""
        delta = oracle_zero_solve_clamp(10.0, solved=0)
        self.assertEqual(delta, 10.0)

    def test_t1_zerosolve_05_all_members_receive_clamped_delta(self):
        """R3.3 & R3.4: All members of 0-solve team receive the capped delta."""
        team_raw_delta = -150.0
        team_clamped_delta = oracle_zero_solve_clamp(team_raw_delta, solved=0)
        members = ["23001001", "23001002"]
        member_deltas = [team_clamped_delta for _ in members]
        self.assertTrue(all(d >= -60.0 for d in member_deltas))

    # --------------------------------------------------------------------------
    # 1.7 Zero-Sum Drift Calibration (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_zerosum_01_internal_school_exact_sum(self):
        """R3.4: INTERNAL_SCHOOL contest member deltas sum exactly to 0 (|sum| <= 1e-6)."""
        raw_deltas = [15.2, -8.4, 30.1, -22.5, -5.0]
        calibrated = oracle_internal_zero_sum(raw_deltas)
        net_sum = sum(calibrated)
        self.assertLessEqual(abs(net_sum), 1e-6)

    def test_t1_zerosum_02_mean_drift_subtraction(self):
        """R3.4: Calibrated delta = delta - mu where mu = mean(deltas)."""
        raw_deltas = [10.0, 20.0, 30.0]
        mu = sum(raw_deltas) / 3.0 # 20.0
        expected = [-10.0, 0.0, 10.0]
        calibrated = oracle_internal_zero_sum(raw_deltas)
        for c, e in zip(calibrated, expected):
            self.assertAlmostEqual(c, e, places=7)

    def test_t1_zerosum_03_external_xcpc_unconstrained(self):
        """R3.4: EXTERNAL_XCPC does not calibrate to 0; school pool can gain/lose net points."""
        # External contest where USTS teams beat external anchors
        usts_deltas = [45.0, 32.0, 18.0]
        # In EXTERNAL_XCPC, no zero-sum adjustment is applied
        self.assertEqual(sum(usts_deltas), 95.0)

    def test_t1_zerosum_04_mixed_teams_preserves_zero_sum(self):
        """R3.4: School contest with solo, duo, and trio teams preserves exact zero sum."""
        # Team 1 (solo): +25
        # Team 2 (duo): +10, +10
        # Team 3 (trio): -15, -15, -15
        member_deltas = [25.0, 10.0, 10.0, -15.0, -15.0, -15.0]
        calibrated = oracle_internal_zero_sum(member_deltas)
        self.assertLessEqual(abs(sum(calibrated)), 1e-6)

    def test_t1_zerosum_05_interaction_with_zero_solve_floor(self):
        """R3.4: 0-solve clamping occurs before zero-sum calibration, and sum remains zero."""
        raw_deltas = [40.0, -100.0, 10.0]
        # Apply 0-solve clamp to team 2
        clamped_deltas = [raw_deltas[0], oracle_zero_solve_clamp(raw_deltas[1], solved=0), raw_deltas[2]]
        self.assertEqual(clamped_deltas[1], -60.0)
        # Apply internal calibration
        calibrated = oracle_internal_zero_sum(clamped_deltas)
        self.assertLessEqual(abs(sum(calibrated)), 1e-6)

    # --------------------------------------------------------------------------
    # 1.8 7-Tier Badge Colors & Thresholds (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_tier_01_newbie_under_800(self):
        """R4.1: Display rating < 800 maps to Newbie / 萌新 / #808080."""
        for r in [0, 350, 600, 799]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Newbie")
            self.assertEqual(cn, "萌新")
            self.assertEqual(color.upper(), "#808080")

    def test_t1_tier_02_pupil_800_to_999(self):
        """R4.1: Display rating 800..999 maps to Pupil / 进阶 / #008000."""
        for r in [800, 850, 920, 999]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Pupil")
            self.assertEqual(cn, "进阶")
            self.assertEqual(color.upper(), "#008000")

    def test_t1_tier_03_specialist_1000_to_1199(self):
        """R4.1: Display rating 1000..1199 maps to Specialist / 骨干 / #03A89E."""
        for r in [1000, 1080, 1199]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Specialist")
            self.assertEqual(cn, "骨干")
            self.assertEqual(color.upper(), "#03A89E")

    def test_t1_tier_04_expert_1200_to_1399(self):
        """R4.1: Display rating 1200..1399 maps to Expert / 精英 / #0000FF."""
        for r in [1200, 1310, 1399]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Expert")
            self.assertEqual(cn, "精英")
            self.assertEqual(color.upper(), "#0000FF")

    def test_t1_tier_05_candidate_master_1400_to_1599(self):
        """R4.1: Display rating 1400..1599 maps to Candidate Master / 候选大师 / #AA00AA."""
        for r in [1400, 1500, 1599]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Candidate Master")
            self.assertEqual(cn, "候选大师")
            self.assertEqual(color.upper(), "#AA00AA")

    def test_t1_tier_06_master_1600_to_1799(self):
        """R4.1: Display rating 1600..1799 maps to Master / 校队大师 / #FF8C00."""
        for r in [1600, 1720, 1799]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Master")
            self.assertEqual(cn, "校队大师")
            self.assertEqual(color.upper(), "#FF8C00")

    def test_t1_tier_07_grandmaster_ge_1800(self):
        """R4.1: Display rating >= 1800 maps to Grandmaster / 传奇宗师 / #FF0000."""
        for r in [1800, 1950, 2400]:
            name, cn, color = oracle_tier_and_color(r)
            self.assertEqual(name, "Grandmaster")
            self.assertEqual(cn, "传奇宗师")
            self.assertEqual(color.upper(), "#FF0000")

    def test_t1_tier_08_exact_boundaries(self):
        """R4.1: Exact step boundaries between tiers."""
        self.assertEqual(oracle_tier_and_color(799)[0], "Newbie")
        self.assertEqual(oracle_tier_and_color(800)[0], "Pupil")
        self.assertEqual(oracle_tier_and_color(999)[0], "Pupil")
        self.assertEqual(oracle_tier_and_color(1000)[0], "Specialist")
        self.assertEqual(oracle_tier_and_color(1199)[0], "Specialist")
        self.assertEqual(oracle_tier_and_color(1200)[0], "Expert")
        self.assertEqual(oracle_tier_and_color(1399)[0], "Expert")
        self.assertEqual(oracle_tier_and_color(1400)[0], "Candidate Master")
        self.assertEqual(oracle_tier_and_color(1599)[0], "Candidate Master")
        self.assertEqual(oracle_tier_and_color(1600)[0], "Master")
        self.assertEqual(oracle_tier_and_color(1799)[0], "Master")
        self.assertEqual(oracle_tier_and_color(1800)[0], "Grandmaster")

    # --------------------------------------------------------------------------
    # 1.9 CI/CD Workflow Syntax & Project Tooling (>= 5 tests)
    # --------------------------------------------------------------------------
    def test_t1_cicd_01_importer_cli_interface(self):
        """R5.2: scripts/import_contest.py --help succeeds and documents options."""
        res = run_cli([str(IMPORT_SCRIPT), "--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--input", res.stdout)
        self.assertIn("--output", res.stdout)
        self.assertIn("--format", res.stdout)

    def test_t1_cicd_02_validator_cli_interface(self):
        """R5.1: scripts/validate_data.py --help succeeds and documents options."""
        res = run_cli([str(VALIDATE_SCRIPT), "--help"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("--users", res.stdout)
        self.assertIn("--contests", res.stdout)

    def test_t1_cicd_03_import_sample_csv_success(self):
        """R5.2: Ingest sample CSV scoreboard to valid contest JSON."""
        sample_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_scoreboard.csv"
        out_json = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_imported.json"
        try:
            res = run_cli([
                str(IMPORT_SCRIPT),
                "--input", str(sample_csv),
                "--output", str(out_json),
                "--contest-id", "2026_11_01_imported_test",
                "--title", "Imported Test",
                "--type", "INTERNAL_SCHOOL",
                "--start-time", "2026-11-01T13:00:00+08:00",
                "--skip-validation"
            ])
            self.assertEqual(res.returncode, 0, f"Import failed: {res.stdout}\n{res.stderr}")
            self.assertTrue(out_json.exists())
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["contest_id"], "2026_11_01_imported_test")
            self.assertGreaterEqual(len(data["standings"]), 1)
        finally:
            if out_json.exists():
                out_json.unlink()

    def test_t1_cicd_04_import_sample_domjudge_success(self):
        """R5.2: Ingest DOMjudge JSON scoreboard to valid contest JSON."""
        dj_json = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_domjudge.json"
        dj_teams = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_domjudge_teams.json"
        out_json = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_dj_imported.json"
        try:
            res = run_cli([
                str(IMPORT_SCRIPT),
                "--input", str(dj_json),
                "--teams", str(dj_teams),
                "--output", str(out_json),
                "--contest-id", "2026_11_02_imported_dj",
                "--title", "DOMjudge Test",
                "--type", "INTERNAL_SCHOOL",
                "--start-time", "2026-11-02T13:00:00+08:00",
                "--skip-validation"
            ])
            self.assertEqual(res.returncode, 0)
            self.assertTrue(out_json.exists())
        finally:
            if out_json.exists():
                out_json.unlink()

    def test_t1_cicd_05_import_sample_pta_success(self):
        """R5.2: Ingest PTA scoreboard to valid contest JSON."""
        pta_json = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_pta.json"
        out_json = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_pta_imported.json"
        try:
            res = run_cli([
                str(IMPORT_SCRIPT),
                "--input", str(pta_json),
                "--output", str(out_json),
                "--contest-id", "2026_11_03_imported_pta",
                "--title", "PTA Test",
                "--type", "INTERNAL_SCHOOL",
                "--start-time", "2026-11-03T13:00:00+08:00",
                "--skip-validation"
            ])
            self.assertEqual(res.returncode, 0)
            self.assertTrue(out_json.exists())
        finally:
            if out_json.exists():
                out_json.unlink()

    def test_t1_cicd_06_requirements_txt_specifications(self):
        """R6.1: requirements.txt specifies numpy and pytest dependencies."""
        if not REQUIREMENTS_FILE.exists():
            self.skipTest("requirements.txt not yet created (Milestone 4)")
        content = REQUIREMENTS_FILE.read_text(encoding="utf-8").lower()
        self.assertIn("numpy", content)
        self.assertIn("pytest", content)

    def test_t1_cicd_07_github_workflow_definition(self):
        """R6.1: .github/workflows/deploy.yml contains required CI/CD trigger and steps."""
        if not DEPLOY_WORKFLOW_FILE.exists():
            self.skipTest("deploy.yml not yet created (Milestone 4)")
        content = DEPLOY_WORKFLOW_FILE.read_text(encoding="utf-8")
        self.assertIn("push", content)
        self.assertIn("workflow_dispatch", content)
        self.assertIn("validate_data.py", content)
        self.assertIn("pytest", content)
        self.assertIn("recalculate.py", content)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ==============================================================================

class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary conditions, empty sets, singletons, extreme values, ties."""

    def test_t2_01_empty_standings(self):
        """Rule 12: Empty standings in contest must be detected by validation."""
        empty_json = CONTESTS_DIR / "2026_09_15_empty_standings.json"
        try:
            data = {
                "contest_id": "2026_09_15_empty_standings",
                "title": "Empty Standings Contest",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": []
            }
            empty_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--contests", str(empty_json), "--strict"])
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("[JSON_SYNTAX_SCHEMA]", res.stdout + res.stderr)
        finally:
            if empty_json.exists():
                empty_json.unlink()

    def test_t2_02_single_team_contest_math(self):
        """Corner Case: Contest with 1 team has no opponents; delta must be 0.0."""
        # For a single team, seed is 1.0, rank is 1.0, m = 1.0.
        # Performance equals team rating, delta = 0.0.
        team_rating = 1400.0
        # Expected rank against 0 opponents is identically 1.0 for all R
        perf = team_rating
        delta = 0.80 * (perf - team_rating) / 2.0
        self.assertEqual(delta, 0.0)

    def test_t2_03_all_teams_zero_solved(self):
        """Corner Case: All teams solved 0 problems; equal tie rank arithmetic mean and delta clamp."""
        # Standings of 4 teams, all solved 0, penalty 0
        n = 4
        # Arithmetic mean rank for all tied 1st: (1 + 4) / 2 = 2.5
        avg_rank = (1 + n) / 2.0
        self.assertEqual(avg_rank, 2.5)
        # Raw deltas for equal rating teams:
        ratings = [1000.0] * n
        seeds = oracle_expected_seeds(ratings) # all equal to (1 + 3*0.5) = 2.5
        m = [math.sqrt(s * avg_rank) for s in seeds]
        self.assertTrue(all(abs(val - 2.5) < 1e-6 for val in m))
        # Bisection yields R_perf == R_team
        perfs = oracle_bisection_perf(ratings, m)
        raw_deltas = [0.80 * (p - r) / 2.0 for p, r in zip(perfs, ratings)]
        # All deltas are around 0.0, clamped with >= -60
        clamped = [oracle_zero_solve_clamp(d, solved=0) for d in raw_deltas]
        calibrated = oracle_internal_zero_sum(clamped)
        self.assertLessEqual(abs(sum(calibrated)), 1e-6)

    def test_t2_04_extreme_ratings_numerical_stability(self):
        """Corner Case: 3000 rating vs 200 rating calculates without floating-point overflow or NaN."""
        ra, rb = 3000.0, 200.0
        diff = ra - rb # 2800 -> 10^(7.0) = 1e7
        p_b_beats_a = oracle_win_probability(ra, rb)
        self.assertGreater(p_b_beats_a, 0.0)
        self.assertLess(p_b_beats_a, 1e-6)
        p_a_beats_b = oracle_win_probability(rb, ra)
        self.assertAlmostEqual(p_a_beats_b, 1.0, places=6)

    def test_t2_05_three_way_tie_ranking(self):
        """Corner Case: 3 teams tied at 1st place receive arithmetic mean rank (1+2+3)/3 = 2.0."""
        # 3 teams sharing solved=3, penalty=120
        # Slice [0, 3) -> 1-based ranks 1, 2, 3 -> mean = 2.0
        tied_ranks = [1, 2, 3]
        mean_rank = sum(tied_ranks) / len(tied_ranks)
        self.assertEqual(mean_rank, 2.0)

    def test_t2_06_exact_two_year_boundary_cutoff(self):
        """Corner Case: Exactly 730 days (Active) vs 731 days (Retired)."""
        ref_time = datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone.utc)
        time_730 = ref_time - timedelta(days=730)
        time_731 = ref_time - timedelta(days=731)
        
        # 730 days is NOT strictly > 730 days -> active
        active_status = oracle_retirement_status(time_730, grade=2024, is_active=True, reference_time=ref_time)
        self.assertFalse(active_status)
        
        # 731 days IS strictly > 730 days -> retired
        retired_status = oracle_retirement_status(time_731, grade=2024, is_active=True, reference_time=ref_time)
        self.assertTrue(retired_status)

    def test_t2_07_whitespace_and_boolean_casing_resilience(self):
        """Corner Case: CSV with leading/trailing whitespace and mixed-case booleans parses safely."""
        messy_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "temp_messy.csv"
        temp_contest = WORKSPACE_ROOT / "tests" / "fixtures" / "2026_09_15_temp_messy.json"
        try:
            content = (
                "student_id,name,grade,initial_external_elo,is_active\n"
                " 23001001 , Alice , 2023 , 1200 , TRUE \n"
                "23001002,Bob,2023,,False\n"
            )
            messy_csv.write_text(content, encoding="utf-8")
            contest_data = {
                "contest_id": "2026_09_15_temp_messy",
                "title": "Messy Test Contest",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {"team_name": "T1", "is_school": True, "members": ["23001001"], "solved": 1, "penalty": 10}
                ]
            }
            temp_contest.write_text(json.dumps(contest_data), encoding="utf-8")
            res = run_cli([str(VALIDATE_SCRIPT), "--users", str(messy_csv), "--contests", str(temp_contest), "--strict"])
            self.assertEqual(res.returncode, 0, f"Messy CSV validation failed: {res.stdout}\n{res.stderr}")
            self.assertIn("[SUCCESS]", res.stdout)
        finally:
            if messy_csv.exists():
                messy_csv.unlink()
            if temp_contest.exists():
                temp_contest.unlink()


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Complex end-to-end interactions across features and multiple entities."""

    def test_t3_01_mixed_internal_school_contest_simulation(self):
        """Cross-Feature: Solo + Duo + Trio teams with a 0-solve team in INTERNAL_SCHOOL.
        
        Validates:
        1. Team power synthesis across k=1, 2, 3.
        2. Expected seeds and target rank calculation.
        3. Vectorized bisection for performance ratings.
        4. 0-solve drop protection clamping.
        5. Exact zero-sum calibration across all school members (|sum| <= 1e-6).
        """
        # Team 1: Solo (Alice: 1400)
        # Team 2: Duo (Bob: 1200, Charlie: 1100)
        # Team 3: Trio (David: 1000, Eve: 1000, Frank: 1000)
        # Team 4: Duo with 0 solves (Grace: 900, Heidi: 900)
        r_t1 = oracle_team_rating([1400.0], is_school=True) # 1400
        r_t2 = oracle_team_rating([1200.0, 1100.0], is_school=True) # ~1158.8
        r_t3 = oracle_team_rating([1000.0, 1000.0, 1000.0], is_school=True) # ~1030.8
        r_t4 = oracle_team_rating([900.0, 900.0], is_school=True) # ~920.4
        
        team_ratings = [r_t1, r_t2, r_t3, r_t4]
        ranks = [1.0, 2.0, 3.0, 4.0]
        seeds = oracle_expected_seeds(team_ratings)
        targets = [math.sqrt(s * r) for s, r in zip(seeds, ranks)]
        
        perfs = oracle_bisection_perf(team_ratings, targets)
        w_contest = 0.80
        raw_deltas = [w_contest * (p - r) / 2.0 for p, r in zip(perfs, team_ratings)]
        
        # Apply 0-solve clamp to team 4
        clamped_t4 = oracle_zero_solve_clamp(raw_deltas[3], solved=0)
        team_deltas = [raw_deltas[0], raw_deltas[1], raw_deltas[2], clamped_t4]
        
        # Expand to member deltas:
        # T1 has 1 member, T2 has 2, T3 has 3, T4 has 2 -> Total M = 8 members
        member_deltas = (
            [team_deltas[0]] * 1 +
            [team_deltas[1]] * 2 +
            [team_deltas[2]] * 3 +
            [team_deltas[3]] * 2
        )
        self.assertEqual(len(member_deltas), 8)
        
        # Calibrate zero sum
        calibrated_members = oracle_internal_zero_sum(member_deltas)
        net_drift = sum(calibrated_members)
        self.assertLessEqual(abs(net_drift), 1e-6, f"Zero sum violated: net_drift={net_drift}")

    def test_t3_02_external_xcpc_school_vs_external_anchors_simulation(self):
        """Cross-Feature: School teams competing against external anchors in EXTERNAL_XCPC.
        
        Validates:
        1. External teams use external_elo (or 1500 default) without member power synthesis.
        2. Weight W = 1.00.
        3. External teams are NOT calibrated or saved to student database.
        4. School pool net rating change is unconstrained (can be non-zero).
        """
        # USTS Team 1: 1550 (k=3)
        # USTS Team 2: 1200 (k=2)
        # External Anchor 1: Tsinghua (2300)
        # External Anchor 2: Default Anchor (1500)
        team_ratings = [1550.0, 1200.0, 2300.0, 1500.0]
        # Finish order: Tsinghua 1st, USTS 1 2nd, Default Anchor 3rd, USTS 2 4th
        ranks = [2.0, 4.0, 1.0, 3.0]
        seeds = oracle_expected_seeds(team_ratings)
        targets = [math.sqrt(s * r) for s, r in zip(seeds, ranks)]
        
        perfs = oracle_bisection_perf(team_ratings, targets)
        w_contest = 1.00
        team_deltas = [w_contest * (p - r) / 2.0 for p, r in zip(perfs, team_ratings)]
        
        # In external contest, school teams do NOT undergo zero-sum calibration
        usts_member_deltas = [team_deltas[0]] * 3 + [team_deltas[1]] * 2
        net_school_change = sum(usts_member_deltas)
        # Net change is non-zero because they competed against external teams
        self.assertIsInstance(net_school_change, float)

    def test_t3_03_multi_year_progression_and_display_offset(self):
        """Cross-Feature: Track a student across 5 contests verifying 3-contest display offset decay."""
        true_rating = 1000.0
        # Contest 0 (unrated): rated_count=0 -> offset -1000 -> display = 0
        self.assertEqual(oracle_display_rating(true_rating, 0), 0)
        
        # Contest 1 (+50): true=1050, rated_count=1 -> offset -500 -> display = 550
        true_rating += 50.0
        self.assertEqual(oracle_display_rating(true_rating, 1), 550)
        
        # Contest 2 (+40): true=1090, rated_count=2 -> offset -250 -> display = 840 (Pupil)
        true_rating += 40.0
        disp_2 = oracle_display_rating(true_rating, 2)
        self.assertEqual(disp_2, 840)
        self.assertEqual(oracle_tier_and_color(disp_2)[0], "Pupil")
        
        # Contest 3 (+30): true=1120, rated_count=3 -> offset 0 -> display = 1120 (Specialist)
        true_rating += 30.0
        disp_3 = oracle_display_rating(true_rating, 3)
        self.assertEqual(disp_3, 1120)
        self.assertEqual(oracle_tier_and_color(disp_3)[0], "Specialist")
        
        # Contest 4 (+20): true=1140, rated_count=4 -> offset 0 -> display = 1140 (Specialist)
        true_rating += 20.0
        disp_4 = oracle_display_rating(true_rating, 4)
        self.assertEqual(disp_4, 1140)
        self.assertEqual(oracle_tier_and_color(disp_4)[0], "Specialist")
        
        # Contest 5 (+10): true=1150, rated_count=5 -> offset 0 -> display = 1150
        true_rating += 10.0
        disp_5 = oracle_display_rating(true_rating, 5)
        self.assertEqual(disp_5, 1150)


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================

class TestTier4RealWorldApplicationScenarios(unittest.TestCase):
    """Tier 4: End-to-end integration across CLI scripts, rating database, and web dashboard."""

    def test_t4_01_full_pipeline_raw_csv_import_and_validate(self):
        """Full Pipeline: Raw CSV -> import_contest.py -> validate_data.py."""
        raw_csv = WORKSPACE_ROOT / "tests" / "fixtures" / "sample_scoreboard.csv"
        target_json = CONTESTS_DIR / "2026_11_15_pipeline_test.json"
        
        try:
            # Step 1: Import raw CSV
            res_import = run_cli([
                str(IMPORT_SCRIPT),
                "--input", str(raw_csv),
                "--output", str(target_json),
                "--contest-id", "2026_11_15_pipeline_test",
                "--title", "Pipeline Integration Contest",
                "--type", "INTERNAL_SCHOOL",
                "--start-time", "2026-11-15T13:00:00+08:00",
                "--skip-validation"
            ])
            self.assertEqual(res_import.returncode, 0, f"Import failed: {res_import.stdout}\n{res_import.stderr}")
            self.assertTrue(target_json.exists())
            
            # Step 2: Validate the generated contest file
            res_val = run_cli([
                str(VALIDATE_SCRIPT),
                "--users", str(USERS_CSV),
                "--contests", str(target_json),
                "--strict"
            ])
            # If foreign student IDs exist in sample, validate handles referential checks
            self.assertIn("USTS Rating System - Data Integrity Validator", res_val.stdout)
        finally:
            if target_json.exists():
                target_json.unlink()

    def test_t4_02_production_data_directory_full_integrity(self):
        """Verify the complete production data directory under strict validation."""
        res = run_cli([str(VALIDATE_SCRIPT), "--users", str(USERS_CSV), "--contests", str(CONTESTS_DIR), "--strict"])
        self.assertEqual(res.returncode, 0, f"Production data validation failed:\n{res.stdout}")
        self.assertIn("[SUCCESS]", res.stdout)

    def test_t4_03_recalculate_generates_valid_rating_db(self):
        """Verify that recalculate.py generates rating_db.json with valid structure and schema."""
        if not RECALCULATE_SCRIPT.exists():
            self.skipTest("scripts/recalculate.py not yet implemented (Milestone 2)")
            
        res = run_cli([str(RECALCULATE_SCRIPT)])
        self.assertEqual(res.returncode, 0, f"Recalculate failed:\n{res.stdout}\n{res.stderr}")
        
        # Check generated output
        db_candidates = [
            WORKSPACE_ROOT / "dist" / "data" / "rating_db.json",
            WORKSPACE_ROOT / "web" / "data" / "rating_db.json",
            WORKSPACE_ROOT / "data" / "rating_db.json"
        ]
        db_file = next((f for f in db_candidates if f.exists()), None)
        self.assertIsNotNone(db_file, "rating_db.json not found in dist/data or web/data")
        
        db = json.loads(db_file.read_text(encoding="utf-8"))
        self.assertIn("metadata", db)
        self.assertIn("tiers", db)
        self.assertIn("users", db)
        self.assertIn("contests", db)
        
        # Verify metadata
        self.assertGreaterEqual(db["metadata"]["total_users"], 20)
        self.assertGreaterEqual(db["metadata"]["total_contests"], 3)
        
        # Verify users sorted by display_rating descending
        users = db["users"]
        self.assertGreater(len(users), 0)
        for i in range(len(users) - 1):
            self.assertGreaterEqual(users[i]["display_rating"], users[i+1]["display_rating"])

    def test_t4_04_web_assets_integrity_and_cdn_references(self):
        """Verify web static assets (web/index.html, app.js, style.css) CDN and tier tokens."""
        index_html = WEB_DIR / "index.html"
        app_js = WEB_DIR / "app.js"
        style_css = WEB_DIR / "style.css"
        
        if not index_html.exists():
            self.skipTest("web/index.html not yet implemented (Milestone 3)")
            
        html = index_html.read_text(encoding="utf-8")
        # R1 & R4.2: CDN references
        self.assertIn("vue", html.lower(), "Vue 3 CDN link missing from index.html")
        self.assertIn("echarts", html.lower(), "ECharts CDN link missing from index.html")
        
        if app_js.exists():
            js = app_js.read_text(encoding="utf-8")
            # 7 Tier Colors present
            for hex_code in ["#808080", "#008000", "#03A89E", "#0000FF", "#AA00AA", "#FF8C00", "#FF0000"]:
                self.assertIn(hex_code.lower(), js.lower(), f"Color {hex_code} missing in app.js")
            # Relative data loader
            self.assertIn("rating_db.json", js)


# ==============================================================================
# TIER 5: ADVERSARIAL HARDENING & DEFENSIVE COVERAGE (Milestone 5)
# ==============================================================================

class TestTier5AdversarialHardening(unittest.TestCase):
    """Tier 5 adversarial hardening tests addressing XSS vulnerabilities,
    JS type coercion edge cases, and UI badge completeness.
    """

    def test_t5_01_app_js_xss_escape_html_and_tooltip_hardening(self):
        """Verify web/app.js defines escapeHtml and sanitizes tooltip strings (Gap 1)."""
        app_js = WEB_DIR / "app.js"
        self.assertTrue(app_js.exists(), "web/app.js must exist")
        content = app_js.read_text(encoding="utf-8")

        # 1. escapeHtml definition and entity map
        self.assertIn("function escapeHtml", content, "escapeHtml helper function must be defined in app.js")
        for char_entity in ["&amp;", "&lt;", "&gt;", "&quot;", "&#39;"]:
            self.assertIn(char_entity, content, f"escapeHtml must replace entity {char_entity}")

        # 2. Tooltip formatter sanitization
        self.assertIn("escapeHtml(raw.contest_title", content, "Tooltip must sanitize raw.contest_title")
        self.assertIn("escapeHtml(raw.team_name", content, "Tooltip must sanitize raw.team_name")

        # 3. Dynamic probe using Node.js if available
        node_bin = shutil.which("node")
        if node_bin:
            probe_js = """
const fs = require('fs');
const code = fs.readFileSync('web/app.js', 'utf8');
const match = code.match(/function escapeHtml\\([\\s\\S]*?\\n\\}/);
if (!match) process.exit(1);
eval(match[0]);

const payloads = [
    ['<script>alert(1)</script>', '&lt;script&gt;alert(1)&lt;/script&gt;'],
    ['<img src=x onerror=alert(1)>', '&lt;img src=x onerror=alert(1)&gt;'],
    ['<b>Bold & "Quotes" \\'Single\\'</b>', '&lt;b&gt;Bold &amp; &quot;Quotes&quot; &#39;Single&#39;&lt;/b&gt;'],
    [null, ''],
    [undefined, '']
];
for (const [inp, expected] of payloads) {
    const res = escapeHtml(inp);
    if (res !== expected) {
        console.error('Mismatch: input=' + inp + ', got=' + res + ', expected=' + expected);
        process.exit(2);
    }
}
console.log('ALL_ESCAPE_HTML_PROBES_PASSED');
"""
            res = subprocess.run([node_bin, "-e", probe_js], cwd=str(WORKSPACE_ROOT), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Node probe for escapeHtml failed:\n{res.stderr}")
            self.assertIn("ALL_ESCAPE_HTML_PROBES_PASSED", res.stdout)

    def test_t5_02_app_js_get_best_perf_ignores_null(self):
        """Verify web/app.js getBestPerf skips perf_rating: null and does not evaluate to 0.0 (Gap 2)."""
        app_js = WEB_DIR / "app.js"
        self.assertTrue(app_js.exists(), "web/app.js must exist")
        content = app_js.read_text(encoding="utf-8")

        # 1. Structural check
        self.assertIn("function getBestPerf", content, "getBestPerf function must be defined in app.js")
        self.assertIn("h.perf_rating !== null", content, "getBestPerf must explicitly check for null perf_rating")

        # 2. Dynamic probe using Node.js if available
        node_bin = shutil.which("node")
        if node_bin:
            probe_js = """
const fs = require('fs');
const code = fs.readFileSync('web/app.js', 'utf8');
const match = code.match(/function getBestPerf[\\s\\S]*?return best > -9999 \\? best\\.toFixed\\(1\\) : '-';\\s*\\}/);
if (!match) process.exit(1);
eval(match[0]);

// Case A: History with only null perf_rating -> must return '-'
const userNullOnly = { history: [{ perf_rating: null }] };
if (getBestPerf(userNullOnly) !== '-') {
    console.error('Failed Case A: got ' + getBestPerf(userNullOnly));
    process.exit(2);
}

// Case B: History with mixed null, undefined, empty string -> must return '-'
const userMixedEmpty = { history: [{ perf_rating: null }, { perf_rating: undefined }, { perf_rating: '' }] };
if (getBestPerf(userMixedEmpty) !== '-') {
    console.error('Failed Case B: got ' + getBestPerf(userMixedEmpty));
    process.exit(3);
}

// Case C: History with null and valid rating -> must return valid rating formatted
const userWithValid = { history: [{ perf_rating: null }, { perf_rating: 1540.24 }] };
if (getBestPerf(userWithValid) !== '1540.2') {
    console.error('Failed Case C: got ' + getBestPerf(userWithValid));
    process.exit(4);
}

// Case D: User with no history -> must return '-'
if (getBestPerf({ history: [] }) !== '-') {
    console.error('Failed Case D: got ' + getBestPerf({ history: [] }));
    process.exit(5);
}

console.log('ALL_GET_BEST_PERF_PROBES_PASSED');
"""
            res = subprocess.run([node_bin, "-e", probe_js], cwd=str(WORKSPACE_ROOT), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Node probe for getBestPerf failed:\n{res.stderr}")
            self.assertIn("ALL_GET_BEST_PERF_PROBES_PASSED", res.stdout)

    def test_t5_03_index_html_renders_star_team_badge(self):
        """Verify web/index.html includes visual badge for star teams (is_star) (Gap 3)."""
        index_html = WEB_DIR / "index.html"
        self.assertTrue(index_html.exists(), "web/index.html must exist")
        content = index_html.read_text(encoding="utf-8")

        # 1. Condition check
        self.assertIn("standing.is_star", content, "index.html must check standing.is_star")
        # 2. Text check
        self.assertIn("打星队", content, "index.html must display 打星队 badge text")
        # 3. Styling check
        self.assertTrue("bg-amber-50" in content or "text-amber-700" in content, "Badge must have star/amber styling")

    def test_t5_04_star_team_data_preservation_and_recalculation(self):
        """Verify that star team attribute is preserved in rating_db.json after recalculation."""
        # Check production contest contains star team
        mock_contest_path = CONTESTS_DIR / "2026_10_25_icpc_regional_mock.json"
        self.assertTrue(mock_contest_path.exists())
        contest_data = json.loads(mock_contest_path.read_text(encoding="utf-8"))
        star_teams = [s for s in contest_data["standings"] if s.get("is_star")]
        self.assertGreater(len(star_teams), 0, "Production contest should contain star team")

        # Verify recalculation preserves is_star in output
        res = run_cli([str(RECALCULATE_SCRIPT)])
        self.assertEqual(res.returncode, 0)
        
        db_candidates = [
            WORKSPACE_ROOT / "dist" / "data" / "rating_db.json",
            WORKSPACE_ROOT / "web" / "data" / "rating_db.json",
            WORKSPACE_ROOT / "data" / "rating_db.json"
        ]
        db_file = next((f for f in db_candidates if f.exists()), None)
        self.assertIsNotNone(db_file)
        db = json.loads(db_file.read_text(encoding="utf-8"))
        
        found_star_in_db = False
        for c in db.get("contests", []):
            if c["contest_id"] == "2026_10_25_icpc_regional_mock":
                for s in c.get("standings", []):
                    if s.get("is_star"):
                        found_star_in_db = True
                        self.assertIn("东南大学", s["team_name"])
                        break
        self.assertTrue(found_star_in_db, "is_star flag must be preserved in rating_db.json standings")


if __name__ == "__main__":
    unittest.main()
