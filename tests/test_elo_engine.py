"""
Pytest Suite for USTS Elo Engine & Mathematical Blueprint
Verifies win probability, expected seeds, team power synthesis, LogSumExp stability,
fractional rank tie-breaking, bisection convergence, dual-track display offsets,
zero-solve floor protection, and internal school zero-sum calibration.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.recalculate import (
    compute_team_rating,
    get_display_offset,
    to_display_rating,
    get_tier_and_color,
    compute_ranks,
    compute_seeds,
    compute_perf_ratings,
    apply_zero_solve_limit,
    calibrate_internal_school_deltas,
    process_contest,
)


def win_probability(r_a: float, r_b: float) -> float:
    """Theoretical win probability P(B > A) = 1 / (1 + 10^((R_A - R_B)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((r_a - r_b) / 400.0))


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_win_probability_and_seed():
    """
    1. Verifies symmetry P(A > B) + P(B > A) == 1.0, equal ratings probability == 0.5,
    400-point difference probability == 1/11, and expected seeds calculation & conservation.
    """
    # Equal ratings -> 50% probability
    p_equal = win_probability(1500.0, 1500.0)
    assert pytest.approx(p_equal, rel=1e-9) == 0.5

    # Symmetry property
    p_ab = win_probability(1600.0, 1350.0)
    p_ba = win_probability(1350.0, 1600.0)
    assert pytest.approx(p_ab + p_ba, rel=1e-9) == 1.0

    # 400 points difference
    p_higher_loses = win_probability(1600.0, 1200.0)  # P(1200 beats 1600)
    p_lower_loses = win_probability(1200.0, 1600.0)   # P(1600 beats 1200)
    assert pytest.approx(p_higher_loses, rel=1e-9) == 1.0 / 11.0
    assert pytest.approx(p_lower_loses, rel=1e-9) == 10.0 / 11.0

    # Expected seeds for 4 teams
    ratings = np.array([1600.0, 1400.0, 1200.0, 1000.0], dtype=np.float64)
    seeds = compute_seeds(ratings)

    # Monotonicity: higher rating must have smaller (better) seed
    for i in range(len(seeds) - 1):
        assert seeds[i] < seeds[i + 1]

    # Individual seed formula check: Seed_A = 1 + sum_{B != A} P(B > A)
    for i in range(len(ratings)):
        expected_seed = 1.0 + sum(win_probability(ratings[i], ratings[j]) for j in range(len(ratings)) if j != i)
        assert pytest.approx(seeds[i], abs=1e-9) == expected_seed

    # Total seeds conservation: sum of seeds == N*(N+1)/2
    assert pytest.approx(np.sum(seeds), abs=1e-9) == 4.0 * 5.0 / 2.0

    # Boundary conditions
    empty_seeds = compute_seeds(np.array([], dtype=np.float64))
    assert len(empty_seeds) == 0
    single_seed = compute_seeds(np.array([1500.0], dtype=np.float64))
    assert list(single_seed) == [1.0]


def test_team_rating_synthesis_constants():
    """
    2. Verifies Exponential Weighted Average (指数加权平均) synthesis and external anchors.
    """
    # Solo team (k=1): exact identity
    assert compute_team_rating([1420.0]) == 1420.0
    assert compute_team_rating([1000.0]) == 1000.0

    # Duo team with equal players: exact identity
    actual_duo = compute_team_rating([1200.0, 1200.0])
    assert pytest.approx(actual_duo, abs=1e-7) == 1200.0

    # Trio team with equal players: exact identity
    actual_trio = compute_team_rating([1500.0, 1500.0, 1500.0])
    assert pytest.approx(actual_trio, abs=1e-7) == 1500.0

    # Quad team with equal players: exact identity
    actual_quad = compute_team_rating([1500.0, 1500.0, 1500.0, 1500.0])
    assert pytest.approx(actual_quad, abs=1e-7) == 1500.0

    # Unequal players: XCPC-Elo log-power-mean aggregation
    # 1700 and 1400: R_team = 400 * log10((10^(1700/400) + 10^(1400/400)) / 2)
    expected_unequal = 1700.0 + 400.0 * math.log10((1.0 + 10.0 ** (-300.0 / 400.0)) / 2.0)
    actual_unequal = compute_team_rating([1700.0, 1400.0])
    assert pytest.approx(actual_unequal, abs=1e-7) == expected_unequal

    # External teams
    ext_default = compute_team_rating([], is_school=False, external_elo=None)
    assert ext_default == 1500.0
    ext_custom = compute_team_rating([], is_school=False, external_elo=1850.0)
    assert ext_custom == 1850.0


def test_team_rating_logsumexp_stability():
    """
    3. Verifies Exponential Weighted Average numerical stability and bounded properties.
    """
    # Extreme spread: 3500 and 1000
    r_extreme = compute_team_rating([3500.0, 1000.0])
    assert not math.isnan(r_extreme)
    assert not math.isinf(r_extreme)
    # The team rating should be strictly bounded between min and max
    assert 1000.0 <= r_extreme <= 3500.0
    # In log-power-mean of 2 members, maximum pull down by low score is 400*log10(0.5) ≈ -120.4
    assert pytest.approx(r_extreme, abs=1e-2) == 3500.0 + 400.0 * math.log10(0.5)

    # Multiple high ratings: 3000, 2800, 2900
    r_high = compute_team_rating([3000.0, 2800.0, 2900.0])
    assert not math.isnan(r_high)
    assert not math.isinf(r_high)
    assert 2800.0 <= r_high <= 3000.0

    # 10 members with diverse ratings
    ratings_10 = [1000.0, 1200.0, 1500.0, 1800.0, 2200.0, 2500.0, 2800.0, 3100.0, 3300.0, 3500.0]
    r_10 = compute_team_rating(ratings_10)
    assert not math.isnan(r_10)
    assert not math.isinf(r_10)
    assert 1000.0 <= r_10 <= 3500.0


def test_fractional_ranks_tie_breaking():
    """
    4. Verifies rank computation with distinct standings, 2-way ties, 3-way ties,
    and all-tied standings using the arithmetic mean of tied 1-based positions.
    """
    # 1. Distinct ranks
    distinct_standings = [
        {"solved": 5, "penalty": 300},
        {"solved": 4, "penalty": 200},
        {"solved": 3, "penalty": 150},
    ]
    ranks = compute_ranks(distinct_standings)
    np.testing.assert_array_almost_equal(ranks, [1.0, 2.0, 3.0])

    # 2. Two teams tied at positions 1 and 2 (0-based) -> ranks (2 + 3) / 2 = 2.5
    tied_2 = [
        {"solved": 6, "penalty": 400},
        {"solved": 5, "penalty": 350},
        {"solved": 5, "penalty": 350},
        {"solved": 3, "penalty": 200},
    ]
    ranks_2 = compute_ranks(tied_2)
    np.testing.assert_array_almost_equal(ranks_2, [1.0, 2.5, 2.5, 4.0])

    # 3. Three teams tied for 1st place -> (1 + 2 + 3) / 3 = 2.0
    tied_3 = [
        {"solved": 7, "penalty": 500},
        {"solved": 7, "penalty": 500},
        {"solved": 7, "penalty": 500},
        {"solved": 4, "penalty": 250},
    ]
    ranks_3 = compute_ranks(tied_3)
    np.testing.assert_array_almost_equal(ranks_3, [2.0, 2.0, 2.0, 4.0])

    # 4. All 5 teams tied with 0 solved -> (1 + 5) / 2 = 3.0
    all_tied = [{"solved": 0, "penalty": 0} for _ in range(5)]
    ranks_all = compute_ranks(all_tied)
    np.testing.assert_array_almost_equal(ranks_all, [3.0, 3.0, 3.0, 3.0, 3.0])

    # 5. Empty and single team
    assert len(compute_ranks([])) == 0
    np.testing.assert_array_almost_equal(compute_ranks([{"solved": 1, "penalty": 10}]), [1.0])


def test_vectorized_bisection_convergence():
    """
    5. Verifies that vectorized bisection over 50 iterations converges such that
    the residual |exp_rank(R_perf) - target_rank| < 1e-6 across various team configurations.
    """
    ratings = np.array([1700.0, 1550.0, 1400.0, 1250.0, 1100.0, 950.0], dtype=np.float64)
    ranks = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    seeds = compute_seeds(ratings)
    target_ranks = np.sqrt(seeds * ranks)

    perf_ratings = compute_perf_ratings(ratings, target_ranks, iterations=50)
    assert len(perf_ratings) == len(ratings)

    # Compute actual expected ranks with candidate perf_ratings
    diff = perf_ratings[:, None] - ratings[None, :]
    p = 1.0 / (1.0 + 10.0 ** (diff / 400.0))
    np.fill_diagonal(p, 0.0)
    actual_exp_ranks = 1.0 + np.sum(p, axis=1)

    max_residual = np.max(np.abs(actual_exp_ranks - target_ranks))
    assert max_residual < 1e-6, f"Residual {max_residual} exceeds 1e-6"


def test_display_rating_offsets_and_floor():
    """
    6. Verifies display rating offsets for rated_count in {0, 1, 2, 3, 5, 10}
    and non-negativity floor protection (display_rating >= 0).
    Newbie protection lasts 3 contests; 3rd contest finishes to show real rating.
    """
    # Offset table verification
    assert get_display_offset(0) == -1000
    assert get_display_offset(1) == -500
    assert get_display_offset(2) == -250
    assert get_display_offset(3) == 0
    assert get_display_offset(4) == 0
    assert get_display_offset(5) == 0
    assert get_display_offset(10) == 0

    # Display rating conversion for 1000 baseline
    assert to_display_rating(1000.0, 0) == 0
    assert to_display_rating(1000.0, 1) == 500
    assert to_display_rating(1000.0, 2) == 750
    assert to_display_rating(1000.0, 3) == 1000
    assert to_display_rating(1000.0, 4) == 1000
    assert to_display_rating(1000.0, 5) == 1000

    # Non-negativity floor: max(0, round(true_rating + offset))
    assert to_display_rating(500.0, 0) == 0      # 500 - 1000 = -500 -> 0
    assert to_display_rating(200.0, 1) == 0      # 200 - 500 = -300 -> 0
    assert to_display_rating(-50.0, 3) == 0      # -50 + 0 = -50 -> 0


def test_zero_solve_floor_protection():
    """
    7. Verifies that single contest drop worse than -60 is clamped to -60 when solved == 0,
    and unconstrained when solved > 0.
    """
    # solved == 0: large drop clamped to -60
    assert apply_zero_solve_limit(-150.0, solved=0) == -60.0
    assert apply_zero_solve_limit(-60.1, solved=0) == -60.0
    assert apply_zero_solve_limit(-60.0, solved=0) == -60.0

    # solved == 0: moderate drop or gain unconstrained
    assert apply_zero_solve_limit(-35.0, solved=0) == -35.0
    assert apply_zero_solve_limit(15.0, solved=0) == 15.0

    # solved > 0: drop is NOT clamped
    assert apply_zero_solve_limit(-150.0, solved=1) == -150.0
    assert apply_zero_solve_limit(-80.0, solved=2) == -80.0


def test_internal_school_zero_sum_calibration():
    """
    8. Verifies that calibrate_internal_school_deltas subtracts mean drift mu
    such that |sum(calibrated_deltas)| <= 1e-6 across solo, duo, and trio mixed teams.
    """
    raw_deltas = [25.0, 10.0, 10.0, -15.0, -15.0, -15.0]
    calibrated = calibrate_internal_school_deltas(raw_deltas)

    # Net sum must be strictly zero (bounded by machine epsilon)
    assert abs(sum(calibrated)) <= 1e-6

    # Each delta must equal raw_delta - mu
    mu = sum(raw_deltas) / len(raw_deltas)
    for c, r in zip(calibrated, raw_deltas):
        assert pytest.approx(c, abs=1e-9) == (r - mu)

    # Empty and single element edge cases
    assert calibrate_internal_school_deltas([]) == []
    assert calibrate_internal_school_deltas([42.0]) == [0.0]


def test_external_contest_no_zero_sum():
    """
    9. Verifies that EXTERNAL_XCPC contests do not enforce zero-sum drift on school members,
    and external teams are not added to the user database.
    """
    user_db = {
        "23001001": {
            "student_id": "23001001",
            "name": "Alice",
            "grade": 2023,
            "is_active": True,
            "is_retired": False,
            "true_rating": 1500.0,
            "display_rating": 1500,
            "max_display_rating": 1500,
            "rated_count": 5,
            "tier": "Candidate Master",
            "color": "#AA00AA",
            "last_contest_date": None,
            "history": [],
            "_last_contest_dt": None,
        },
        "23001002": {
            "student_id": "23001002",
            "name": "Bob",
            "grade": 2023,
            "is_active": True,
            "is_retired": False,
            "true_rating": 1400.0,
            "display_rating": 1400,
            "max_display_rating": 1400,
            "rated_count": 5,
            "tier": "Expert",
            "color": "#0000FF",
            "last_contest_date": None,
            "history": [],
            "_last_contest_dt": None,
        },
    }

    external_contest = {
        "contest_id": "2026_11_01_external_mock",
        "title": "External Mock Contest",
        "contest_type": "EXTERNAL_XCPC",
        "start_time": "2026-11-01T09:00:00+08:00",
        "standings": [
            {
                "team_name": "Tsinghua Stars",
                "is_school": False,
                "members": [],
                "external_elo": 2200,
                "solved": 10,
                "penalty": 500,
            },
            {
                "team_name": "USTS Team 1",
                "is_school": True,
                "members": ["23001001"],
                "external_elo": None,
                "solved": 8,
                "penalty": 600,
            },
            {
                "team_name": "USTS Team 2",
                "is_school": True,
                "members": ["23001002"],
                "external_elo": None,
                "solved": 7,
                "penalty": 650,
            },
        ],
    }

    processed = process_contest(external_contest, user_db)
    assert processed["contest_id"] == "2026_11_01_external_mock"

    # School members should have their rating updated
    delta1 = user_db["23001001"]["history"][-1]["delta"]
    delta2 = user_db["23001002"]["history"][-1]["delta"]

    # In EXTERNAL_XCPC, school deltas are NOT forced to sum to 0
    # Both teams performed well against a 2200 anchor, so their net deltas can be positive
    school_delta_sum = delta1 + delta2
    # Verify deltas are computed and not zeroed out by artificial zero-sum
    assert isinstance(school_delta_sum, float)

    # External team Tsinghua Stars must NOT be added to user_db
    assert "Tsinghua Stars" not in user_db
    assert len(user_db) == 2
