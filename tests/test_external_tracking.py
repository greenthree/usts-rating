"""
tests/test_external_tracking.py - Pytest Suite for External Team Elo Evolution & XCPC-Elo Sync.

Verifies:
  1. Organization and player Elo extraction from XCPC-Elo data payload.
  2. USTS historical member filtering and ranking.
  3. Continuous external team rating evolution across consecutive contests in recalculate.py.
  4. Automatic fallback and seeding from external_elo_map.json.
  5. Multi-contest batch importing from SRK ranklists.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sync_xcpc_elo import (
    extract_organization_elo,
    extract_player_elo,
    extract_usts_members,
)
from scripts.recalculate import (
    process_contest,
    recalculate_all,
)
from scripts.import_srk_batch import slugify
from scripts.validate_data import validate_contest_file


@pytest.fixture
def sample_xcpc_data() -> dict:
    """Mock XCPC-Elo payload with multiple universities and USTS contestants."""
    return {
        "generatedAt": "2026-11-15T12:00:00.000Z",
        "players": [
            {
                "id": "p1",
                "organization": "清华大学",
                "name": "清华选手A",
                "history": [[1, 1, 100, 3100]],
            },
            {
                "id": "p2",
                "organization": "清华大学",
                "name": "清华选手B",
                "history": [[1, 1, 100, 3050]],
            },
            {
                "id": "p3",
                "organization": "清华大学",
                "name": "清华选手C",
                "history": [[1, 1, 100, 3000]],
            },
            {
                "id": "p4",
                "organization": "清华大学",
                "name": "清华选手D",
                "history": [[1, 1, 50, 2800]],
            },
            {
                "id": "p5",
                "organization": "北京大学",
                "name": "北大选手A",
                "history": [[1, 2, 80, 2950]],
            },
            {
                "id": "p6",
                "organization": "苏州科技大学",
                "name": "苏科选手A",
                "history": [[1, 20, 20, 1480], [2, 18, 15, 1495]],
            },
            {
                "id": "p7",
                "organization": "苏州科技大学",
                "name": "苏科选手B",
                "history": [[1, 20, 10, 1420]],
            },
        ],
    }


def test_xcpc_extraction_organization_and_players(sample_xcpc_data: dict):
    """Verifies organization baseline (top-3 mean) and player mapping."""
    org_map = extract_organization_elo(sample_xcpc_data, top_k=3)
    # Tsinghua top 3: 3100, 3050, 3000 -> mean = 3050
    assert org_map["清华大学"] == 3050
    # Peking top 1: 2950
    assert org_map["北京大学"] == 2950
    # USTS top 2: 1495, 1420 -> mean = 1458
    assert org_map["苏州科技大学"] == 1458

    player_map = extract_player_elo(sample_xcpc_data)
    assert player_map["清华选手A"] == 3100
    assert player_map["清华大学::清华选手A"] == 3100
    assert player_map["苏科选手A"] == 1495


def test_xcpc_extraction_usts_members(sample_xcpc_data: dict):
    """Verifies filtering of USTS members from dataset."""
    usts_members = extract_usts_members(sample_xcpc_data)
    assert len(usts_members) == 2
    # Highest rated first
    assert usts_members[0]["name"] == "苏科选手A"
    assert usts_members[0]["latest_elo"] == 1495
    assert usts_members[0]["contest_count"] == 2
    assert usts_members[1]["name"] == "苏科选手B"
    assert usts_members[1]["latest_elo"] == 1420


def test_external_team_continuous_evolution():
    """
    Verifies that external teams carry their ratings forward across consecutive contests
    in external_db, matching the user's vision of a continuous multi-contest replay.
    """
    user_db = {
        "2023001": {
            "student_id": "2023001",
            "name": "张三",
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
        }
    }

    external_db: dict[str, dict] = {}
    external_elo_map = {"清华大学": 3100.0, "北京大学": 3000.0}

    # Contest 1: Tsinghua enters with no explicit external_elo, seeded from external_elo_map
    contest_1 = {
        "contest_id": "2026_10_01_xcpc_1",
        "title": "National Contest #1",
        "contest_type": "EXTERNAL_XCPC",
        "start_time": "2026-10-01T09:00:00+08:00",
        "standings": [
            {
                "team_name": "清华大学 - 战神队",
                "is_school": False,
                "solved": 11,
                "penalty": 500,
            },
            {
                "team_name": "北京大学 - 始终如一",
                "is_school": False,
                "solved": 10,
                "penalty": 550,
            },
            {
                "team_name": "苏州科技大学 - 一队",
                "is_school": True,
                "members": ["2023001"],
                "solved": 6,
                "penalty": 700,
            },
        ],
    }

    p1 = process_contest(contest_1, user_db, external_db=external_db, external_elo_map=external_elo_map)
    assert p1["contest_id"] == "2026_10_01_xcpc_1"

    # External teams must now be recorded in external_db
    assert "清华大学 - 战神队" in external_db
    assert "北京大学 - 始终如一" in external_db
    tsinghua_r1 = external_db["清华大学 - 战神队"]["rating"]
    pku_r1 = external_db["北京大学 - 始终如一"]["rating"]
    assert external_db["清华大学 - 战神队"]["rated_count"] == 1

    # Since Tsinghua finished 1st, their rating should be maintained or increased
    assert tsinghua_r1 > 3000.0

    # Contest 2: 2 weeks later, Tsinghua and PKU compete again with NO explicit external_elo in JSON!
    contest_2 = {
        "contest_id": "2026_10_15_xcpc_2",
        "title": "National Contest #2",
        "contest_type": "EXTERNAL_XCPC",
        "start_time": "2026-10-15T09:00:00+08:00",
        "standings": [
            {
                "team_name": "清华大学 - 战神队",
                "is_school": False,
                "solved": 12,
                "penalty": 480,
            },
            {
                "team_name": "北京大学 - 始终如一",
                "is_school": False,
                "solved": 10,
                "penalty": 520,
            },
            {
                "team_name": "苏州科技大学 - 一队",
                "is_school": True,
                "members": ["2023001"],
                "solved": 7,
                "penalty": 680,
            },
        ],
    }

    p2 = process_contest(contest_2, user_db, external_db=external_db, external_elo_map=external_elo_map)

    # In Contest 2 standings, Tsinghua's starting team_rating must equal tsinghua_r1 (from contest 1)!
    tsinghua_c2_standing = [t for t in p2["standings"] if t["team_name"] == "清华大学 - 战神队"][0]
    assert pytest.approx(tsinghua_c2_standing["team_rating"], abs=0.2) == tsinghua_r1

    # In external_db, rated_count is now 2
    assert external_db["清华大学 - 战神队"]["rated_count"] == 2
    tsinghua_r2 = external_db["清华大学 - 战神队"]["rating"]
    assert tsinghua_r2 >= tsinghua_r1  # Finished 1st again, evolved further!


def test_batch_srk_slugify():
    """Verifies slugify utility produces clean filenames."""
    assert slugify("2024 ICPC Asia East Continent Final") == "2024_icpc_asia_east_continent_final"
    assert slugify("赛区一") != slugify("赛区二")
    assert slugify("  test / slash & symbols  ") == "test_slash_symbols"


def test_batch_srk_cli_only_overwrites_when_requested(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_srk_batch.py"
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "contests"
    scoreboard = source / "Round-01.srk.json"
    contest = {
        "contest": {"startAt": "2026-11-19T08:00:00+08:00"},
        "rows": [{"user": {"name": "Example Team"}, "score": {"value": 1}}],
    }
    scoreboard.write_text(json.dumps(contest), encoding="utf-8")
    command = [sys.executable, str(script), str(source), "--output-dir", str(output)]

    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    target = output / "2026_11_19_round_01.json"
    original = target.read_bytes()
    assert not validate_contest_file(target)

    contest["rows"][0]["score"]["value"] = 2
    scoreboard.write_text(json.dumps(contest), encoding="utf-8")
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert target.read_bytes() == original
    assert list(output.iterdir()) == [target]

    forced = subprocess.run([*command, "--overwrite"], cwd=tmp_path, capture_output=True, text=True)
    assert forced.returncode == 0, forced.stderr
    assert json.loads(target.read_bytes())["standings"][0]["solved"] == 2

    contest["rows"][0]["user"]["organization"] = "USTS"
    scoreboard.write_text(json.dumps(contest), encoding="utf-8")
    invalid = subprocess.run([*command, "--overwrite"], cwd=tmp_path, capture_output=True, text=True)
    assert invalid.returncode == 1
    assert "no participant names or IDs" in invalid.stderr
    assert json.loads(target.read_bytes())["standings"][0]["solved"] == 2
