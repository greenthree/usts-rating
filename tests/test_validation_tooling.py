#!/usr/bin/env python3
"""
tests/test_validation_tooling.py - Automated tests for validate_data.py and import_contest.py.

Verifies:
1. Positive validation on data/users.csv and data/contests/*.
2. 15 negative validation checks (error detection and non-zero exit codes).
3. Multi-format contest import (CSV, DOMjudge, PTA).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_data import (
    main as validate_main,
    parse_bool,
    validate_all,
    validate_contest_dict,
    validate_contest_file,
    validate_users_csv,
)
from scripts.import_contest import import_contest
from scripts.recalculate import load_contests, recalculate_all

SAMPLE_USERS = Path(__file__).resolve().parent / "fixtures" / "sample_users.csv"


class TestDataValidationPositive(unittest.TestCase):
    """Test standard production data files."""

    def test_production_users_and_contests(self):
        """Standard production users.csv and 3 contests must pass with zero errors."""
        is_valid, errors = validate_all("data/users.csv", "data/contests")
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_cli_exit_code_zero(self):
        """CLI main() must return 0 on valid data."""
        exit_code = validate_main(["--users", "data/users.csv", "--contests", "data/contests"])
        self.assertEqual(exit_code, 0)

    def test_contest_file_with_utf8_bom(self):
        """UTF-8 BOM must be accepted by both validation and recalculation."""
        temp_dir = tempfile.mkdtemp()
        try:
            bom_json = Path(temp_dir) / "2026_09_15_school_solo_1.json"
            contest_data = {
                "contest_id": "2026_09_15_school_solo_1",
                "title": "BOM Test Contest",
                "contest_type": "INTERNAL_SCHOOL",
                "start_time": "2026-09-15T13:00:00+08:00",
                "standings": [
                    {
                        "team_name": "T1",
                        "is_school": True,
                        "members": ["23001003"],
                        "solved": 1,
                        "penalty": 10,
                    }
                ],
            }
            # Write with utf-8-sig encoding (adds BOM: 0xEF, 0xBB, 0xBF)
            bom_json.write_text(json.dumps(contest_data, ensure_ascii=False), encoding="utf-8-sig")
            errors = validate_contest_file(bom_json, valid_student_ids={"23001003"})
            self.assertEqual(errors, [])
            self.assertEqual(load_contests(temp_dir)[0]["standings"], contest_data["standings"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_parse_bool_chinese_and_aliases(self):
        """Test parse_bool supports Chinese boolean characters and common aliases."""
        self.assertTrue(parse_bool("是"))
        self.assertTrue(parse_bool("是 "))
        self.assertTrue(parse_bool("true"))
        self.assertTrue(parse_bool("TRUE"))
        self.assertTrue(parse_bool("yes"))
        self.assertTrue(parse_bool("y"))
        self.assertTrue(parse_bool("t"))
        self.assertTrue(parse_bool("1"))
        self.assertTrue(parse_bool(1))
        self.assertTrue(parse_bool(True))

        self.assertFalse(parse_bool("否"))
        self.assertFalse(parse_bool("否 "))
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("FALSE"))
        self.assertFalse(parse_bool("no"))
        self.assertFalse(parse_bool("n"))
        self.assertFalse(parse_bool("f"))
        self.assertFalse(parse_bool("0"))
        self.assertFalse(parse_bool(0))
        self.assertFalse(parse_bool(False))

        self.assertIsNone(parse_bool("maybe"))
        self.assertIsNone(parse_bool("不确定"))
        self.assertIsNone(parse_bool(None))
        self.assertIsNone(parse_bool(42))


class TestDataValidationNegative(unittest.TestCase):
    """Test all negative cases to ensure strict enforcement."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dir_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_file_existence_users(self):
        """Rule 1: Non-existent users file triggers FILE_EXISTENCE error."""
        _, errors = validate_users_csv(self.dir_path / "nonexistent.csv")
        self.assertTrue(any("FILE_EXISTENCE" in e for e in errors))

    def test_header_match(self):
        """Rule 2: Mismatched CSV header triggers HEADER_MATCH error."""
        csv_file = self.dir_path / "bad_header.csv"
        csv_file.write_text("id,name,grade,elo,active\n", encoding="utf-8")
        _, errors = validate_users_csv(csv_file)
        self.assertTrue(any("HEADER_MATCH" in e for e in errors))

    def test_duplicate_student_id(self):
        """Rule 3: Duplicate student_id in users CSV triggers STUDENT_ID_UNIQUE error."""
        csv_file = self.dir_path / "dup_user.csv"
        content = (
            "student_id,name,grade,initial_external_elo,is_active\n"
            "23001001,张三,2023,,true\n"
            "23001001,李四,2023,,true\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        _, errors = validate_users_csv(csv_file)
        self.assertTrue(any("STUDENT_ID_UNIQUE" in e for e in errors))

    def test_student_id_format(self):
        """Rule 4: Malformed student_id triggers STUDENT_ID_FORMAT error."""
        csv_file = self.dir_path / "bad_id.csv"
        content = (
            "student_id,name,grade,initial_external_elo,is_active\n"
            "ab!,张三,2023,,true\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        _, errors = validate_users_csv(csv_file)
        self.assertTrue(any("STUDENT_ID_FORMAT" in e for e in errors))

    def test_grade_out_of_range(self):
        """Rule 6: Grade out of range [2010, 2035] triggers GRADE_VALID error."""
        csv_file = self.dir_path / "bad_grade.csv"
        content = (
            "student_id,name,grade,initial_external_elo,is_active\n"
            "23001001,张三,1998,,true\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        _, errors = validate_users_csv(csv_file)
        self.assertTrue(any("GRADE_VALID" in e for e in errors))

    def test_elo_out_of_range(self):
        """Rule 7: ELO out of range [100, 3500] triggers ELO_VALID error."""
        csv_file = self.dir_path / "bad_elo.csv"
        content = (
            "student_id,name,grade,initial_external_elo,is_active\n"
            "23001001,张三,2023,4500,true\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        _, errors = validate_users_csv(csv_file)
        self.assertTrue(any("ELO_VALID" in e for e in errors))

    def test_is_active_invalid(self):
        """Rule 8: is_active not boolean triggers IS_ACTIVE_VALID error."""
        csv_file = self.dir_path / "bad_active.csv"
        content = (
            "student_id,name,grade,initial_external_elo,is_active\n"
            "23001001,张三,2023,,maybe\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        _, errors = validate_users_csv(csv_file)
        self.assertTrue(any("IS_ACTIVE_VALID" in e for e in errors))

    def test_empty_contests_dir(self):
        """Rule 9: Empty contests directory triggers DIR_NOT_EMPTY error."""
        empty_contests = self.dir_path / "empty_contests"
        empty_contests.mkdir()
        is_valid, errors = validate_all("data/users.csv", empty_contests)
        self.assertFalse(is_valid)
        self.assertTrue(any("DIR_NOT_EMPTY" in e for e in errors))

    def test_contest_filename_format(self):
        """Rule 10: Filename not matching convention triggers FILENAME_FORMAT error."""
        bad_json = self.dir_path / "invalid_name.json"
        bad_json.write_text("{}", encoding="utf-8")
        errors = validate_contest_file(bad_json)
        self.assertTrue(any("FILENAME_FORMAT" in e for e in errors))

    def test_date_consistency(self):
        """Rule 11: Filename date mismatch triggers DATE_CONSISTENCY error."""
        mismatch_json = self.dir_path / "2026_09_15_test.json"
        contest_data = {
            "contest_id": "2026_09_15_test",
            "title": "Date Mismatch Test",
            "contest_type": "INTERNAL_SCHOOL",
            "start_time": "2026-10-20T13:00:00+08:00",
            "standings": [
                {
                    "team_name": "T1",
                    "is_school": True,
                    "members": ["23001003"],
                    "solved": 1,
                    "penalty": 10,
                }
            ],
        }
        mismatch_json.write_text(json.dumps(contest_data), encoding="utf-8")
        errors = validate_contest_file(mismatch_json, valid_student_ids={"23001003"})
        self.assertTrue(any("DATE_CONSISTENCY" in e for e in errors))

    def test_contest_type_enum(self):
        """Rule 13: Invalid contest_type triggers TYPE_ENUM error."""
        contest_data = {
            "contest_id": "test",
            "title": "Invalid Type Test",
            "contest_type": "CF_ROUND",
            "start_time": "2026-09-15T13:00:00+08:00",
            "standings": [
                {
                    "team_name": "T1",
                    "is_school": True,
                    "members": ["23001003"],
                    "solved": 1,
                    "penalty": 10,
                }
            ],
        }
        errors = validate_contest_dict(contest_data, valid_student_ids={"23001003"})
        self.assertTrue(any("TYPE_ENUM" in e for e in errors))

    def test_referential_integrity(self):
        """Rule 14: Unregistered student_id triggers REFERENTIAL_INTEGRITY error."""
        contest_data = {
            "contest_id": "test",
            "title": "Ref Integrity Test",
            "contest_type": "INTERNAL_SCHOOL",
            "start_time": "2026-09-15T13:00:00+08:00",
            "standings": [
                {
                    "team_name": "T1",
                    "is_school": True,
                    "members": ["UNKNOWN_99999"],
                    "solved": 1,
                    "penalty": 10,
                }
            ],
        }
        errors = validate_contest_dict(contest_data, valid_student_ids={"23001003"})
        self.assertTrue(any("REFERENTIAL_INTEGRITY" in e for e in errors))

    def test_duplicate_members_in_same_team(self):
        """Rule 15: Duplicate member in same team triggers CONTEST_INTEGRITY error."""
        contest_data = {
            "contest_id": "test",
            "title": "Dup Member Test",
            "contest_type": "INTERNAL_SCHOOL",
            "start_time": "2026-09-15T13:00:00+08:00",
            "standings": [
                {
                    "team_name": "T1",
                    "is_school": True,
                    "members": ["23001003", "23001003"],
                    "solved": 1,
                    "penalty": 10,
                }
            ],
        }
        errors = validate_contest_dict(contest_data, valid_student_ids={"23001003"})
        self.assertTrue(any("CONTEST_INTEGRITY" in e and "Duplicate member" in e for e in errors))

    def test_student_in_multiple_teams(self):
        """Rule 15: Student in multiple teams triggers CONTEST_INTEGRITY error."""
        contest_data = {
            "contest_id": "test",
            "title": "Multi Team Test",
            "contest_type": "INTERNAL_SCHOOL",
            "start_time": "2026-09-15T13:00:00+08:00",
            "standings": [
                {
                    "team_name": "T1",
                    "is_school": True,
                    "members": ["23001003"],
                    "solved": 1,
                    "penalty": 10,
                },
                {
                    "team_name": "T2",
                    "is_school": True,
                    "members": ["23001003"],
                    "solved": 1,
                    "penalty": 10,
                },
            ],
        }
        errors = validate_contest_dict(contest_data, valid_student_ids={"23001003"})
        self.assertTrue(any("CONTEST_INTEGRITY" in e and "already assigned" in e for e in errors))

    def test_negative_solved_and_penalty(self):
        """Rule 15: Negative solved or penalty triggers CONTEST_INTEGRITY error."""
        contest_data = {
            "contest_id": "test",
            "title": "Negative Test",
            "contest_type": "INTERNAL_SCHOOL",
            "start_time": "2026-09-15T13:00:00+08:00",
            "standings": [
                {
                    "team_name": "T1",
                    "is_school": True,
                    "members": ["23001003"],
                    "solved": -1,
                    "penalty": -50,
                }
            ],
        }
        errors = validate_contest_dict(contest_data, valid_student_ids={"23001003"})
        self.assertTrue(any("solved must be non-negative" in e for e in errors))
        self.assertTrue(any("penalty must be non-negative" in e for e in errors))


class TestImportContestTool(unittest.TestCase):
    """Test import_contest.py across CSV, DOMjudge, and PTA formats."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dir_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_import_csv(self):
        """Verify generic CSV import produces valid contest JSON."""
        csv_file = self.dir_path / "raw.csv"
        csv_file.write_text(
            "rank,team_name,members,is_school,solved,penalty\n"
            "1,寒山闻钟,23001003,true,5,300\n"
            "2,清华紫荆,,false,4,280\n",
            encoding="utf-8",
        )
        out_json = self.dir_path / "2026_11_10_csv_test.json"
        res = import_contest(
            input_path=csv_file,
            output_path=out_json,
            fmt="csv",
            type="EXTERNAL_XCPC",
            users_path=SAMPLE_USERS,
        )
        self.assertEqual(len(res["standings"]), 2)
        self.assertTrue(out_json.is_file())
        errors = validate_contest_file(out_json, valid_student_ids={"23001003"})
        self.assertEqual(len(errors), 0)

    def test_import_domjudge(self):
        """Verify DOMjudge JSON import produces valid contest JSON."""
        sb_file = self.dir_path / "sb.json"
        sb_file.write_text(
            json.dumps([{"team_id": "1", "score": {"num_solved": 6, "total_time": 450}}]),
            encoding="utf-8",
        )
        teams_file = self.dir_path / "teams.json"
        teams_file.write_text(
            json.dumps([{
                "id": "1",
                "name": "蓝雨先锋",
                "affiliation": "苏州科技大学",
                "members": "23001003, 23001004",
            }]),
            encoding="utf-8",
        )
        out_json = self.dir_path / "2026_11_11_domjudge_test.json"
        res = import_contest(
            input_path=sb_file,
            output_path=out_json,
            fmt="domjudge",
            teams_path=teams_file,
            type="EXTERNAL_XCPC",
            users_path=SAMPLE_USERS,
        )
        self.assertEqual(len(res["standings"]), 1)
        self.assertEqual(res["standings"][0]["members"], ["23001003", "23001004"])
        errors = validate_contest_file(out_json, valid_student_ids={"23001003", "23001004"})
        self.assertEqual(len(errors), 0)

    def test_import_pta(self):
        """Verify PTA JSON import produces valid contest JSON."""
        pta_file = self.dir_path / "pta.json"
        pta_data = {
            "commonRankings": [
                {
                    "ranking": 1,
                    "user": {
                        "studentUser": {"studentNumber": "23001003", "name": "测试队员甲"},
                        "organization": "苏州科技大学",
                    },
                    "score": 7,
                    "timeConsuming": 400,
                }
            ]
        }
        pta_file.write_text(json.dumps(pta_data), encoding="utf-8")
        out_json = self.dir_path / "2026_11_12_pta_test.json"
        res = import_contest(
            input_path=pta_file,
            output_path=out_json,
            fmt="pta",
            type="INTERNAL_SCHOOL",
            users_path=SAMPLE_USERS,
        )
        self.assertEqual(len(res["standings"]), 1)
        self.assertEqual(res["standings"][0]["members"], ["23001003"])
        errors = validate_contest_file(out_json, valid_student_ids={"23001003"})
        self.assertEqual(len(errors), 0)

    def test_import_csv_chinese_headers_and_values(self):
        """CSV import with Chinese headers (是否本校) and values (是/否) recognizes school team and preserves members."""
        csv_file = self.dir_path / "chinese_headers.csv"
        csv_file.write_text(
            "排名,队名,队员学号,是否本校,解题数,罚时\n"
            "1,寒山闻钟,23001003,是,5,300\n"
            "2,外校队伍,,否,4,280\n"
            "3,未知队伍,23001004,不确定,3,250\n",
            encoding="utf-8-sig",
        )
        out_json = self.dir_path / "2026_11_15_chinese_csv_test.json"
        res = import_contest(
            input_path=csv_file,
            output_path=out_json,
            fmt="csv",
            type="EXTERNAL_XCPC",
            users_path=SAMPLE_USERS,
        )
        self.assertEqual(len(res["standings"]), 3)

        # Team 1: '是' -> is_school=True, preserves members
        t1 = next(t for t in res["standings"] if t["team_name"] == "寒山闻钟")
        self.assertTrue(t1["is_school"])
        self.assertEqual(t1["members"], ["23001003"])
        self.assertIsNone(t1["external_elo"])

        # Team 2: '否' -> is_school=False, empty members, has external_elo
        t2 = next(t for t in res["standings"] if t["team_name"] == "外校队伍")
        self.assertFalse(t2["is_school"])
        self.assertEqual(t2["members"], [])
        self.assertEqual(t2["external_elo"], 1500)

        # Team 3: '不确定' (unparseable) -> fallback to is_school_team() which identifies registered student 23001004
        t3 = next(t for t in res["standings"] if t["team_name"] == "未知队伍")
        self.assertTrue(t3["is_school"])
        self.assertEqual(t3["members"], ["23001004"])

        errors = validate_contest_file(out_json, valid_student_ids={"23001003", "23001004"})
        self.assertEqual(errors, [])

    def test_import_external_elo_anchoring(self):
        """Verify dynamic external Elo anchoring by rank and by file mapping."""
        csv_data = (
            "rank,team_name,organization,solved,penalty,members\n"
            "1,清华大学1队,清华大学,11,800,\n"
            "2,上海交通大学1队,上海交通大学,10,750,\n"
            "3,苏科大1队,苏州科技大学,8,600,23001003\n"
            "4,某普通高校队,某普通高校,2,200,\n"
        )
        csv_file = self.dir_path / "ranking_test.csv"
        csv_file.write_text(csv_data, encoding="utf-8")

        # Test mode="rank"
        out_rank_json = self.dir_path / "2026_11_20_rank_anchor.json"
        res_rank = import_contest(
            input_path=csv_file,
            output_path=out_rank_json,
            fmt="csv",
            contest_type="EXTERNAL_XCPC",
            users_path=SAMPLE_USERS,
            external_elo_mode="rank",
        )
        t_tsinghua = next(t for t in res_rank["standings"] if "清华" in t["team_name"])
        t_bottom = next(t for t in res_rank["standings"] if "普通高校" in t["team_name"])
        self.assertGreaterEqual(t_tsinghua["external_elo"], 2500)
        self.assertLessEqual(t_bottom["external_elo"], 1400)

        # Test external_elo_file
        elo_map_file = self.dir_path / "elo_map.json"
        elo_map_file.write_text(json.dumps({"清华大学": 2900, "上海交通大学": 2650}), encoding="utf-8")
        out_map_json = self.dir_path / "2026_11_20_map_anchor.json"
        res_map = import_contest(
            input_path=csv_file,
            output_path=out_map_json,
            fmt="csv",
            contest_type="EXTERNAL_XCPC",
            users_path=SAMPLE_USERS,
            external_elo_file=elo_map_file,
        )
        t_tsinghua_map = next(t for t in res_map["standings"] if "清华" in t["team_name"])
        self.assertEqual(t_tsinghua_map["external_elo"], 2900)

    def test_import_srk(self):
        """Test importing standard ranklist (.srk.json) format from algoux/srk-collection."""
        srk_data = {
            "type": "general",
            "version": "0.3.13",
            "contest": {
                "title": {
                    "zh-CN": "2025年江苏省大学生程序设计竞赛",
                    "fallback": "2025 Jiangsu Collegiate Programming Contest"
                },
                "startAt": "2025-06-02T09:30:00+08:00"
            },
            "problems": [
                {"alias": "A", "title": "Problem A"}
            ],
            "rows": [
                {
                    "user": {
                        "name": "青石队",
                        "organization": "南京大学",
                        "teamMembers": [
                            {"name": "某选手A"},
                            {"name": "某选手B"}
                        ],
                        "official": True
                    },
                    "score": {
                        "value": 9,
                        "time": [42000, "s"]
                    }
                },
                {
                    "user": {
                        "name": "苏科大一队",
                        "organization": "苏州科技大学",
                        "teamMembers": [
                            {"name": "测试队员甲"},
                            {"name": "测试队员乙"},
                            {"name": "某指导老师", "role": "coach"}
                        ],
                        "official": True
                    },
                    "score": {
                        "value": 7,
                        "time": [36000, "s"]
                    }
                }
            ]
        }

        srk_file = self.dir_path / "2025_06_02_jscpc.srk.json"
        with srk_file.open("w", encoding="utf-8") as f:
            json.dump(srk_data, f, ensure_ascii=False)

        users_csv = self.dir_path / "users.csv"
        users_csv.write_text(
            "student_id,name,grade,initial_external_elo,is_active\n"
            "23001003,测试队员甲,2023,1650,true\n"
            "23001004,测试队员乙,2023,1580,true\n",
            encoding="utf-8"
        )

        out_json = self.dir_path / "2025_06_02_jscpc.json"
        elo_file = self.dir_path / "external_elo_map.json"
        elo_file.write_text(json.dumps({"organizations": {"南京大学": 2500}}), encoding="utf-8")
        res = import_contest(
            input_path=srk_file,
            output_path=out_json,
            fmt="auto",
            contest_type="EXTERNAL_XCPC",
            users_path=users_csv,
            external_elo_file=elo_file,
        )

        self.assertEqual(res["title"], "2025年江苏省大学生程序设计竞赛")
        self.assertEqual(res["start_time"], "2025-06-02T09:30:00+08:00")
        self.assertEqual(len(res["standings"]), 2)

        # The external team name does not contain its organization's name.
        t1 = res["standings"][0]
        self.assertEqual(t1["team_name"], "青石队")
        self.assertEqual(t1["organization"], "南京大学")
        self.assertFalse(t1["is_school"])
        self.assertEqual(t1["solved"], 9)
        self.assertEqual(t1["penalty"], 700)  # 42000s // 60 = 700min
        self.assertEqual(t1["external_elo"], 2500)
        self.assertEqual(t1["members"], [])

        # Team 2: 苏科大一队 (school, mapped to student IDs, coach filtered)
        t2 = res["standings"][1]
        self.assertEqual(t2["team_name"], "苏科大一队")
        self.assertTrue(t2["is_school"])
        self.assertEqual(t2["solved"], 7)
        self.assertEqual(t2["penalty"], 600)  # 36000s // 60 = 600min
        self.assertEqual(t2["members"], ["23001003", "23001004"])

        errors = validate_contest_file(out_json, valid_student_ids={"23001003", "23001004"})
        self.assertEqual(errors, [])

    def test_import_srk_rejects_missing_school_members(self):
        users_csv = self.dir_path / "users.csv"
        users_csv.write_text(
            "student_id,name,grade,initial_external_elo,is_active\n"
            "23001003,Known Student,2023,,true\n",
            encoding="utf-8",
        )
        srk_file = self.dir_path / "school.srk.json"
        out_json = self.dir_path / "2025_06_02_school.json"
        for members in (["Known Student", "Unknown Student"], ["Unknown Student"], []):
            with self.subTest(members=members):
                srk_file.write_text(json.dumps({
                    "contest": {"startAt": "2025-06-02T09:30:00+08:00"},
                    "rows": [{
                        "user": {
                            "name": "Local Team",
                            "organization": "USTS",
                            "teamMembers": members,
                        },
                        "score": {"value": 1},
                    }],
                }), encoding="utf-8")
                message = "Unknown Student" if members else "no participant names or IDs"
                with self.assertRaisesRegex(ValueError, message):
                    import_contest(
                        srk_file, out_json, fmt="srk", contest_type="EXTERNAL_XCPC",
                        users_path=users_csv,
                    )
                self.assertFalse(out_json.exists())


def test_recalculation_rejects_unreadable_scoreboard(tmp_path):
    broken = tmp_path / "2025_06_02_broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match=broken.name):
        load_contests(tmp_path)


def test_outside_freshman_has_rating_history_before_and_after_joining(tmp_path):
    start = datetime.now(timezone.utc)
    roster = tmp_path / "users.csv"
    roster.write_text(
        "student_id,name,grade,initial_external_elo,is_active,is_team_member\n"
        f"SCHOOL01,Team member,{start.year},,true,true\n"
        f"SCHOOL02,Freshman,{start.year},,true,false\n",
        encoding="utf-8",
    )
    raw = tmp_path / "scoreboard.csv"
    raw.write_text(
        "team_name,members,is_school,solved,penalty\n"
        "Team member,SCHOOL01,true,2,90\n"
        "Freshman,SCHOOL02,true,3,120\n",
        encoding="utf-8",
    )
    contests = tmp_path / "contests"
    contest_path = contests / f"{start:%Y_%m_%d}_training.json"
    import_contest(raw, contest_path, users_path=roster, start_time=start.isoformat())
    assert validate_all(roster, contests) == (True, [])

    before = recalculate_all(tmp_path, [tmp_path / "before.json"])
    freshman = next(user for user in before["users"] if user["name"] == "Freshman")
    assert freshman["is_team_member"] is False
    assert freshman["is_retired"] is False
    assert freshman["rated_count"] == 1
    assert freshman["history"][0]["delta"] > 0
    assert before["contests"][0]["standings"][0]["members"] == [freshman["participant_id"]]

    roster.write_text(
        roster.read_text(encoding="utf-8").replace(",true,false\n", ",true,true\n"),
        encoding="utf-8",
    )
    after = recalculate_all(tmp_path, [tmp_path / "after.json"])
    promoted = next(user for user in after["users"] if user["name"] == "Freshman")
    assert promoted == {**freshman, "is_team_member": True}


def test_roster_rejects_invalid_team_membership(tmp_path):
    roster = tmp_path / "users.csv"
    roster.write_text(
        "student_id,name,grade,initial_external_elo,is_active,is_team_member\n"
        "SCHOOL01,Test Student,2026,,true,maybe\n",
        encoding="utf-8",
    )
    _, errors = validate_users_csv(roster)
    assert len(errors) == 1
    assert "[IS_TEAM_MEMBER_VALID]" in errors[0]


if __name__ == "__main__":
    unittest.main()
