"""
Test suite for cold/hot data separation and first-screen performance optimization:
1. summary.json size strictly <= 200 KB (uncompressed).
2. summary.json schema integrity (users, history with solved/penalty, contest metadata).
3. Individual contest files generated in contests/<contest_id>.json.
4. Compact formatting (no extraneous indentation/whitespace).
5. Frontend app.js behavior (Node.js vm probe): SWR localStorage, lazy contest loading, Map cache.
"""

import json
from pathlib import Path
import shutil
import subprocess
import unittest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
WEB_DIR = WORKSPACE_ROOT / "web"
DIST_DIR = WORKSPACE_ROOT / "dist"


class TestColdHotDataSplitting(unittest.TestCase):
    """Verify cold/hot data splitting and size budgets."""

    def test_summary_json_size_under_200kb(self):
        """summary.json uncompressed size must be strictly under 200 KB."""
        for target_dir in [WEB_DIR / "data", DIST_DIR / "data"]:
            summary_path = target_dir / "summary.json"
            self.assertTrue(summary_path.exists(), f"{summary_path} must exist")
            size_bytes = summary_path.stat().st_size
            size_kb = size_bytes / 1024.0
            self.assertLess(
                size_kb,
                200.0,
                f"{summary_path} size {size_kb:.2f} KB exceeds 200 KB limit",
            )
            # Verify compact JSON: no leading spaces or indentation
            raw_text = summary_path.read_text(encoding="utf-8")
            self.assertNotIn("\n  ", raw_text, f"{summary_path} should be compact without indentation")

    def test_summary_json_schema_and_metadata(self):
        """summary.json must contain required metadata, tiers, users, and contest summaries."""
        summary_path = WEB_DIR / "data" / "summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertIn("metadata", data)
        self.assertIn("tiers", data)
        self.assertIn("users", data)
        self.assertIn("contests", data)

        self.assertEqual(data["metadata"]["total_users"], 116)
        self.assertEqual(data["metadata"]["total_contests"], 44)
        self.assertEqual(len(data["users"]), 116)
        self.assertEqual(len(data["contests"]), 44)

        # Check that users have history with solved and penalty
        users_with_history = [u for u in data["users"] if len(u.get("history", [])) > 0]
        self.assertGreater(len(users_with_history), 0)
        for u in users_with_history:
            for h in u["history"]:
                self.assertIn("solved", h)
                self.assertIn("penalty", h)
                self.assertIsInstance(h["solved"], int)
                self.assertIsInstance(h["penalty"], int)

        # Contests in summary must have team_count and no heavy standings
        for c in data["contests"]:
            self.assertIn("contest_id", c)
            self.assertIn("title", c)
            self.assertIn("contest_type", c)
            self.assertIn("start_time", c)
            self.assertIn("team_count", c)
            self.assertGreater(c["team_count"], 0)
            self.assertEqual(len(c.get("standings", [])), 0)

    def test_individual_contests_generated(self):
        """All 44 contests must exist in contests/<contest_id>.json in both web and dist."""
        summary_path = WEB_DIR / "data" / "summary.json"
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        contest_ids = [c["contest_id"] for c in summary_data["contests"]]
        self.assertEqual(len(contest_ids), 44)

        for target_dir in [WEB_DIR / "data" / "contests", DIST_DIR / "data" / "contests"]:
            self.assertTrue(target_dir.is_dir(), f"{target_dir} must be a directory")
            for cid in contest_ids:
                c_file = target_dir / f"{cid}.json"
                self.assertTrue(c_file.exists(), f"Contest file {c_file} must exist")
                c_data = json.loads(c_file.read_text(encoding="utf-8"))
                self.assertEqual(c_data["contest_id"], cid)
                self.assertIn("team_count", c_data)
                self.assertIn("standings", c_data)
                self.assertGreater(len(c_data["standings"]), 0)
                self.assertEqual(c_data["team_count"], len(c_data["standings"]))


class TestFrontendDataLoadingBehavior(unittest.TestCase):
    """Dynamic test for web/app.js lazy loading, SWR caching, and Map cache."""

    def test_app_js_swr_and_lazy_load_node_simulation(self):
        node_bin = shutil.which("node")
        if not node_bin:
            self.skipTest("Node.js is required for dynamic app.js simulation.")

        probe_script = """
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const appCode = fs.readFileSync('web/app.js', 'utf8');

// Mock localStorage
const mockStorage = new Map();
const localStorage = {
    getItem: key => mockStorage.get(key) || null,
    setItem: (key, val) => mockStorage.set(key, String(val)),
    removeItem: key => mockStorage.delete(key),
    clear: () => mockStorage.clear()
};

// Mock fetch
const fetchResponses = new Map();
const fetchCalls = [];
global.fetch = async function(url) {
    fetchCalls.push(url);
    if (fetchResponses.has(url)) {
        return {
            ok: true,
            json: async () => JSON.parse(JSON.stringify(fetchResponses.get(url)))
        };
    }
    return { ok: false, status: 404 };
};

let state;
const context = {
    Date: Date,
    localStorage: localStorage,
    fetch: global.fetch,
    Vue: {
        createApp: ({ setup }) => ({ mount() { state = setup(); } }),
        ref: value => ({ value }),
        computed: getter => ({ get value() { return getter(); } }),
        onMounted: (fn) => { /* manual trigger */ },
        onUnmounted: () => {}
    }
};

vm.runInNewContext(appCode, context);

// 1. Initial test: summary.json loading without cache
const dummySummary = {
    metadata: { generated_at: '2026-11-22T00:00:00Z', total_users: 1, total_contests: 1 },
    tiers: [],
    users: [{
        participant_id: 1,
        name: 'Alice',
        grade: 2024,
        is_active: true,
        is_retired: false,
        is_team_member: true,
        display_rating: 1200,
        rated_count: 1,
        history: [{ contest_id: 'c1', solved: 4, penalty: 200, delta: 50, display_rating_after: 1200 }]
    }],
    contests: [{
        contest_id: 'c1',
        title: 'Contest 1',
        contest_type: 'INTERNAL_SCHOOL',
        start_time: '2026-11-20T00:00:00Z',
        team_count: 1,
        standings: []
    }]
};
fetchResponses.set('data/summary.json', dummySummary);

(async () => {
    // Call loadData without localStorage
    await state.loadData();
    assert.equal(state.db.value.users.length, 1);
    assert.equal(state.db.value.users[0].name, 'Alice');
    assert.equal(state.loading.value, false);
    // Verify cached in localStorage
    assert.ok(mockStorage.has('usts_rating_summary_cache_v1'));

    // 2. Test getUserContestStats from user history (without contest standings loaded)
    const stats = state.getUserContestStats(1, 'c1');
    assert.equal(stats.solved, 4);
    assert.equal(stats.penalty, 200);

    // 3. Test on-demand contest standings lazy load via toggleContest
    const dummyContest = {
        contest_id: 'c1',
        title: 'Contest 1',
        contest_type: 'INTERNAL_SCHOOL',
        start_time: '2026-11-20T00:00:00Z',
        standings: [{
            rank: 1,
            team_name: 'Team Alice',
            is_school: true,
            members: [1],
            solved: 4,
            penalty: 200,
            delta: 50
        }]
    };
    fetchResponses.set('data/contests/c1.json', dummyContest);

    const initialFetchCalls = fetchCalls.length;
    await state.toggleContest('c1');
    assert.ok(state.isContestExpanded('c1'));
    // Verify contestCache populated
    assert.ok(state.contestCache.has('c1'));
    assert.equal(state.db.value.contests[0].standings.length, 1);
    assert.equal(state.db.value.contests[0].standings[0].team_name, 'Team Alice');

    // 4. Test memory cache: toggle closed and open again should NOT trigger network fetch
    await state.toggleContest('c1'); // close
    assert.equal(state.isContestExpanded('c1'), false);
    const countBeforeReopen = fetchCalls.length;
    await state.toggleContest('c1'); // open again
    assert.ok(state.isContestExpanded('c1'));
    assert.equal(fetchCalls.length, countBeforeReopen, 'Should use Map cache without extra fetch');

    // 5. Test concurrent fetch deduplication on a new contest 'c2'
    const dummyContest2 = {
        contest_id: 'c2',
        title: 'Contest 2',
        contest_type: 'EXTERNAL_XCPC',
        start_time: '2026-11-21T00:00:00Z',
        standings: [{ rank: 1, team_name: 'Team 2', is_school: true, members: [1], solved: 2, penalty: 100 }]
    };
    fetchResponses.set('data/contests/c2.json', dummyContest2);
    const callsBeforeConcurrent = fetchCalls.length;
    const [p1, p2] = await Promise.all([
        state.loadContestStandings('c2'),
        state.loadContestStandings('c2')
    ]);
    assert.equal(fetchCalls.length, callsBeforeConcurrent + 1, 'Concurrent requests should be deduplicated to 1 fetch');
    assert.equal(p1.contest_id, 'c2');
    assert.equal(p2.contest_id, 'c2');

    // 6. Test localStorage hygiene: ensure standings are NOT saved into localStorage summary cache
    assert.ok(mockStorage.has('usts_rating_summary_cache_v1'));
    const savedInStorage = JSON.parse(mockStorage.get('usts_rating_summary_cache_v1'));
    for (const sc of savedInStorage.contests) {
        assert.equal(sc.standings.length, 0, 'LocalStorage must never contain contest standings');
    }

    // 7. Test SWR: with cached data in localStorage, loadData renders immediately
    const cachedItem = JSON.parse(mockStorage.get('usts_rating_summary_cache_v1'));
    cachedItem.users[0].name = 'Cached Alice';
    cachedItem.metadata.generated_at = '2026-11-21T00:00:00Z'; // older version in cache
    mockStorage.set('usts_rating_summary_cache_v1', JSON.stringify(cachedItem));

    // Server has newer version (generated_at: 2026-11-22T00:00:00Z)
    // Reset db
    state.db.value = { users: [], contests: [] };
    // Call loadData - it should read cache immediately
    const loadPromise = state.loadData();
    // Immediate state check (stale)
    assert.equal(state.db.value.users[0].name, 'Cached Alice');
    await loadPromise;
    // Fresh revalidation replaced it
    assert.equal(state.db.value.users[0].name, 'Alice');

    // Storage must STILL have standings: []
    const updatedStorage = JSON.parse(mockStorage.get('usts_rating_summary_cache_v1'));
    for (const sc of updatedStorage.contests) {
        assert.equal(sc.standings.length, 0, 'LocalStorage must still have standings: [] after revalidation');
    }

    console.log('ALL_SWR_AND_LAZY_LOAD_PROBES_PASSED');
})();
"""
        res = subprocess.run([node_bin, "-e", probe_script], cwd=str(WORKSPACE_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Node probe failed:\n{res.stderr}\n{res.stdout}")
        self.assertIn("ALL_SWR_AND_LAZY_LOAD_PROBES_PASSED", res.stdout)


if __name__ == "__main__":
    unittest.main()
