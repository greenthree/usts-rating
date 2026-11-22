/**
 * Suzhou University of Science and Technology (USTS)
 * ACM/XCPC Rating System - Core Application Logic
 * Pure Static Vue 3 + Tailwind CSS + Apache ECharts Dashboard
 */

const { createApp, ref, computed, onMounted, onUnmounted, nextTick } = Vue;

// 7-Tier Codeforces-style classification system (800+ green name start, 200 pts/tier)
const TIERS = [
  {
    name: 'Newbie',
    zh: '萌新',
    min: 0,
    max: 799,
    color: '#808080',
    bg: '#f3f4f6',
    text: '#4b5563',
    border: '#d1d5db',
    badgeClass: 'badge-newbie'
  },
  {
    name: 'Pupil',
    zh: '进阶',
    min: 800,
    max: 999,
    color: '#008000',
    bg: '#dcfce7',
    text: '#15803d',
    border: '#86efac',
    badgeClass: 'badge-pupil'
  },
  {
    name: 'Specialist',
    zh: '骨干',
    min: 1000,
    max: 1199,
    color: '#03A89E',
    bg: '#ccfbf1',
    text: '#0f766e',
    border: '#5eead4',
    badgeClass: 'badge-specialist'
  },
  {
    name: 'Expert',
    zh: '精英',
    min: 1200,
    max: 1399,
    color: '#0000FF',
    bg: '#dbeafe',
    text: '#1d4ed8',
    border: '#93c5fd',
    badgeClass: 'badge-expert'
  },
  {
    name: 'Candidate Master',
    zh: '候选大师',
    min: 1400,
    max: 1599,
    color: '#AA00AA',
    bg: '#f3e8ff',
    text: '#7e22ce',
    border: '#d8b4fe',
    badgeClass: 'badge-candidate-master'
  },
  {
    name: 'Master',
    zh: '校队大师',
    min: 1600,
    max: 1799,
    color: '#FF8C00',
    bg: '#ffedd5',
    text: '#c2410c',
    border: '#fdba74',
    badgeClass: 'badge-master'
  },
  {
    name: 'Grandmaster',
    zh: '传奇宗师',
    min: 1800,
    max: 99999,
    color: '#FF0000',
    bg: '#fee2e2',
    text: '#b91c1c',
    border: '#fca5a5',
    badgeClass: 'badge-grandmaster'
  }
];

// Helper to look up tier object from display rating
function getTier(rating) {
  const r = Math.max(0, Math.round(Number(rating) || 0));
  if (r < 800) return TIERS[0];
  if (r < 1000) return TIERS[1];
  if (r < 1200) return TIERS[2];
  if (r < 1400) return TIERS[3];
  if (r < 1600) return TIERS[4];
  if (r < 1800) return TIERS[5];
  return TIERS[6];
}

// Provisional display offset schedule (3-contest newbie protection, real rating shown upon finishing 3rd contest)
function getDisplayOffset(ratedCount) {
  const cnt = Number(ratedCount) || 0;
  if (cnt <= 0) return -1000;
  if (cnt === 1) return -500;
  if (cnt === 2) return -250;
  return 0;
}

const NON_MEMBER_INACTIVITY_MS = 365 * 24 * 60 * 60 * 1000;

function isLeaderboardVisible(user, now) {
  if (user.is_team_member !== false) return true;
  if (!user.is_active) return false;

  const enrollmentYear = Number(user.grade);
  const graduation = Date.parse(`${enrollmentYear + 4}-07-01T00:00:00+08:00`);
  const lastParticipation = Date.parse(
    user.last_contest_date || `${enrollmentYear}-09-01T00:00:00+08:00`
  );

  return now < graduation && now - lastParticipation < NON_MEMBER_INACTIVITY_MS;
}

// HTML entity escaping utility to prevent XSS in raw HTML contexts (e.g. ECharts tooltips)
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const app = createApp({
  setup() {
    // Navigation & View State
    const activeTab = ref('leaderboard'); // 'leaderboard' | 'contests' | 'rules'
    const loading = ref(true);
    const errorMessage = ref('');

    // Rating database reactive state
    const db = ref({
      metadata: {
        generated_at: '',
        total_users: 0,
        active_users: 0,
        retired_users: 0,
        total_contests: 0
      },
      tiers: TIERS,
      users: [],
      contests: []
    });

    // Leaderboard Filter State
    const statusFilter = ref('active'); // 'active' | 'retired' | 'all'
    const gradeFilter = ref('all'); // 'all' | '2026' | '2025' ...
    const searchQuery = ref('');

    // Modal State
    const showModal = ref(false);
    const selectedUser = ref(null);
    let chartInstance = null;

    // Contests View State
    const contestCategories = [
      { type: 'INTERNAL_SCHOOL', label: '校内赛' },
      { type: 'EXTERNAL_XCPC', label: '校外赛' }
    ];
    const contestType = ref('EXTERNAL_XCPC');
    const expandedContestIds = ref(new Set());

    // Candidate relative paths for resilient data loading
    const candidateSummaryPaths = [
      'data/summary.json',
      '../data/summary.json',
      './data/summary.json',
      '../dist/data/summary.json',
      'dist/data/summary.json',
      // Legacy fallback
      'data/rating_db.json',
      '../data/rating_db.json',
      './data/rating_db.json',
      '../dist/data/rating_db.json',
      'dist/data/rating_db.json'
    ];
    const SUMMARY_CACHE_KEY = 'usts_rating_summary_cache_v1';
    let activeDataBasePath = '';

    // In-memory cache and loading state for lazy-loaded contest standings
    const contestCache = new Map();
    const contestLoading = ref(new Set());
    const contestErrors = ref(new Map());
    const inFlightContestRequests = new Map();

    function isContestLoading(contestId) {
      return contestLoading.value.has(contestId);
    }

    function getContestError(contestId) {
      return contestErrors.value.get(contestId) || '';
    }

    function applyContestStandings(contestId, standings) {
      if (db.value && Array.isArray(db.value.contests)) {
        const contest = db.value.contests.find(c => c.contest_id === contestId);
        if (contest) {
          contest.standings = standings;
        }
      }
    }

    // Lazy load contest details with Map cache and request deduplication
    async function loadContestStandings(contestId) {
      if (contestCache.has(contestId)) {
        const cached = contestCache.get(contestId);
        applyContestStandings(contestId, cached.standings);
        return cached;
      }

      if (inFlightContestRequests.has(contestId)) {
        return inFlightContestRequests.get(contestId);
      }

      contestLoading.value.add(contestId);
      contestErrors.value.delete(contestId);

      const candidateContestPaths = [];
      if (activeDataBasePath) {
        candidateContestPaths.push(`${activeDataBasePath}contests/${contestId}.json`);
      }
      for (const p of [
        `data/contests/${contestId}.json`,
        `../data/contests/${contestId}.json`,
        `./data/contests/${contestId}.json`,
        `../dist/data/contests/${contestId}.json`,
        `dist/data/contests/${contestId}.json`
      ]) {
        if (!candidateContestPaths.includes(p)) {
          candidateContestPaths.push(p);
        }
      }

      const fetchPromise = (async () => {
        let contestData = null;
        for (const p of candidateContestPaths) {
          try {
            const res = await fetch(p);
            if (res.ok) {
              contestData = await res.json();
              if (contestData && Array.isArray(contestData.standings)) {
                break;
              }
            }
          } catch (e) {
            // Continue attempting candidate paths
          }
        }

        contestLoading.value.delete(contestId);
        inFlightContestRequests.delete(contestId);

        if (contestData && Array.isArray(contestData.standings)) {
          contestCache.set(contestId, contestData);
          applyContestStandings(contestId, contestData.standings);
          return contestData;
        } else {
          contestErrors.value.set(contestId, '无法加载赛事榜单明细，请重试');
          return null;
        }
      })();

      inFlightContestRequests.set(contestId, fetchPromise);
      return fetchPromise;
    }

    // Fast lookup maps
    const usersMap = computed(() => {
      const map = new Map();
      for (const u of (db.value.users || [])) {
        map.set(String(u.participant_id), u);
      }
      return map;
    });

    // Detailed contest member stats lookup map: contest_id -> participant_id -> stats
    const contestMemberStats = computed(() => {
      const map = {};
      for (const c of (db.value.contests || [])) {
        map[c.contest_id] = {};
        for (const s of (c.standings || [])) {
          if (Array.isArray(s.members)) {
            for (const mId of s.members) {
              map[c.contest_id][String(mId)] = {
                solved: s.solved,
                penalty: s.penalty,
                rank: s.rank,
                team_name: s.team_name,
                team_rating: s.team_rating,
                perf_rating: s.perf_rating,
                team_delta: s.delta
              };
            }
          }
        }
      }
      return map;
    });

    const leaderboardUsers = computed(() => {
      const now = Date.now();
      return (db.value.users || []).filter(user => isLeaderboardVisible(user, now));
    });

    // Dynamic Grades list
    const grades = computed(() => {
      const gradeSet = new Set();
      for (const u of leaderboardUsers.value) {
        if (u.grade) gradeSet.add(Number(u.grade));
      }
      return Array.from(gradeSet).sort((a, b) => b - a);
    });

    // Summary Statistics Cards
    const summaryStats = computed(() => {
      const allUsers = leaderboardUsers.value;
      const teamMembers = allUsers.filter(u => u.is_team_member !== false);
      const activeMembers = teamMembers.filter(u => u.is_active && !u.is_retired);
      const activeCount = activeMembers.length;

      // High tier members: Master (>=1600) & Grandmaster (>=1800)
      const highTierCount = allUsers.filter(u => Number(u.display_rating) >= 1600).length;

      // Average display rating of active members
      const avgRating = activeCount > 0
        ? Math.round(activeMembers.reduce((sum, u) => sum + (Number(u.display_rating) || 0), 0) / activeCount)
        : 0;

      const contestCount = db.value.contests ? db.value.contests.length : (db.value.metadata.total_contests || 0);

      return {
        activeCount,
        teamCount: teamMembers.length,
        highTierCount,
        avgRating,
        contestCount
      };
    });

    // Filtered & Ranked Users for Leaderboard Table
    const filteredUsers = computed(() => {
      let list = [...leaderboardUsers.value];

      // 1. Status Filter
      if (statusFilter.value === 'active') {
        list = list.filter(u => u.is_active && !u.is_retired);
      } else if (statusFilter.value === 'retired') {
        list = list.filter(u => !u.is_active || u.is_retired);
      }

      // 2. Grade Filter
      if (gradeFilter.value !== 'all') {
        list = list.filter(u => String(u.grade) === String(gradeFilter.value));
      }

    // 3. Search Query (Name, Pinyin, Class/Major)
      const q = searchQuery.value.trim().toLowerCase();
      if (q) {
        list = list.filter(u => {
          const nameMatch = String(u.name || '').toLowerCase().includes(q);
          const pinyinMatch = String(u.pinyin || '').toLowerCase().includes(q);
          const classMatch = String(u.class_name || u.major || '').toLowerCase().includes(q);
          return nameMatch || pinyinMatch || classMatch;
        });
      }

      // 4. Stable sort by display rating, rated count, then participant ID
      list.sort((a, b) => {
        if (b.display_rating !== a.display_rating) {
          return b.display_rating - a.display_rating;
        }
        if (b.rated_count !== a.rated_count) {
          return b.rated_count - a.rated_count;
        }
        return a.participant_id - b.participant_id;
      });

      return list;
    });

    // Group contests by type, keeping the latest first in each category.
    const contestsByType = computed(() => {
      const list = [...(db.value.contests || [])];
      list.sort((a, b) => new Date(b.start_time || 0) - new Date(a.start_time || 0));
      return {
        INTERNAL_SCHOOL: list.filter(contest => contest.contest_type === 'INTERNAL_SCHOOL'),
        EXTERNAL_XCPC: list.filter(contest => contest.contest_type === 'EXTERNAL_XCPC')
      };
    });
    const visibleContests = computed(() => contestsByType.value[contestType.value]);

    // Resilient data loader with Stale-While-Revalidate caching strategy
    async function loadData() {
      errorMessage.value = '';
      let hasValidCache = false;

      // 1. Stale: Immediately render from localStorage cache if available
      try {
        if (typeof localStorage !== 'undefined') {
          const cachedStr = localStorage.getItem(SUMMARY_CACHE_KEY);
          if (cachedStr) {
            const cachedData = JSON.parse(cachedStr);
            if (cachedData && (Array.isArray(cachedData.users) || Array.isArray(cachedData.contests))) {
              db.value = cachedData;
              loading.value = false;
              hasValidCache = true;
            }
          }
        }
      } catch (e) {
        // Ignore localStorage error
      }

      if (!hasValidCache) {
        loading.value = true;
      }

      // 2. Revalidate: Fetch latest summary.json in background/foreground without cache-busting timestamp
      let freshData = null;
      let matchedPath = '';
      for (const p of candidateSummaryPaths) {
        try {
          const res = await fetch(p);
          if (res.ok) {
            const data = await res.json();
            if (data && Array.isArray(data.users) && Array.isArray(data.contests)) {
              freshData = data;
              matchedPath = p;
              break;
            }
          }
        } catch (e) {
          // Continue attempting candidate paths
        }
      }

      if (matchedPath) {
        const lastSlash = matchedPath.lastIndexOf('/');
        activeDataBasePath = lastSlash >= 0 ? matchedPath.substring(0, lastSlash + 1) : '';
      }

      if (freshData) {
        const curGen = db.value?.metadata?.generated_at;
        const freshGen = freshData.metadata?.generated_at;

        // Persist clean summary to localStorage (without heavy standings) to prevent storage quota exhaustion
        try {
          if (typeof localStorage !== 'undefined') {
            const cachePayload = {
              metadata: freshData.metadata,
              tiers: freshData.tiers,
              users: freshData.users,
              contests: (freshData.contests || []).map(c => ({
                contest_id: c.contest_id,
                title: c.title,
                contest_type: c.contest_type,
                start_time: c.start_time,
                team_count: c.team_count !== undefined ? c.team_count : (c.standings || []).length,
                standings: []
              }))
            };
            localStorage.setItem(SUMMARY_CACHE_KEY, JSON.stringify(cachePayload));
          }
        } catch (e) {
          // Storage quota exceeded or disabled
        }

        // If no cache was present or data was updated on server, update state
        if (!hasValidCache || curGen !== freshGen) {
          // Re-attach any already in-memory cached contest standings
          if (Array.isArray(freshData.contests)) {
            for (const fc of freshData.contests) {
              if (contestCache.has(fc.contest_id)) {
                fc.standings = contestCache.get(fc.contest_id).standings;
              }
            }
          }
          db.value = freshData;

          // If a user modal was open during revalidation, refresh selectedUser and chart
          if (selectedUser.value && Array.isArray(freshData.users)) {
            const updated = freshData.users.find(u => u.participant_id === selectedUser.value.participant_id);
            if (updated) {
              selectedUser.value = updated;
              nextTick(() => {
                renderChart(updated);
              });
            }
          }
        }

        loading.value = false;
        errorMessage.value = '';
      } else if (!hasValidCache) {
        loading.value = false;
        errorMessage.value = '未检测到积分数据库文件 summary.json (或 rating_db.json)。请先在项目根目录运行 `python scripts/recalculate.py` 生成积分聚合数据。';
      }
    }

    // Toggle contest standings accordion with on-demand lazy loading
    async function toggleContest(contestId) {
      if (expandedContestIds.value.has(contestId)) {
        expandedContestIds.value.delete(contestId);
      } else {
        expandedContestIds.value.add(contestId);
        await loadContestStandings(contestId);
      }
    }

    function isContestExpanded(contestId) {
      return expandedContestIds.value.has(contestId);
    }

    // Date formatting helper
    function formatDate(dateStr) {
      if (!dateStr) return '-';
      try {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
      } catch (e) {
        return dateStr;
      }
    }

    function formatDateTime(dateStr) {
      if (!dateStr) return '-';
      try {
        const d = new Date(dateStr);
        if (isNaN(d.getTime())) return dateStr;
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hour = String(d.getHours()).padStart(2, '0');
        const minute = String(d.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day} ${hour}:${minute}`;
      } catch (e) {
        return dateStr;
      }
    }

    // Best performance rating of a user
    function getBestPerf(user) {
      if (!user || !Array.isArray(user.history) || user.history.length === 0) {
        return '-';
      }
      let best = -9999;
      for (const h of user.history) {
        if (h.perf_rating !== null && h.perf_rating !== undefined && h.perf_rating !== '') {
          const p = Number(h.perf_rating);
          if (!isNaN(p) && p > best) best = p;
        }
      }
      return best > -9999 ? best.toFixed(1) : '-';
    }

    // Lookup solved/penalty for user's contest history
    function getUserContestStats(userId, contestId) {
      const stats = contestMemberStats.value[contestId]?.[String(userId)];
      if (stats && stats.solved !== undefined) {
        return {
          solved: stats.solved ?? '-',
          penalty: stats.penalty ?? '-'
        };
      }
      const user = usersMap.value.get(String(userId));
      const h = user?.history?.find(x => x.contest_id === contestId);
      if (h && h.solved !== undefined) {
        return {
          solved: h.solved,
          penalty: h.penalty
        };
      }
      return {
        solved: stats?.solved ?? '-',
        penalty: stats?.penalty ?? '-'
      };
    }

    // Render interactive ECharts line chart inside modal
    function renderChart(user) {
      const chartDom = document.getElementById('ratingChart');
      if (!chartDom) return;

      if (chartInstance) {
        chartInstance.dispose();
      }

      if (!window.echarts) {
        chartDom.innerHTML = '<div class="text-center text-gray-400 py-12">ECharts 库未加载完成</div>';
        return;
      }

      chartInstance = echarts.init(chartDom);

      const history = (user && Array.isArray(user.history)) ? [...user.history] : [];
      // Sort chronologically for chart display
      history.sort((a, b) => new Date(a.contest_date || 0) - new Date(b.contest_date || 0));

      if (history.length === 0) {
        chartInstance.setOption({
          title: {
            text: '暂无计分比赛记录',
            left: 'center',
            top: 'center',
            textStyle: { color: '#9CA3AF', fontSize: 14 }
          }
        });
        return;
      }

      const xData = history.map(h => {
        const dateFormatted = formatDate(h.contest_date);
        return dateFormatted !== '-' ? dateFormatted : (h.contest_title || h.contest_id);
      });

      const yData = history.map(h => {
        const disp = Number(h.display_rating_after) || 0;
        const tier = getTier(disp);
        return {
          value: disp,
          itemStyle: {
            color: tier.color,
            borderColor: '#FFFFFF',
            borderWidth: 2
          },
          raw: h
        };
      });

      const allRatings = history.map(h => Number(h.display_rating_after) || 0);
      const maxRating = Math.max(...allRatings, 1200);
      const yMax = Math.max(2000, Math.ceil((maxRating + 200) / 200) * 200);

      const option = {
        grid: {
          top: 35,
          left: 50,
          right: 35,
          bottom: 45,
          containLabel: true
        },
        tooltip: {
          trigger: 'item',
          backgroundColor: 'rgba(255, 255, 255, 0.98)',
          borderColor: '#E2E8F0',
          borderWidth: 1,
          padding: [12, 16],
          extraCssText: 'box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05); border-radius: 10px;',
          formatter: function(params) {
            const raw = params.data.raw || {};
            const delta = Number(raw.delta) || 0;
            const deltaFormatted = delta >= 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1);
            const deltaColor = delta >= 0 ? '#16A34A' : '#DC2626';
            const disp = Number(raw.display_rating_after) || 0;
            const tier = getTier(disp);
            const isInternal = raw.contest_type === 'INTERNAL_SCHOOL';
            const contestTypeZh = isInternal ? '校内赛 (权重 0.80)' : '校外赛 (权重 1.00)';
            const stats = getUserContestStats(user.participant_id, raw.contest_id);
            const safeTitle = escapeHtml(raw.contest_title || '比赛详情');
            const safeTeam = escapeHtml(raw.team_name || '-');

            return `
              <div style="font-family: system-ui, -apple-system, sans-serif; font-size: 13px; line-height: 1.6; min-width: 240px; color: #1E293B;">
                <div style="font-weight: 700; font-size: 14px; margin-bottom: 2px; color: #0F172A;">${safeTitle}</div>
                <div style="color: #64748B; font-size: 11px; margin-bottom: 10px; display: flex; align-items: center; gap: 4px;">
                  <span>📅 ${formatDate(raw.contest_date)}</span>
                  <span>•</span>
                  <span>${contestTypeZh}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span style="color: #64748B;">参赛队伍:</span>
                  <b style="color: #0F172A;">${safeTeam}</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span style="color: #64748B;">实际名次:</span>
                  <b style="color: #0F172A;">#${raw.rank || '-'} (${stats.solved} 题 / ${stats.penalty} 分)</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                  <span style="color: #64748B;">表现分 (Perf):</span>
                  <b style="color: #4F46E5;">${raw.perf_rating ? Number(raw.perf_rating).toFixed(1) : '-'}</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                  <span style="color: #64748B;">积分变动 (ΔR):</span>
                  <b style="color: ${deltaColor}; font-weight: 700;">${deltaFormatted}</b>
                </div>
                <div style="border-top: 1px solid #E2E8F0; padding-top: 8px; display: flex; justify-content: space-between; align-items: center;">
                  <span style="color: #64748B;">赛后显示分:</span>
                  <div style="display: flex; align-items: baseline; gap: 4px;">
                    <b style="color: ${tier.color}; font-size: 16px;">${disp}</b>
                    <span style="font-size: 11px; color: ${tier.color}; font-weight: 600;">(${tier.name})</span>
                  </div>
                </div>
              </div>
            `;
          }
        },
        xAxis: {
          type: 'category',
          data: xData,
          axisLine: { lineStyle: { color: '#CBD5E1' } },
          axisTick: { alignWithLabel: true },
          axisLabel: {
            color: '#64748B',
            fontSize: 11,
            rotate: xData.length > 5 ? 25 : 0
          }
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: yMax,
          splitLine: {
            show: true,
            lineStyle: { color: '#F1F5F9', type: 'dashed' }
          },
          axisLine: { show: true, lineStyle: { color: '#CBD5E1' } },
          axisLabel: { color: '#64748B', fontSize: 11 }
        },
        series: [
          {
            name: 'Display Rating',
            type: 'line',
            data: yData,
            smooth: 0.15,
            symbol: 'circle',
            symbolSize: 10,
            lineStyle: {
              color: '#475569',
              width: 3
            },
            markArea: {
              silent: true,
              data: [
                // 1. Newbie: 0 ~ 799 (#808080, gray)
                [
                  { yAxis: 0, itemStyle: { color: 'rgba(128, 128, 128, 0.08)' } },
                  { yAxis: 800 }
                ],
                // 2. Pupil: 800 ~ 999 (#008000, green)
                [
                  { yAxis: 800, itemStyle: { color: 'rgba(0, 128, 0, 0.08)' } },
                  { yAxis: 1000 }
                ],
                // 3. Specialist: 1000 ~ 1199 (#03A89E, cyan)
                [
                  { yAxis: 1000, itemStyle: { color: 'rgba(3, 168, 158, 0.08)' } },
                  { yAxis: 1200 }
                ],
                // 4. Expert: 1200 ~ 1399 (#0000FF, blue)
                [
                  { yAxis: 1200, itemStyle: { color: 'rgba(0, 0, 255, 0.07)' } },
                  { yAxis: 1400 }
                ],
                // 5. Candidate Master: 1400 ~ 1599 (#AA00AA, purple)
                [
                  { yAxis: 1400, itemStyle: { color: 'rgba(170, 0, 170, 0.08)' } },
                  { yAxis: 1600 }
                ],
                // 6. Master: 1600 ~ 1799 (#FF8C00, orange)
                [
                  { yAxis: 1600, itemStyle: { color: 'rgba(255, 140, 0, 0.08)' } },
                  { yAxis: 1800 }
                ],
                // 7. Grandmaster: >= 1800 (#FF0000, red)
                [
                  { yAxis: 1800, itemStyle: { color: 'rgba(255, 0, 0, 0.08)' } },
                  { yAxis: yMax }
                ]
              ]
            }
          }
        ]
      };

      chartInstance.setOption(option);
    }

    // Modal Opening & Closing
    function openUserModal(userOrId) {
      let user = null;
      if (typeof userOrId === 'string' || typeof userOrId === 'number') {
        user = usersMap.value.get(String(userOrId)) || null;
      } else {
        user = userOrId;
      }

      if (!user) return;

      selectedUser.value = user;
      showModal.value = true;

      nextTick(() => {
        setTimeout(() => {
          renderChart(user);
        }, 60);
      });
    }

    function closeUserModal() {
      showModal.value = false;
      if (chartInstance) {
        chartInstance.dispose();
        chartInstance = null;
      }
      selectedUser.value = null;
    }

    // Resize handler
    function handleResize() {
      if (chartInstance) {
        chartInstance.resize();
      }
    }

    // Keyboard navigation (Esc to close modal)
    function handleKeyDown(e) {
      if (e.key === 'Escape' && showModal.value) {
        closeUserModal();
      }
    }

    onMounted(() => {
      loadData();
      window.addEventListener('resize', handleResize);
      window.addEventListener('keydown', handleKeyDown);
    });

    onUnmounted(() => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('keydown', handleKeyDown);
      if (chartInstance) {
        chartInstance.dispose();
      }
    });

    return {
      activeTab,
      loading,
      errorMessage,
      db,
      TIERS,
      getTier,
      getDisplayOffset,
      statusFilter,
      gradeFilter,
      searchQuery,
      showModal,
      selectedUser,
      usersMap,
      grades,
      summaryStats,
      filteredUsers,
      contestCategories,
      contestType,
      contestsByType,
      visibleContests,
      toggleContest,
      isContestExpanded,
      contestCache,
      isContestLoading,
      getContestError,
      loadContestStandings,
      formatDate,
      formatDateTime,
      getBestPerf,
      getUserContestStats,
      openUserModal,
      closeUserModal,
      loadData
    };
  }
});

app.mount('#app');
