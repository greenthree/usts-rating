# USTS ACM/XCPC 竞赛融合 Rating 积分系统

> **苏州科技大学（USTS）ACM/XCPC 集训队校内外竞赛融合 Rating 积分系统**  
> 采用 **Data-as-Code（数据即代码）** 架构构建，依托 **GitHub Actions + GitHub Pages** 实现全自动数据校验、NumPy 向量化 Elo 算分及纯静态前端看板展示，零后端服务器依赖，开箱即用。

---

## 目录 (Table of Contents)

- [1. 项目背景与系统介绍](#1-项目背景与系统介绍)
  - [1.1 业务背景与痛点](#11-业务背景与痛点)
  - [1.2 Data-as-Code 架构哲学](#12-data-as-code-架构哲学)
  - [1.3 纯静态无后端技术方案](#13-纯静态无后端技术方案)
- [2. 核心机制与算法特性](#2-核心机制与算法特性)
  - [2.1 双轨制积分体系 (True Rating vs Display Rating)](#21-双轨制积分体系-true-rating-vs-display-rating)
  - [2.2 多人体量战力合成公式 ($R_{team}$)](#22-多人体量战力合成公式-r_team)
  - [2.3 多人 Elo 结算与二分表现分查找](#23-多人-elo-结算与二分表现分查找)
  - [2.4 零题掉分保护机制 (0-Solve Floor)](#24-零题掉分保护机制-0-solve-floor)
  - [2.5 校内赛零和校准机制 (Zero-Sum Drift Calibration)](#25-校内赛零和校准机制-zero-sum-drift-calibration)
  - [2.6 Codeforces 风格七阶段位徽章体系](#26-codeforces-风格七阶段位徽章体系)
  - [2.7 两年未参赛自动退役判定规则](#27-两年未参赛自动退役判定规则)
- [3. 目录结构说明](#3-目录结构说明)
- [4. 本地开发与运行指南](#4-本地开发与运行指南)
  - [4.1 环境准备](#41-环境准备)
  - [4.2 数据完整性校验](#42-数据完整性校验)
  - [4.3 自动化测试套件执行](#43-自动化测试套件执行)
  - [4.4 积分数据库全量重新计算](#44-积分数据库全量重新计算)
  - [4.5 本地静态服务启动与预览](#45-本地静态服务启动与预览)
- [5. 数据录入与比赛导入指南](#5-数据录入与比赛导入指南)
  - [5.1 选手花名册录入规范 (`users.csv`)](#51-选手花名册录入规范-userscsv)
  - [5.2 比赛榜单 JSON 格式规范 (`contests/*.json`)](#52-比赛榜单-json-格式规范-contestsjson)
  - [5.3 榜单多源格式转换脚本 (`import_contest.py`)](#53-榜单多源格式转换脚本-import_contestpy)
- [6. GitHub Actions 自动化部署流水线](#6-github-actions-自动化部署流水线)
  - [6.1 持续集成与部署架构](#61-持续集成与部署架构)
  - [6.2 双重质量门禁机制](#62-双重质量门禁机制)
- [7. GitHub Pages 开启与配置指南](#7-github-pages-开启与配置指南)

---

## 1. 项目背景与系统介绍

### 1.1 业务背景与痛点
高校程序设计竞赛（ACM-ICPC、CCPC、天梯赛、省赛及校内选拔赛等）具有高度竞技性与协同性特点。然而在高校集训队日常管理与梯队选拔中，长期面临以下三大痛点：
1. **个人赛与团队赛评价标准割裂**：校内训练多以个人周赛为主，而区域赛均为三人成队。传统积分系统（如经典 Elo 或纯场次累加）无法在单人战力与团队合力之间建立科学公允的数学换算。
2. **校内赛与校外赛基准脱节**：校内赛选手水平相对封闭，长期内部结算极易产生系统性通胀或通缩；校外区域赛对手强劲，缺少外部锚点容易导致校内外打分严重失真。
3. **运维与基础设施成本高**：传统全栈管理系统需要采购云服务器、维护 MySQL/PostgreSQL 数据库、处理后端安全漏洞，长期运维极易因集训队队长换届而废弃。

### 1.2 Data-as-Code 架构哲学
本系统采用 **Data-as-Code（数据即代码）** 现代工程理念，将所有选手花名册与历史比赛数据作为核心代码资产对待：
- **版本受控**：选手名单（CSV）与每场比赛榜单（JSON）全部以纯文本形式纳入 Git 版本控制，每一次榜单导入、信息更正都有据可查，支持通过 `git blame` 与 `git log` 追溯历史。
- **不可变审计**：严禁在生产端直接修改计算结果；每一次变更均由 Git Commit 或 Pull Request 驱动，经过 CI 流水线全自动重新计算衍生出最终展示数据库。
- **纯文本自治**：格式采用通用透明的 CSV 与 JSON，无私有数据库二进制锁死，方便集训队教练与管理人员通过 Excel、文本编辑器或 Python 批量录入管理。

### 1.3 纯静态无后端技术方案
- **静态前端看板**：基于原生 HTML5 + Tailwind CSS (Play CDN) + Vue 3 (CDN) + Apache ECharts (CDN)，彻底剔除 Node.js、Vite、Webpack 等前端构建工具链与 `node_modules` 依赖膨胀。
- **极速分发**：构建产物直接为静态 HTML/JS/CSS 与 JSON 数据文件，通过 GitHub Pages 免费全球 CDN 分发，具备超强高并发访问与零运维成本特性。

---

## 2. 核心机制与算法特性

本系统的数学模型由针对竞赛特性的 Multi-player Elo 算法演化而来，兼具公平性、防小号虚高机制以及团队协作评估能力。

### 2.1 双轨制积分体系 (True Rating vs Display Rating)
为防止新队员单场超常发挥后积分瞬间虚高（小号刷分），并激励队员积极、持续地参加训练，系统设计了**双轨制积分架构**：
- **真实战力分 ($R_{true}$)**：系统底层运算完全依据 $R_{true}$。初始无任何外部公开发表记录的新选手默认赋予基准值 $R_{true} = 1000.0$。若在 `users.csv` 中登记了历史外部权威天梯分（如公开 XCPC-Elo），则采用该初始分。
- **展示积分 ($R_{display}$)**：前端展示与排行榜排名的依据，由真实分加上平滑偏移量获得：
  $$R_{display} = \max(0, \text{round}(R_{true} + \text{offset}(\text{rated\_count})))$$

平滑偏移量根据选手累计参加的有效场次 $\text{rated\_count}$ 阶梯递进：

| 累计参赛场次 ($\text{rated\_count}$) | 偏移量 ($\text{offset}$) | 初始真实分 1000 对应的展示分 | 说明 |
| :---: | :---: | :---: | :--- |
| **0 场** | **-1000** | **0 分** | 未参加任何比赛，展示为 0 分（萌新） |
| **1 场** | **-650** | **350 分** | 完成初次定级赛，开始获得基础分值 |
| **2 场** | **-400** | **600 分** | 渐进过渡 |
| **3 场** | **-200** | **800 分** | 积分大幅释放 |
| **4 场** | **-80** | **920 分** | 接近真实水平 |
| **$\ge$ 5 场** | **0** | **1000 分** | 偏移量完全归零，完全展现个人真实实力 |

---

### 2.2 多人体量战力合成公式 ($R_{team}$)
在多人组队比赛中，队伍的综合实力并非成员个人能力的简单线性算术平均，而是呈现非线性的**木桶效应与长板效应交织**。系统采用基于 LogSumExp 的战力合成模型：

对于人数为 $k \ge 1$ 的校内队伍：
$$R_{team} = 400 \cdot \log_{10}\left(\sum_{i=1}^k 10^{R_i / 400}\right) - C_k$$

为保障数值稳定性，避免计算 $10^{R_i/400}$ 时在大分值下出现浮点溢出，算法内部采用 LogSumExp 标准形式进行向量化计算：
$$M = \max_{1 \le i \le k} R_i, \quad R_{team} = M + 400 \cdot \log_{10}\left(\sum_{i=1}^k 10^{(R_i - M)/400}\right) - C_k$$

体量折减常数体系 $C_k$：
- **单人队 ($k=1$)**：$C_1 = 0.0$（队伍实力完全等于个人战力：$R_{team} = R_1$）
- **双人队 ($k=2$)**：$C_2 = 100.0$
- **三人队 ($k=3$)**：$C_3 = 160.0$
- **超大规模队 ($k > 3$)**：$C_k = 400 \cdot \log_{10}(k) \times 0.84$

> **外部队伍动态演化与全国真实 Elo 追踪（Persistent External Elo）**：
> 为彻底解决传统积分系统中「外校队伍一律固定 1500 分，导致清华北大等 3000 分顶尖队伍被严重低估、本校队伍即使战胜高水平队伍也难以加分」的系统性失真问题，本系统实现了**全生命周期外部战力追踪引擎**：
> 1. **官方 XCPC-Elo 基准无缝对齐**：系统通过 `scripts/sync_xcpc_elo.py` 自动拉取权威开源项目 [Zzzzzzyt/xcpc-elo](https://github.com/Zzzzzzyt/xcpc-elo) 的全国积分数据库，收录 1,676 所高校/机构的战力基准（清华大学 ~3150、北京大学 ~3171、上海交通大学 ~3097、浙江大学 ~2862、复旦大学 ~2773、哈尔滨工业大学 ~2383 等）。
> 2. **跨赛事连续演化（Continuous Cross-Contest Evolution）**：外部队伍在多场区域赛、省赛中连续出战时，上一场赛后的最新分数将自动作为下一场的初始战力，随比赛结果动态成长，在几百场真实榜单重放中自然达到 2500~3200+ 的真实水平，与全国 XCPC-Elo 大盘高度一致。
> 3. **校内队员真实对抗体验**：当本校（USTS）队伍参加 ICPC/CCPC 区域赛时，面对的是真实 2000~3000 分的外部强队，打出优异名次将获得符合其高难度含金量的 Elo 涨幅；而在校内训练赛中，零和校准（Zero-Sum Drift）则确保校内生态不发生通胀或通缩。

---

### 2.3 多人 Elo 结算与二分表现分查找
每场比赛完成后，算分引擎执行以下标准化结算流程：
1. **实际排名确定**：按队伍解题数降序、罚时升序排序。对于完全相同的解题数和罚时，采用并列名次的算术平均值（Fractional Ranks）。
2. **期望胜率与种子排名 (Expected Seed)**：队伍 $B$ 击败队伍 $A$ 的概率为：
   $$P(B > A) = \frac{1}{1 + 10^{(R_A - R_B)/400}}$$
   队伍 $A$ 的期望种子排名为：
   $$\text{Seed}_A = 1 + \sum_{B \neq A} P(B > A)$$
3. **目标名次确定 (Geometric Mean Target)**：
   结合赛前预期名次与实际名次，计算几何平均目标排名：
   $$m_A = \sqrt{\text{Seed}_A \times \text{Rank}_A}$$
4. **二分查找表现分 ($R_{perf}$)**：
   在区间 $[-500.0, 3500.0]$ 内通过 50 次高精度二分迭代，找到一个虚拟表现分 $R_{perf}$，使得若将队伍 $A$ 的战力替换为 $R_{perf}$ 时，队伍 $A$ 面对所有对手的期望名次恰好收敛至 $m_A$。
5. **队伍基础变动分**：
   $$\Delta R_{team} = W_{contest} \times \frac{R_{perf} - R_{team}}{2}$$
   其中比赛权重系数 $W_{contest}$：
   - **校内赛 (`INTERNAL_SCHOOL`)**：$W_{contest} = 0.80$
   - **校外赛 (`EXTERNAL_XCPC`)**：$W_{contest} = 1.00$

---

### 2.4 零题掉分保护机制 (0-Solve Floor)
为了保护初学者和偶发失误队员的参赛积极性，避免因零题签到失败遭受毁灭性扣分，系统严格执行**零题掉分保底保护**：
- 若队伍解题数 $\text{solved} == 0$，其单场扣分幅度下限为 $-60.0$ 分：
  $$\Delta R_{team} = \max(-60.0, \Delta R_{team})$$
- 团队所有队员分配到的单场扣分均受此下限保护（全队同奖同惩）。

---

### 2.5 校内赛零和校准机制 (Zero-Sum Drift Calibration)
在校内赛（`contest_type: "INTERNAL_SCHOOL"`）中，比赛完全在集训队内部进行。若不对全场漂移进行校准，可能因整体排名的非对称性导致整个集训队的平均分发生系统性膨胀或收缩。

为此，引擎在校内赛中引入**零和均值校准**：
1. 队伍各队员获得其队伍变动分 $\Delta R_i = \Delta R_{team}$。
2. 统计所有参赛本校选手（共 $M$ 人）的均值漂移：
   $$\mu = \frac{1}{M} \sum_{i=1}^M \Delta R_i$$
3. 对所有校内选手扣除均值漂移进行校准：
   $$\Delta R_i \leftarrow \Delta R_i - \mu$$
4. 校准后满足：
   $$\left|\sum_{i=1}^M \Delta R_i\right| \le 10^{-6}$$

> **注意**：在校外赛（`EXTERNAL_XCPC`）中，本校队伍与外部锚点队伍同台竞技，本校队伍的总分池允许因战胜外部强队而获得净增长（或因失利而净减少），因此校外赛不执行零和校准。

---

### 2.6 Codeforces 风格七阶段位徽章体系
系统依据选手的展示分 $R_{display}$，划分为 Codeforces 风格的 7 个段位体系。从 800 分开始，每 200 分作为一个段位：

| 段位英文标识 | 中文段位称号 | 展示积分区间 ($R_{display}$) | 官方视觉色值 | 浅色背景底色 | 边框色值 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Newbie** | **萌新** | `0 ~ 799` | `#808080` (灰色) | `#F3F4F6` | `#D1D5DB` |
| **Pupil** | **进阶** | `800 ~ 999` | `#008000` (绿色) | `#DCFCE7` | `#86EFAC` |
| **Specialist** | **骨干** | `1000 ~ 1199` | `#03A89E` (青色) | `#CCFBF1` | `#5EEAD4` |
| **Expert** | **精英** | `1200 ~ 1399` | `#0000FF` (蓝色) | `#DBEAFE` | `#93C5FD` |
| **Candidate Master** | **候选大师** | `1400 ~ 1599` | `#AA00AA` (紫色) | `#F3E8FF` | `#D8B4FE` |
| **Master** | **校队大师** | `1600 ~ 1799` | `#FF8C00` (橙色) | `#FFEDD5` | `#FDBA74` |
| **Grandmaster** | **传奇宗师** | `≥ 1800` | `#FF0000` (红色) | `#FEE2E2` | `#FCA5A5` |

- 前端姓名颜色、徽章底色与个人主页折线图背景色块严格对齐上述色系标准。

---

### 2.7 两年未参赛自动退役判定规则
为了让排行榜客观反映当前集训队的现役实力结构，系统建立自动退役规则：
- **判定条件**：
  1. 选手最近一次参加有效计分比赛的时间距今超过 **2 年（730.5 天）**；或
  2. 选手入队注册年份满 **2 年以上且参赛记录为 0 场**；或
  3. 选手在 `users.csv` 中被管理员显式标记为 `is_active: false`。
- **展示逻辑**：
  - 前端总排行榜默认开启「仅现役」筛选，退役选手默认隐藏；
  - 提供「现役 / 退役 / 全部」切换按钮，方便回溯已毕业学长学姐的历史荣誉。

---

## 3. 目录结构说明

```text
usts-rating/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions: 自动化数据校验、测试、全量算分与 Pages 部署流水线
├── data/                       # 核心数据目录（受 Git 版本控制）
│   ├── users.csv               # 校内选手花名册（学号、姓名、年级、外部基准分、现役状态）
│   └── contests/               # 标准化比赛历史榜单（按 YYYY_MM_DD 命名时序排列）
│       ├── 2026_09_15_school_solo_1.json      # 样例文档：校内个人热身赛
│       ├── 2026_10_10_school_team_1.json      # 样例文档：校内三人组队对抗赛
│       └── 2026_10_25_icpc_regional_mock.json # 样例文档：ICPC 区域模拟赛（含校外锚点）
├── scripts/                    # 工具链与核心算分引擎
│   ├── recalculate.py          # 核心算分引擎：NumPy 向量化运算，输出 rating_db.json
│   ├── validate_data.py        # 严格数据校验脚本：检查学号冲突、参照完整性与 JSON 格式
│   └── import_contest.py       # 榜单转换工具：支持 DOMjudge JSON / PTA / CSV 导入为标准 JSON
├── tests/                      # 自动化测试套件（覆盖率 100%）
│   ├── fixtures/               # 测试固件（样例 CSV、DOMjudge、PTA 原始榜单数据）
│   ├── test_elo_engine.py      # 核心算法单元测试（测试胜率、LogSumExp合成、零和校准等）
│   ├── test_validation_tooling.py # 数据校验与导入工具测试
│   └── test_e2e_suite.py       # 4-Tier 深度端到端集成测试（覆盖边界、并发与全真管道）
├── web/                        # 纯静态前端看板（零构建依赖，开箱即用）
│   ├── index.html              # 主应用入口（排行榜、赛事列表、个人详情弹窗、规则解析）
│   ├── app.js                  # Vue 3 核心交互、段位渲染与 Apache ECharts 可视化逻辑
│   ├── style.css               # 自定义视觉层（滚动条、段位色调微调与弹窗动画）
│   └── data/
│       └── rating_db.json      # 算分引擎生成的全量静态数据包
├── requirements.txt            # Python 环境依赖规范
└── README.md                   # 系统全量中文技术与操作说明文档
```

---

## 4. 本地开发与运行指南

### 4.1 环境准备
系统仅依赖标准 Python 3.11+ 运行环境：
```bash
# 1. 验证 Python 版本（推荐 3.11 或更高）
python --version

# 2. 安装项目依赖
pip install -r requirements.txt
```

`requirements.txt` 保持绝对精简：
- `numpy>=1.24.0`：用于 Elo 胜率矩阵计算与二分法向量化加速。
- `pytest>=7.4.0`：用于执行全量自动化测试套件。

---

### 4.2 数据完整性校验
在任何数据更新后，执行数据校验脚本：
```bash
python scripts/validate_data.py --strict
```
- 参数说明：
  - `--users`：指定花名册路径（默认 `data/users.csv`）。
  - `--contests`：指定比赛目录或单个 JSON 文件（默认 `data/contests`）。
  - `--strict`：启用严格模式，遇到任何警告或错误均返回非零退出码。

---

### 4.3 自动化测试套件执行
运行仓库全量自动化单元测试与端到端测试：
```bash
# 运行全部 102 个测试用例
pytest tests/ -v

# 仅运行端到端 E2E 套件
pytest tests/test_e2e_suite.py -v

# 仅运行核心算分模型单元测试
pytest tests/test_elo_engine.py -v
```

---

### 4.4 积分数据库全量重新计算
根据现有的所有历史比赛，按时间正序重新运算全量积分流水并输出 `rating_db.json`：
```bash
python scripts/recalculate.py
```
默认会在 `web/data/rating_db.json` 与 `dist/data/rating_db.json` 中同步生成聚合数据库。

---

### 4.5 本地静态服务启动与预览
无需安装 Node.js 或任何前端构建工具，直接启动 Python 原生 HTTP 服务：
```bash
# 在项目根目录下启动本地静态服务器
python -m http.server 8000
```
在浏览器中打开：
```text
http://localhost:8000/web/
```
即可完整体验总排行榜、搜索筛选、比赛记录展开以及选手详情折线图。

---

## 5. 数据录入与比赛导入指南

### 5.1 选手花名册录入规范 (`users.csv`)
文件位于 `data/users.csv`，编码必须为 UTF-8（允许带 BOM）。

#### 表头与字段规范
```csv
student_id,name,grade,initial_external_elo,is_active
```
| 字段名 | 类型 | 必填 | 说明与约束 |
| :--- | :---: | :---: | :--- |
| `student_id` | String | 是 | 学号主键，必须全表唯一，仅含字母数字下划线 |
| `name` | String | 是 | 选手真实姓名，支持中文，不可为空 |
| `grade` | Integer | 是 | 入学年份四位整数（例如 `2023`、`2024`） |
| `initial_external_elo` | Float/Int | 否 | 外部公开基准分，若无记录可留空，引擎将自动赋予初始 1000.0 分 |
| `is_active` | Boolean | 是 | 管理员状态标记，填 `true`、`false`、`1` 或 `0` |

#### 录入示例
```csv
student_id,name,grade,initial_external_elo,is_active
21001001,张三,2021,1420,true
22002002,李四,2022,,true
23003003,王五,2023,1200,true
24004004,赵六,2024,,true
```

---

### 5.2 比赛榜单 JSON 格式规范 (`contests/*.json`)
新比赛文件需放入 `data/contests/` 目录下，文件命名遵循 `YYYY_MM_DD_contest_name.json` 规范。

#### 标准 JSON 字段定义
```json
{
  "contest_id": "2026_11_15_school_autumn_cup",
  "title": "2026 苏州科技大学秋季选拔赛",
  "contest_type": "INTERNAL_SCHOOL",
  "start_time": "2026-11-15T13:00:00+08:00",
  "standings": [
    {
      "team_id": "team_alpha",
      "team_name": "阿尔法小队",
      "is_school": true,
      "members": [
        "22002002",
        "23003003"
      ],
      "solved": 6,
      "penalty": 420
    },
    {
      "team_id": "ext_team_njup",
      "team_name": "南京邮电大学特邀队",
      "is_school": false,
      "external_elo": 1650.0,
      "members": [],
      "solved": 7,
      "penalty": 390
    }
  ]
}
```

- `contest_type` 仅允许两类枚举：
  - `"INTERNAL_SCHOOL"`：校内赛，权重 0.80，实施零和均值漂移校准；
  - `"EXTERNAL_XCPC"`：校外赛，权重 1.00，外部队伍充当基准锚点。
- `is_school` 为 `true` 时，`members` 内列出的每一个学号必须已存在于 `data/users.csv` 中。
- `is_school` 为 `false` 时为外部队伍，可不填写 `members`，可通过 `external_elo` 指定其实力锚点（缺省为 1500.0）。

---

### 5.3 榜单多源格式转换脚本 (`import_contest.py`)
为降低从各类判题系统录入比赛的门槛，系统提供了 `scripts/import_contest.py` 转换工具。

```bash
python scripts/import_contest.py --help
```

#### 场景 1：从 CSV 格式成绩单导入
若主办方提供 Excel 导出的 CSV 成绩单（包含队伍、解题数、罚时、队员学号等信息）：
```bash
python scripts/import_contest.py \
  --input raw_scoreboard.csv \
  --output data/contests/2026_11_20_warmup.json \
  --contest-id 2026_11_20_warmup \
  --title "2026 校队热身赛" \
  --type INTERNAL_SCHOOL \
  --start-time "2026-11-20T13:00:00+08:00"
```

#### 场景 2：从 DOMjudge 导出的 JSON 榜单导入
线下 ICPC/CCPC 赛点通常使用 DOMjudge 评测机，可导出 `scoreboard.json` 与 `teams.json`：
```bash
python scripts/import_contest.py \
  --input domjudge_scoreboard.json \
  --teams domjudge_teams.json \
  --output data/contests/2026_11_25_ccpc_mock.json \
  --contest-id 2026_11_25_ccpc_mock \
  --title "2026 CCPC 模拟排位赛" \
  --type EXTERNAL_XCPC \
  --school-keywords "苏州科技大学,USTS"
```

#### 场景 3：从 PTA (拼题 A) 导出的榜单导入
```bash
python scripts/import_contest.py \
  --input pta_export.json \
  --output data/contests/2026_12_01_gplt_school.json \
  --contest-id 2026_12_01_gplt_school \
  --title "2026 团体程序设计天梯赛校内选拔赛" \
  --type INTERNAL_SCHOOL
```

#### 场景 4：从开源生态 algoux/srk-collection 与 XCPC-Elo 导入标准榜单 (.srk.json)
全国各大 ICPC 区域赛、CCPC 分站赛及江苏省赛（JSCPC）等赛事在开源榜单库 `algoux/srk-collection`（RankLand 平台）中均有公开标准 JSON 存档：
```bash
# 一键导入江苏省赛（JSCPC）或 ICPC 区域赛榜单：
# 脚本会自动识别高校名称（"苏州科技大学"/"苏科大"）及花名册队员姓名（如"喻文州" -> 23001003），
# 过滤教练信息，并将各兄弟高校队伍自动转换为校外实力锚点（默认 1500 分或指定基准分）：
python scripts/import_contest.py \
  --input jscpc10th.srk.json \
  --output data/contests/2025_06_02_jscpc_10th.json \
  --format srk \
  --type EXTERNAL_XCPC \
  --school-keywords "苏州科技大学,USTS,苏科大"
```

#### 场景 5：同步全国官方 XCPC-Elo 战力基准 (`sync_xcpc_elo.py`)
系统支持直接从官方项目 [zzzzzzyt.github.io/xcpc-elo](https://zzzzzzyt.github.io/xcpc-elo/) 实时拉取全国数万名选手的真实 Elo 积分与各大高校梯队战力：
```bash
# 一键同步全国高校战力与选手 Elo 映射（生成 data/external_elo_map.json）
# 同时自动筛选苏州科技大学（USTS）全部 53 位真实参赛队员历史（导出至 data/users_xcpc_history.csv）
python scripts/sync_xcpc_elo.py
```
同步后：
- `data/external_elo_map.json`：包含 1,676 所高校基准（如清华 3150、北大 3171、交大 3097 等），导入任何校外赛事时外校队伍将直接获得精准定级！
- `data/users_xcpc_history.csv`：包含陈家杰 (1479)、江帆 (1476)、贺鹏飞 (1464)、王永刚 (1432) 等本校真实选手的全国官方积分，可直接作为花名册的参考依据。

#### 场景 6：批量导入 algoux/srk-collection 历史赛季全部榜单 (`import_srk_batch.py`)
如果您克隆了 `algoux/srk-collection`（或下载了历史多场 `.srk.json` 榜单），可以一键批量导入整季比赛：
```bash
# 批量将某个目录下的全部 .srk.json 转换为 USTS 标准比赛 JSON：
python scripts/import_srk_batch.py /path/to/srk-collection/contests/
```
导入完成后运行 `python scripts/recalculate.py`，引擎将按照时间线全量重放，使外校强队分数随比赛自然流动，与全国官方 XCPC-Elo 大盘完全同构！

---

## 6. GitHub Actions 自动化部署流水线

### 6.1 持续集成与部署架构
流水线定义于 `.github/workflows/deploy.yml`，执行架构如下图所示：

```text
[ Git Push to main ]  或  [ 手动触发 workflow_dispatch ]
          │
          ▼
┌───────────────────────────────────────────────┐
│ Job 1: build-and-test                         │
│  ├─ 1. actions/checkout@v4                    │
│  ├─ 2. actions/setup-python@v5 (Python 3.11)  │
│  ├─ 3. pip install -r requirements.txt        │
│  ├─ 4. [门禁1] python scripts/validate_data.py│
│  ├─ 5. [门禁2] pytest tests/ -v               │
│  ├─ 6. [算分] python scripts/recalculate.py   │
│  ├─ 7. 打包静态分发包 (web + data -> dist)    │
│  └─ 8. actions/upload-pages-artifact@v3       │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ Job 2: deploy                                 │
│  └─ actions/deploy-pages@v4                   │
│     └──> 自动发布至 GitHub Pages CDN          │
└───────────────────────────────────────────────┘
```

### 6.2 双重质量门禁机制
流水线具备严格的质量防线，保证任何格式瑕疵或逻辑错误均不会污染线上页面：
1. **数据防线 (`validate_data.py`)**：在算分开始前，检查全量 CSV 与 JSON 数据。如果出现未在花名册注册的外校学号混入校内队、学号格式错误、负数解题数或文件名与比赛日期不一致，立即熔断并阻断流水线。
2. **算法防线 (`pytest tests/`)**：执行包括 70 项端到端测试与全部算法单元测试，确保零和均值校准、零题保底、段位映射无任何代码回归。

---

## 7. GitHub Pages 开启与配置指南

当您将本项目 Fork 或新建到 GitHub 组织/个人账号下时，仅需一次性开启 GitHub Pages 配置：

1. **进入仓库设置**：  
   打开您的 GitHub 仓库主页，点击顶部导航栏中的 **Settings**（设置）。
2. **找到 Pages 选项**：  
   在左侧功能导航栏中，点击 **Pages**（位于 Code and automation 分类下）。
3. **切换构建来源 (Source)**：  
   在页面中央的 **Build and deployment** 区域，找到 **Source** 下拉菜单，将其从默认的 `Deploy from a branch` 修改为 **`GitHub Actions`**。
   > ⚠️ **重要**：切勿选择指定分支（如 gh-pages 分支），本项目使用现代化 GitHub Actions 原生工件部署，无需污染仓库分支历史。
4. **触发初次部署**：  
   - 方式 A：进入仓库顶部的 **Actions** 标签页，选择左侧的 `Deploy USTS Rating System to GitHub Pages`，点击右侧的 **Run workflow** 手动触发一次。
   - 方式 B：向 `main` 分支提交一次包含新比赛或花名册的 `git push`。
5. **获取线上访问链接**：  
   当流水线执行完毕（全部打绿勾）后，刷新 Pages 设置页面，顶部将显示您的专属域名：
   ```text
   https://<your-username-or-org>.github.io/<repo-name>/
   ```
   点击链接即可在全网任何设备上流畅浏览 USTS ACM/XCPC Rating 积分榜！

---

## 许可证 (License)

本项目遵循 MIT 开源许可证。欢迎各高校程序设计竞赛集训队交流使用。
