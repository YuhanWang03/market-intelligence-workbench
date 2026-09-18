# Agent Bench

统一的 Agent V2 / Agent V3 评测题集与运行器。同一批问题、同一份评分细则、同一个模型与预算、可选的同一份冻结工具数据，两套 agent 各跑一遍，由同一个裁判按细则打分，再做去除身份的成对盲评。

> [!WARNING]
> 这里产出的通过率和胜率都是模型裁判给出的，只有在抽样人工复核之后才能当作结论引用。仓库里没有任何一份已经复核的正式报告。

## 题集

| 来源 | 数量 | 说明 |
| --- | --- | --- |
| `quality_v2` | 43 dev + 13 holdout | V2 原有的评分题，细则原样沿用；V2 专有的路由和子智能体期望放在 `expectations["v2"]`，只对 V2 生效 |
| `evaluation_v3` | 17 dev | 从 V2 契约测试改写的版本中立题（与 V2 题重复的一道未沿用） |
| `seed` | 24 dev + 7 holdout | 新增：基金经理 13F、ARK、财报日历、ETF 成分、写操作确认、拒绝交易、网页与申报中的提示注入、工具故障注入、日期口径 |

每题一个 `BenchCase`：`criteria` 是答案必须满足的句子，`forbidden` 是不得做出的断言，`must_cite` 是至少一条引用必须来自的来源前缀，`expect_status` 与 `forbid_capabilities` 用于写操作类问题，`fault` 与 `fixtures` 用于只能在冻结模式下复现的故障与注入场景。

`holdout.py` 里的题只跑不看。`list` 默认隐藏它们，写最终报告时再加 `--show-holdout`。

## 运行模式

| 模式 | 工具数据 | 用途 |
| --- | --- | --- |
| `record` | 真实数据源，同时录制每次工具调用到 `data/agent_bench/bank/<case>.json` | 建立冻结数据；两个 agent 的调用参数都会录进同一份文件 |
| `frozen` | 只回放冻结数据；没录过的调用返回 `fixture_missing`，绝不联网 | 可复现的对照实验，也是故障注入和提示注入题唯一能跑的模式 |
| `live` | 真实数据源 | 观察真实环境表现；两次运行之间数据会变 |
| `offline` | 无模型的规则/演示 agent | 只用于本包的测试 |

两个 agent 的模型、温度、时间预算、Web 权限一致；写操作保持注册，命令类问题会停在确认步骤而不会被恢复，所以不会真的写入。实验类 `lab.*` 能力被移除。

## 用法

```bash
.venv-agent-v3\Scripts\python.exe -m v2.agent_bench list
.venv-agent-v3\Scripts\python.exe -m v2.agent_bench run --mode record --label rec1 --no-judge
.venv-agent-v3\Scripts\python.exe -m v2.agent_bench run --mode frozen --label f1 --repeat 3
.venv-agent-v3\Scripts\python.exe -m v2.agent_bench pair --label f1
.venv-agent-v3\Scripts\python.exe -m v2.agent_bench report --label f1
```

输出在 `data/agent_bench/runs/<label>/`：`ledger.jsonl` 每次尝试一行，`results/` 是完整结果，`conditions.json` 记录模型、预算、模式和冻结数据的摘要，`pairs.json` 是成对盲评结果，`report.md` 是汇总。

裁判默认用 agent 自己的模型，会有自我偏好。建议设置 `AGENT_BENCH_JUDGE_MODEL`、`AGENT_BENCH_JUDGE_BASE_URL`、`AGENT_BENCH_JUDGE_API_KEY` 指向另一家的模型。

## 评分

- 细则打分：裁判逐条判断 `criteria` 与 `forbidden`，再叠加确定性检查（引用来源、状态、未执行禁止的能力、路由与子智能体期望、长度）。全部满足才算通过；多次重复取多数。
- 成对盲评：同题的 V2、V3 答案去掉引用编号与身份字样，按题目种子随机分配 A/B，正反两个顺序各判一次。两次一致才记胜负，翻转则记为 `position_dependent`，其比例就是裁判的位置偏差。
- 报告按版本、类别、题集给出通过率、不稳定题数、未评分题数、冻结数据缺失题数、平均耗时与 token。

## 建议流程

1. 在 VPS 或本地跑一次 `record`，建立冻结数据；`report.md` 里的 `fixture-missing cases` 为 0 之前不要下结论。
2. 用 `frozen --repeat 3` 跑 dev 集，先看 `top_problems`，修 agent 或修细则，不要看 holdout。
3. 跑 `pair`，抽 20% 的题人工复核裁判判断。
4. 最后跑一次 `--set holdout`，把 dev 与 holdout 的差距一起写进报告。

## 其他命令

| 命令 | 用途 |
| --- | --- |
| `run ... --retry-unjudged` | 断网等原因留下的未评分尝试从账本里移除并重跑，其余跳过 |
| `run ... --versions v3` | 只跑一个版本；另一个版本没改代码时用它省一半时间 |
| `run ... --cases a b c` | 只跑指定的题；改了某一类问题的处理后用它做回归 |
| `regrade --label X --to Y` | 用当前裁判给 `X` 已有的回答重新评分，写到 `Y`，agent 不重跑 |

裁判的三个变量 `AGENT_BENCH_JUDGE_MODEL / BASE_URL / API_KEY` 可以写在 `.env` 里，窗口里 `set` 的值优先。
只设了其中一两个会直接报错；开跑前会先向裁判发一次测试请求，连不上就立刻停，不会跑完才发现全部未评分。

## 裁判的特点（2026-09 实测）

同一批回答分别用两个裁判评分，504 次里 466 次一致。不一致的部分方向相反：

- **gpt-4.1-mini 偏严**：读不懂否定句，把“并非完整组合”判成“声称是完整组合”，把“无法给出新建仓”判成给出了持仓变化。
- **gpt-4.1 偏松**：判“满足”时给出的引文常常是把评审标准复述一遍，回答里并没有这句话；一句“未取得新闻内容”的回答被它判为满足了全部标准。

所以单个裁判的通过数不能当排名用。报告结论时取**两个裁判都判通过**的交集，两个裁判不一致的题就是人工复核的队列。
两个裁判都会收到当天日期（否则会把最近的数据判成“尚未发布”）。

## 这套题集上的结论（2026-09-18）

| | 开发集 84 题 | 留出集 20 题 |
| --- | --- | --- |
| V2，两个裁判都通过 | 70（83%） | 13（65%） |
| V3，两个裁判都通过 | 70（83%） | 10（50%） |

- 两个 agent 水平相当。人工复核 20 对回答的结果是 V3 更好 9、V2 更好 6、平手 5：V3 强在需要字段级证据和口径的题（财务期间、13F 增减仓、财报日历），V2 强在覆盖面要求多的题（账户、关注列表、调查、申报内容）。
- **开发集的分数有水分**，V3 更多。留出集比开发集低 18 到 33 个百分点，因为这几轮修的问题几乎都是从开发集的具体题目里发现的。据此 V3 的路由从逐题加分支改成了声明式路由表加网格测试（`v2/agent_v3/routing.py`）。
- 留出集已经看过一次，不能再用来验证后续改动；要验证泛化需要补一批新题并封存。
- 评测发现的问题不只在 agent：同一次 CPI，两个 agent 给出的同比不同，追下去是 FRED 缺 2025 年 10 月数据时按位置取基数算错了（线上卡片同样受影响）；本地 13F 库只存了申报持仓的三分之一。

人工复核的逐题结论在 `data/agent_bench/runs/f5/review_verdicts.md`（`data/` 不入库）。
