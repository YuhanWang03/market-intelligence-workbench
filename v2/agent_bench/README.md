# Agent Bench

统一的 Agent V2 / Agent V3 评测题集与运行器。同一批问题、同一份评分细则、同一个模型与预算、可选的同一份冻结工具数据，两套 agent 各跑一遍，由同一个裁判按细则打分，再做去除身份的成对盲评。

> [!WARNING]
> 这里产出的通过率和胜率都是模型裁判给出的，只有在抽样人工复核之后才能当作结论引用。仓库里没有任何一份已经复核的正式报告。

## 题集

| 来源 | 数量 | 说明 |
| --- | --- | --- |
| `quality_v2` | 43 dev + 13 holdout | V2 原有的评分题，细则原样沿用；V2 专有的路由和子智能体期望放在 `expectations["v2"]`，只对 V2 生效 |
| `evaluation_v3` | 18 dev | 从 V2 契约测试改写的版本中立题 |
| `seed` | 25 dev + 7 holdout | 新增：基金经理 13F、ARK、财报日历、ETF 成分、写操作确认、拒绝交易、网页与申报中的提示注入、工具故障注入、日期口径 |

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
