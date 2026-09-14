# Agent V3

Agent V3 是 Market Intelligence Workbench 的 LangGraph 智能体运行时。它面向复杂、多步骤、需要持久化、恢复、严格依赖控制和审计的研究任务，通过显式状态图组织分类、计划、执行、搜索、合成、验证、修复、辩论与确认节点。

> [!WARNING]
> **本系统仍在调整过程中，正式测评尚未完成。** 当前的单元测试、契约测试、离线演示、live smoke、acceptance 和 synthetic quality 脚本主要用于开发验证。它们不是统一真实数据集上的正式质量测评，也不能证明 V3 比 V2 更准确或更适合投资决策。

## 设计目标

- 使用显式、可检查的状态图代替隐式长流程。
- 对计划 DAG、任务依赖、fan-out、并发和截止时间设置硬边界。
- 为长任务提供 checkpoint、会话和 job 持久化。
- 在写操作前进入确认节点，并记录 mutation journal。
- 复用共享能力与证据标准，但不调用 Agent V2 执行器。
- 对校验失败提供有限修复、确定性降级和可选对抗审阅。

## 状态图

```text
START
  → classify
  → plan
  → confirmation（需要写操作时）
  → execute
  → web_fallback（显式开启时）
  → synthesize
  → verify
  → repair / fallback / debate
  → finish
  → END
```

实际图定义位于 `graph.py`，仓库同时保留 `architecture.mmd` 便于查看结构。运行时状态使用经过校验的契约对象传递，而不是在节点之间共享任意字典。

## 主要目录与文件

| 路径 | 作用 |
| --- | --- |
| `contracts.py` | V3 请求、语义意图、计划、任务和结果契约 |
| `graph.py` | LangGraph 状态图、节点和条件边 |
| `brain.py` | 分类、计划、合成和审阅所需的模型逻辑 |
| `execution.py` / `tools.py` | DAG 执行、能力注册、边界与取消控制 |
| `runtime.py` | 工作台数据源、模型、持久化和图运行时组装 |
| `persistence.py` / `jobs.py` | checkpoint、会话、后台任务和恢复 |
| `context.py` / `page_context.py` | 会话和网页上下文解析 |
| `research.py` / `market.py` / `sec.py` | 研究、行情和 SEC 证据处理 |
| `portfolio_*.py` | 组合概览、风险分析和验收支持 |
| `news_research.py` / `specialists.py` | 新闻检索及受 schema 约束的专家流程 |
| `tests/` | 单元、契约、边界和回归测试 |
| `*_acceptance.py` / `*_eval.py` / `live_smoke.py` | 开发期验收和评测脚手架；不是正式测评报告 |

## 环境与运行

Agent V3 建议使用独立虚拟环境，以免 LangGraph 依赖影响主后端：

```bash
python -m venv .venv-agent-v3
.venv-agent-v3/bin/pip install -r v2/agent_v3/requirements.lock
```

Windows 使用：

```powershell
python -m venv .venv-agent-v3
.venv-agent-v3\Scripts\pip.exe install -r v2\agent_v3\requirements.lock
```

不使用 API Key 的离线演示：

```bash
.venv-agent-v3/bin/python -m v2.agent_v3 --demo
```

打印实际编译图的 Mermaid：

```bash
.venv-agent-v3/bin/python -m v2.agent_v3 --graph
```

生产环境通常通过独立 FastAPI 进程提供：

```bash
.venv-agent-v3/bin/python -m uvicorn agent-v3-server:app --app-dir web/deploy --host 127.0.0.1 --port 8104
```

主要接口：

```text
POST /api/agent-v3/ask
GET  /api/agent-v3/jobs/{job_id}
POST /api/agent-v3/jobs/{job_id}/retry
```

模型配置通常使用 `AGENT_V3_MODEL`、`AGENT_V3_BASE_URL`、`AGENT_V3_API_KEY` 和 `AGENT_V3_THINKING`。Web 访问必须通过 `AGENT_V3_WEB_ENABLED` 和单次请求共同允许。不要提交真实密钥、checkpoint、会话或运行日志。

## 测试与现有评测工具

运行 V3 测试：

```bash
.venv-agent-v3/bin/python -m pytest v2/agent_v3/tests v2/agent_common -q -p no:cacheprovider
```

`quality_eval.py` 使用固定合成证据检查部分输出约束，`live_smoke.py` 检查真实运行链路是否可用，`acceptance.py` 和其他 acceptance 脚本检查特定业务路径。这些工具的存在不表示正式测评已经完成；部分脚本也在源码中明确声明其结果不是质量分数。

正式评估 V3 时，应与 V2 使用完全相同的模型、问题集、数据快照、搜索权限、工具预算、超时设置和评分标准，并额外衡量 checkpoint 恢复、依赖失败传播、写操作确认和长任务取消。当前尚未发布这种受控对比报告。

## 当前限制

- LangGraph 与持久化层增加了部署和运维复杂度。
- 节点、状态契约、专家流程和恢复语义仍可能调整。
- 真实数据源失败、限流和时间口径差异仍会影响输出。
- live smoke 或 acceptance 通过只能说明指定链路工作，不代表回答质量达标。
- 长任务可能产生更多模型调用和费用，必须设置预算和超时。
- 真实写操作应默认关闭，并始终经过显式确认与外部风控。

## 与 Agent V2 的关系

V3 复用 `v2/agent_common/` 中的共享语义和部分经过验证的能力适配，但拥有独立的图执行器、会话、任务和持久化机制。V2 仍是网页默认选择，V3 是可选的复杂任务路径。版本号只代表架构迭代，不代表质量排名；在正式测评完成前，两个系统都应视为仍在开发中的研究助手。
