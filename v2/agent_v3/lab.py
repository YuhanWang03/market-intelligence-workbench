"""Reuse quantitative runners with isolated V3 history and bounded universes."""
from dataclasses import replace
from v2.agent_v2.adapters.workspace_lab import WorkspaceLabPort, lab_envelope
from v2.agent_v2.models import ToolEnvelope, ResultStatus
from v2.agent_v2.models import EvidenceItem


def percentage_payload(value):
    # Existing engine names these *_pct but returns decimal fractions.
    percentage_keys = {"total_return_pct","annualized_return_pct","max_drawdown_pct","avg_return_pct","return_pct","benchmark_pct","excess_return_pct","win_rate","car_0_1","car_0_5","car_0_20","car_2_20","mean_car","std_car","ci_lower","ci_upper"}
    if isinstance(value,list):
        return [percentage_payload(row) for row in value]
    if isinstance(value,dict):
        return {key:round(number*100,6) if key in percentage_keys and isinstance(number,(int,float)) and not isinstance(number,bool) else percentage_payload(number) for key,number in value.items()}
    return value


class V3LabPort:
    def __init__(self):
        self._bindings = None

    def run(self, capability, arguments, context):
        if self._bindings is None:
            self._bindings = WorkspaceLabPort().bindings
        binding = self._bindings[capability]
        args = dict(arguments)
        # No silent expansion to hundreds of index members on a small server.
        if not args.get("tickers"):
            return ToolEnvelope(capability,ResultStatus.PARTIAL_DATA,limitations=["请明确提供实验股票列表（最多8只），不会默认扫描整个指数。"])
        if len(args["tickers"]) > 8:
            return ToolEnvelope(capability,ResultStatus.PARTIAL_DATA,limitations=["当前V3交互实验最多8只股票；请缩小范围。"])
        fields = binding.input_model.model_fields
        if "universe" in fields:
            args["universe"] = "custom"
        if "source" in fields:
            args["source"] = "tickers"
        body = binding.input_model(**args)
        if capability == "lab.sweep" and len(body.top_ns)*len(body.holding_days_list)*len(body.near_high_pcts) > 12:
            return ToolEnvelope(capability,ResultStatus.PARTIAL_DATA,limitations=["交互参数扫描最多12组；请缩小参数网格。"])
        context.v3_run.check()
        payload = binding.runner(body, on_tick=lambda n:context.v3_run.emit(f"{capability}: {n}")) if binding.supports_progress else binding.runner(body)
        if capability == "lab.event_study":
            payload = {**payload,"n_events":len(payload.get("events",[])), "groups":[{"group":group.get("group") or group.get("source_type"),"window":row["window"],"n_events":row["n_events"],"mean_car":row["mean_car"],"p_value":row["p_value"],"ci_lower":row["ci"]["lower"],"ci_upper":row["ci"]["upper"]} for group in payload.get("aggregates",[]) for row in group.get("windows",[])]}
        result = lab_envelope(capability,percentage_payload(payload),context.run_id)
        result.metadata["percentage_metrics"] = "percentage points representation: 25 means 25%, not 0.25%"
        if hasattr(body,"cost_bps"):
            result.evidence.append(EvidenceItem(f"lab-cost-{context.run_id}",",".join(args["tickers"]),f"回测按单边 {body.cost_bps} 基点计入交易成本；数据接口费用与交易成本不同。",metric="cost_bps",value=body.cost_bps,unit="bps",source_id="backtest_parameters"))
            result.metadata["answer_constraints"]=[{"forbid_claim":"把数据接口费用为零解释为回测未计入交易成本。", "warning":"数据接口费用不等于交易成本；引用cost_bps参数说明成本。"}]
        if hasattr(body,"capital"):
            allocation = {"capital":body.capital, **({"per_trade":body.per_trade} if hasattr(body,"per_trade") else {})}
            result.evidence.append(EvidenceItem(f"lab-allocation-{context.run_id}",",".join(args["tickers"]),f"回测资金参数（美元）：{allocation}。收益按总资金计算，未投入资金仍计入分母。",value=allocation,source_id="backtest_parameters"))
        if capability == "lab.event_study":
            # Statistical group labels are dimensions, not security identities.
            result.evidence = [replace(item, entity=",".join(args["tickers"])) for item in result.evidence]
            result.limitations.append("事件锚点为申报日期，不保证等于实际财报发布时间；CAR及置信区间已转换为百分数，p值未转换。")
            result.status = ResultStatus.PARTIAL_DATA
        if capability == "lab.committee":
            result.limitations.append("委员会是预设投资风格模型的模拟输出，不代表所借鉴投资者本人的意见。")
        return result
