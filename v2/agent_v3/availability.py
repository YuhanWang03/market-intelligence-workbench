"""Labels for explicit capability IDs; no natural-language routing."""
LABELS = {
    "macro.release": "宏观数据发布查询", "macro.overview": "宏观面板",
    "account.earnings_schedule": "账户财报日历", "institutional.manager_portfolio": "基金经理持仓",
    "etf.ark_activity": "ARK 交易活动", "etf.holdings": "ETF 成分持仓及权重",
    "lab.backtest": "策略回测", "lab.sweep": "策略参数扫描", "lab.screen": "量化筛选",
    "lab.event_study": "事件研究", "lab.committee": "实验室评审",
    "state.mutate": "关注列表和提醒修改", "state.read": "用户状态读取",
    "account.portfolio": "账户持仓查询", "account.performance": "账户收益查询", "account.risk": "账户风险查询",
}


def unavailable(names):
    names = list(dict.fromkeys(names))
    labels = "、".join(LABELS.get(name, name) for name in names)
    return {"status": "partial", "stop_reason": "capability_unavailable",
            "answer": f"当前 V3 尚未接入或启用：{labels}。本次无法执行这项查询，也不会生成相应结果。",
            "report": {"ok": False, "warnings": ["Unavailable capabilities: " + ", ".join(names)]}}
