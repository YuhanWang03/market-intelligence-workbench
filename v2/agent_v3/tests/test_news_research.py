import time
import pytest
from v2.agent_v3.news_research import investigated_news,evidence_result,source_rows,resolve_event_date,publisher_family,readable_body
from v2.agent_v3.tests.test_specialists import ScriptedModel,call
from v2.agent_v3.context import RunContext

WINDOW={"start":"2026-09-01","end":"2026-09-13","basis":"publication"}


def test_invalid_structured_news_output_gets_one_schema_retry():
    from langchain_core.messages import AIMessage
    from v2.agent_v3.brain import ModelBrain
    from v2.agent_v3.news_research import SearchPlan
    model=ScriptedModel(responses=[AIMessage(content="not a tool response"),call("SearchPlan",{"queries":["EXMP earnings"]},1)])
    result=ModelBrain(model,structured_retries=1).structured(SearchPlan,"Plan",{},RunContext("retry",time.monotonic()+30),"news.plan")
    assert result.queries==["EXMP earnings"] and model._cursor==2


def test_failed_final_review_keeps_initial_reviewed_facts():
    from langchain_core.messages import AIMessage
    review={"index":0,"supported":True,"relevant":True,"event_date_supported":False,"role":"news","origin":"","reason":"Supported"}
    fact={"source":"s0","blocks":[0],"summary":"EXMP announced a product."}
    model=ScriptedModel(responses=[call("SearchPlan",{"queries":["discover"]},1),call("Extraction",{"facts":[fact]},2),call("ReviewReport",{"facts":[review]},3),call("SearchPlan",{"queries":["supplement"]},4),call("Extraction",{"facts":[fact]},5),AIMessage(content="invalid"),AIMessage(content="invalid")])
    def search(query,**kwargs):
        return [{"url":"https://example.org/"+query.split()[0],"published_date":"2026-09-11","raw_content":"EXMP announced a product."}]
    result=investigated_news(model,search,"EXMP news","EXMP",WINDOW,RunContext("retained",time.monotonic()+90))
    assert result.metrics["accepted_fact_count"]==1
    assert any("initial reviewed facts retained" in limit for limit in result.limitations)


@pytest.mark.parametrize("approved",[True,False])
def test_research_expands_to_second_source_and_cross_checks(approved):
    calls=[]
    body="On September 10, 2026, EXMP announced a new product."
    def search(query,**kwargs):
        calls.append(query)
        domain="issuer.example" if query.startswith("discovery") else "reporter.example"
        return [{"url":f"https://{domain}/news","title":"Product announcement","published_date":"2026-09-11","raw_content":body}]
    fact=lambda source:{"source":"wrong-model-source-id","blocks":[0],"summary":"EXMP announced a new product.","event_date":"2026-09-10","date_explanation":"Explicit in body"}
    review=lambda i:{"index":i,"supported":True,"relevant":True,"event_date_supported":True,"role":"news","origin":"","reason":"Supported"}
    model=ScriptedModel(responses=[call("SearchPlan",{"queries":["discovery"]},1),call("Extraction",{"facts":[fact("s0")],"followup_queries":["corroborate"]},2),call("ReviewReport",{"facts":[review(0)]},3),call("SearchPlan",{"queries":["corroborate"]},4),call("Extraction",{"facts":[fact("s1")]},5),call("ReviewReport",{"facts":[review(0),review(1)],"comparisons":[{"left":0,"right":1,"relation":"consistent","common_claim":"EXMP announced a new product."}]},6)])
    model.responses.append(call("ComparisonAudit",{"approved_indices":[0] if approved else []},7))
    result=investigated_news(model,search,"EXMP news","EXMP",WINDOW,RunContext("test",time.monotonic()+90))
    assert len(calls)==2 and calls[0].startswith("discovery") and calls[1].startswith("corroborate")
    assert result.metrics["accepted_fact_count"]==2 and result.metrics["accepted_source_count"]==2
    assert (result.metadata["cross_checks"][0]["relation"]=="consistent") if approved else not result.metadata["cross_checks"]
    assert result.evidence[0].metadata["quote"]==body


@pytest.mark.parametrize("origin,expected",[("Reuters","same_origin"),("","consistent")])
def test_syndicated_sources_do_not_count_as_independent(origin,expected):
    facts=[{"url":f"https://{domain}/news","domain":domain,"summary":"Event","quote":"Original event report","event_date":"2026-09-10","published":"2026-09-11","title":"News"} for domain in ("a.example","b.example")]
    checks=[{"index":i,"supported":True,"relevant":True,"event_date_supported":True,"role":"context","origin":origin,"reason":"fixture"} for i in (0,1)]
    state={"facts":facts,"review":{"facts":checks,"comparisons":[{"left":0,"right":1,"relation":"consistent","common_claim":"Event"}]}}
    result=evidence_result(state,"EXMP",WINDOW,False)
    assert result.metadata["cross_checks"][0]["relation"]==expected


def test_publication_date_cannot_establish_driver_event_date():
    state={"facts":[{"url":"https://a.example/news","domain":"a.example","summary":"A reported driver","quote":"Source text","event_date":"2026-09-10","published":"2026-09-11","title":"News"}],"review":{"facts":[{"index":0,"supported":True,"relevant":True,"event_date_supported":False,"role":"reported_driver","origin":"","reason":"No event date"}],"comparisons":[]}}
    result=evidence_result(state,"EXMP",WINDOW,True)
    assert result.metrics["reported_driver_count"]==0 and result.metrics["confirmed_driver_count"]==0
    assert result.evidence[0].metadata["claim_role"]=="context"
    assert result.evidence[0].metadata["event_date"]==""
    assert result.evidence[0].metadata["date_basis"]=="publication"


def test_search_snippet_without_body_is_not_evidence():
    assert source_rows([{"url":"https://a.example","content":"snippet"}])==[]


def test_markdown_transport_does_not_consume_article_read_budget():
    raw="![photo](https://example.org/"+"a"*30000+")\n\nOn September 10, 2026, [EXMP](https://example.org) reported revenue of $120 million."
    body=readable_body(raw)
    assert "$120 million" in body and "September 10, 2026" in body and len(body)<150
    assert source_rows([{"url":"https://example.org/article","raw_content":raw}])[0]["blocks"]==[body]


def test_old_event_cannot_be_relocated_to_publication_month():
    assert resolve_event_date("June 10, 2026","explicit","On June 10, 2026 earnings were released.","2026-09-10")=="2026-06-10"
    assert resolve_event_date("September 10, 2026","explicit","On June 10, 2026 earnings were released.","2026-09-10")==""
    assert resolve_event_date("Tuesday","weekday","Shares rose Tuesday.","2026-09-09")=="2026-09-08"


def test_long_pages_include_later_body_not_only_navigation():
    rows=source_rows([{"url":"https://a.example","raw_content":"navigation "*3000+"ARTICLE_BODY "*2000}])
    assert any("ARTICLE_BODY" in block for block in rows[0]["blocks"])
    assert rows[0]["body_truncated"]


def test_issuer_subdomains_are_not_independent_publishers():
    assert publisher_family("ir.amd.com")==publisher_family("www.amd.com")


def test_date_anchor_whitespace_weekday_and_timezone():
    assert resolve_event_date("September 1, 2026","explicit","September 1,\n2026","2026-09-02")=="2026-09-01"
    assert resolve_event_date("周四(10日)","weekday","周四(10日)","2026-09-11")=="2026-09-10"
    assert resolve_event_date("香港時間9月10日凌晨1時","explicit","香港時間9月10日凌晨1時","2026-09-10")=="2026-09-09"


def test_conflicts_and_inference_keep_explicit_evidence_basis():
    facts=[{"url":f"https://{domain}/news","domain":domain,"summary":"The company increased investment.","quote":"The company increased investment.","event_date":"2026-09-10","published":"2026-09-11","title":"News"} for domain in ("a.example","b.example")]
    checks=[{"index":i,"supported":True,"relevant":True,"event_date_supported":True,"role":"context","origin":"","reason":"fixture"} for i in (0,1)]
    state={"facts":facts,"review":{"facts":checks,"comparisons":[{"left":0,"right":1,"relation":"conflicting","common_claim":"The reports disagree about the investment."}],"mechanisms":[{"basis":[0],"explanation":"Higher investment could pressure cash flow.","direction":"negative"},{"basis":[9],"explanation":"Unsupported basis","direction":"positive"}]}}
    result=evidence_result(state,"EXMP",WINDOW,True)
    assert result.metadata["cross_checks"][0]["relation"]=="conflicting"
    inferred=[e for e in result.evidence if e.metadata.get("claim_role")=="inference"]
    assert len(inferred)==1 and inferred[0].metadata["from"]
    assert result.metrics["confirmed_driver_count"]==0
