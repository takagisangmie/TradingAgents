"""Regression coverage for the evidence-first role redesign."""

from unittest.mock import MagicMock

import pytest
from langgraph.prebuilt import ToolNode

from tradingagents.agents.philosophy import (
    PHILOSOPHY_REVIEWER_NAMES,
    PHILOSOPHY_SPECS,
    create_philosophy_reviewer,
)
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import (
    create_market_liquidity_risk_analyst,
)
from tradingagents.agents.risk_mgmt.conservative_debator import (
    create_fundamental_event_risk_analyst,
)
from tradingagents.agents.risk_mgmt.neutral_debator import (
    create_portfolio_exposure_risk_analyst,
)
from tradingagents.agents.schemas import (
    EvidenceItem,
    PhilosophyReview,
    render_philosophy_review,
)
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup


def _research_state(*, count: int, current_response: str, bear_history: str = ""):
    return {
        "asset_type": "stock",
        "instrument_context": "Ticker: NVDA",
        "market_report": "market evidence",
        "sentiment_report": "sentiment evidence",
        "news_report": "news evidence",
        "fundamentals_report": "fundamental evidence",
        "information_audit_report": "audit evidence",
        "investment_debate_state": {
            "history": "prior debate",
            "bull_history": "",
            "bear_history": bear_history,
            "current_response": current_response,
            "judge_decision": "",
            "count": count,
        },
    }


@pytest.mark.unit
def test_opening_theses_do_not_anchor_on_the_other_side():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="thesis")

    create_bull_researcher(llm)(
        _research_state(count=0, current_response="BEAR_ANCHOR")
    )
    bull_prompt = llm.invoke.call_args.args[0]
    assert "BEAR_ANCHOR" not in bull_prompt

    create_bear_researcher(llm)(
        _research_state(count=1, current_response="BULL_ANCHOR")
    )
    bear_prompt = llm.invoke.call_args.args[0]
    assert "BULL_ANCHOR" not in bear_prompt


def _risk_state():
    return {
        "asset_type": "stock",
        "instrument_context": "Ticker: NVDA",
        "portfolio_context": "Current weight 2%; maximum weight 5%.",
        "investment_plan": "Overweight conditionally.",
        "market_report": "market evidence",
        "sentiment_report": "sentiment evidence",
        "news_report": "news evidence",
        "fundamentals_report": "fundamental evidence",
        "information_audit_report": "one source is low confidence",
        "risk_debate_state": {
            "market_liquidity_history": "",
            "fundamental_event_history": "",
            "portfolio_exposure_history": "",
            "history": "",
            "latest_speaker": "",
            "current_market_liquidity_response": "",
            "current_fundamental_event_response": "",
            "current_portfolio_exposure_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "history_key", "label"),
    [
        (
            create_market_liquidity_risk_analyst,
            "market_liquidity_history",
            "Market & Liquidity Risk Analyst",
        ),
        (
            create_fundamental_event_risk_analyst,
            "fundamental_event_history",
            "Fundamental & Event Risk Analyst",
        ),
        (
            create_portfolio_exposure_risk_analyst,
            "portfolio_exposure_history",
            "Portfolio Exposure Risk Analyst",
        ),
    ],
)
def test_domain_risk_roles_consume_plan_audit_and_portfolio_context(
    factory, history_key, label
):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="domain review")
    result = factory(llm)(_risk_state())
    prompt = llm.invoke.call_args.args[0]
    assert "Overweight conditionally" in prompt
    assert "Current weight 2%" in prompt
    assert "low confidence" in prompt
    assert label in result["risk_debate_state"][history_key]


@pytest.mark.unit
def test_initial_state_has_portfolio_context_and_no_trader_layer():
    state = Propagator().create_initial_state(
        "NVDA", "2026-01-10", portfolio_context="Current weight 2%."
    )
    assert state["portfolio_context"] == "Current weight 2%."
    assert "trader_investment_plan" not in state


def _philosophy_state():
    state = _risk_state()
    state["philosophy_reviews"] = [
        {"reviewer": "Other Reviewer", "content": "OTHER_REVIEW_ANCHOR"}
    ]
    return state


@pytest.mark.unit
@pytest.mark.parametrize("spec", PHILOSOPHY_SPECS, ids=lambda spec: spec.key)
def test_philosophy_reviewers_are_independent_and_bounded(spec):
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = MagicMock(content="independent review")
    result = create_philosophy_reviewer(llm, spec)(_philosophy_state())
    prompt = llm.invoke.call_args.args[0]

    assert spec.framework_name in prompt
    assert "employed by the CIO" in prompt
    assert "not a Buy/Sell instruction" in prompt
    assert "OTHER_REVIEW_ANCHOR" not in prompt
    assert result["philosophy_reviews"] == [
        {
            "key": spec.key,
            "reviewer": spec.node_name,
            "content": "independent review",
        }
    ]


@pytest.mark.unit
def test_six_reviewer_roster_is_complete_and_unique():
    assert len(PHILOSOPHY_SPECS) == 6
    assert len(set(PHILOSOPHY_REVIEWER_NAMES)) == 6
    assert {spec.key for spec in PHILOSOPHY_SPECS} == {
        "buffett",
        "duan_yongping",
        "graham",
        "fisher",
        "lynch",
        "howard_marks",
    }


@pytest.mark.unit
def test_philosophy_review_render_has_traceable_evidence():
    review = PhilosophyReview(
        circle_of_competence="inside",
        business_quality="strong",
        management_quality="acceptable",
        valuation_view="fair",
        recommendation="watch",
        thesis="Good business, but the margin of safety is limited.",
        evidence_ledger=[
            EvidenceItem(
                claim="Cash conversion is resilient.",
                evidence="Five-year operating cash flow stayed positive.",
                source="Fundamentals report",
                confidence="medium",
                counterevidence="Working capital rose in the latest period.",
            )
        ],
        disconfirming_evidence=["Recent margin compression"],
        missing_information=["Segment ROIC"],
        invalidation_conditions=["Sustained negative free cash flow"],
    )
    rendered = render_philosophy_review(
        review,
        "Buffett Methodology Reviewer",
        "Buffett-inspired quality and value",
    )
    assert "**Methodology Outcome**: Watch" in rendered
    assert "Fundamentals report" in rendered
    assert "Working capital rose" in rendered


@pytest.mark.unit
def test_graph_runs_six_reviewers_behind_a_parallel_barrier():
    quick_llm = MagicMock()
    deep_llm = MagicMock()
    tool_nodes = {
        key: ToolNode([])
        for key in ("market", "social", "news", "fundamentals")
    }
    workflow = GraphSetup(
        quick_llm,
        deep_llm,
        tool_nodes,
        ConditionalLogic(),
    ).setup_graph()
    workflow.compile()

    assert set(PHILOSOPHY_REVIEWER_NAMES) <= set(workflow.nodes)
    assert "Trader" not in workflow.nodes
    assert (
        (tuple(PHILOSOPHY_REVIEWER_NAMES), "Bull Researcher")
        in workflow.waiting_edges
    )
