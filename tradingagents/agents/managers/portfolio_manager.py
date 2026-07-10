"""Portfolio Manager: synthesises independent risk reviews into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.philosophy import format_philosophy_reviews
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_market_profile_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        portfolio_context = state.get("portfolio_context", "")
        audit_report = state.get("information_audit_report", "")
        source_reports = "\n\n".join(
            [
                f"Market report:\n{state.get('market_report', '')}",
                f"Sentiment report:\n{state.get('sentiment_report', '')}",
                f"News report:\n{state.get('news_report', '')}",
                f"Fundamentals report:\n{state.get('fundamentals_report', '')}",
            ]
        )
        philosophy_reviews = format_philosophy_reviews(
            state.get("philosophy_reviews", [])
        )

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, act as the final investment-committee decision maker. Verify every material conclusion against the source reports, audit findings, research plan, portfolio constraints, and domain-specific risk reviews.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Portfolio context and constraints: **{portfolio_context}**
{lessons_line}
**Information Audit:**
{audit_report}

**Source Reports:**
{source_reports}

**Independent Investment-Methodology Reviews:**
{philosophy_reviews}

**Independent Risk Reviews:**
{history}

---

Be decisive but do not manufacture precision. If holdings, cash, risk budget, horizon, costs, or a reliable price anchor are absent, state that limitation and keep sizing/price levels conditional. Ground conclusions in traceable evidence and name the strongest counterevidence.{get_market_profile_instruction()}{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "market_liquidity_history": risk_debate_state["market_liquidity_history"],
            "fundamental_event_history": risk_debate_state["fundamental_event_history"],
            "portfolio_exposure_history": risk_debate_state["portfolio_exposure_history"],
            "latest_speaker": "Judge",
            "current_market_liquidity_response": risk_debate_state["current_market_liquidity_response"],
            "current_fundamental_event_response": risk_debate_state["current_fundamental_event_response"],
            "current_portfolio_exposure_response": risk_debate_state["current_portfolio_exposure_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
