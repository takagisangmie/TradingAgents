"""Research Manager: turns evidence and symmetric debate into an investment plan."""

from __future__ import annotations

from tradingagents.agents.philosophy import format_philosophy_reviews
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")
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

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, independently verify the debate against the source reports and deliver a clear investment plan for risk review. Persuasive wording is not evidence.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.
Do not invent holdings or position sizes. When portfolio context is missing, make actions explicitly conditional.

**Portfolio Context:**
{portfolio_context}

**Information Audit:**
{audit_report}

**Source Reports:**
{source_reports}

**Independent Investment-Methodology Reviews:**
{philosophy_reviews}

---

**Debate History:**
{history}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
