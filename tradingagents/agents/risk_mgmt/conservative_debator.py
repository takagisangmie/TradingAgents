from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_market_profile_instruction,
)


def create_fundamental_event_risk_analyst(llm):
    def fundamental_event_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        role_history = risk_debate_state.get("fundamental_event_history", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        research_plan = state["investment_plan"]
        portfolio_context = state.get("portfolio_context", "")
        audit_report = state.get("information_audit_report", "")

        prompt = f"""You are the Fundamental & Event Risk Analyst. Independently evaluate the proposed research plan; do not default to pessimism. Cover balance-sheet and cash-flow fragility, valuation sensitivity, earnings and disclosure risk, regulation, governance, macro/sector events, and thesis-invalidating catalysts.

Research plan: {research_plan}
Portfolio context and constraints: {portfolio_context}

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Information Audit: {audit_report}
Other completed risk reviews: {history}

Return: identified risks, supporting evidence and source, confidence, concrete thesis-invalidation triggers or mitigations, and missing data. Separate observed facts from inference.""" + get_market_profile_instruction() + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Fundamental & Event Risk Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "market_liquidity_history": risk_debate_state.get("market_liquidity_history", ""),
            "fundamental_event_history": role_history + "\n" + argument,
            "portfolio_exposure_history": risk_debate_state.get("portfolio_exposure_history", ""),
            "latest_speaker": "Fundamental & Event",
            "current_market_liquidity_response": risk_debate_state.get(
                "current_market_liquidity_response", ""
            ),
            "current_fundamental_event_response": argument,
            "current_portfolio_exposure_response": risk_debate_state.get(
                "current_portfolio_exposure_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return fundamental_event_node


def create_conservative_debator(llm):
    """Deprecated compatibility alias."""
    return create_fundamental_event_risk_analyst(llm)
