from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_market_profile_instruction,
)


def create_market_liquidity_risk_analyst(llm):
    def market_liquidity_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        role_history = risk_debate_state.get("market_liquidity_history", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        research_plan = state["investment_plan"]
        portfolio_context = state.get("portfolio_context", "")
        audit_report = state.get("information_audit_report", "")

        prompt = f"""You are the Market & Liquidity Risk Analyst. Independently evaluate the proposed research plan; do not advocate for or against it as a matter of personality. Cover volatility regime, drawdown and gap risk, trading liquidity, volume/price-impact concerns, stop feasibility, and A-share execution constraints where applicable.

Research plan: {research_plan}
Portfolio context and constraints: {portfolio_context}

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Information Audit: {audit_report}
Other completed risk reviews: {history}

Return: identified risks, supporting evidence and source, confidence, measurable limits or mitigations, and missing data. Never invent portfolio facts or precise limits when the portfolio context is absent.""" + get_market_profile_instruction() + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Market & Liquidity Risk Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "market_liquidity_history": role_history + "\n" + argument,
            "fundamental_event_history": risk_debate_state.get("fundamental_event_history", ""),
            "portfolio_exposure_history": risk_debate_state.get("portfolio_exposure_history", ""),
            "latest_speaker": "Market & Liquidity",
            "current_market_liquidity_response": argument,
            "current_fundamental_event_response": risk_debate_state.get("current_fundamental_event_response", ""),
            "current_portfolio_exposure_response": risk_debate_state.get(
                "current_portfolio_exposure_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return market_liquidity_node


def create_aggressive_debator(llm):
    """Deprecated compatibility alias."""
    return create_market_liquidity_risk_analyst(llm)
