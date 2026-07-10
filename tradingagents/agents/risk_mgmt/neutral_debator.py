from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_market_profile_instruction,
)


def create_portfolio_exposure_risk_analyst(llm):
    def portfolio_exposure_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        role_history = risk_debate_state.get("portfolio_exposure_history", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)

        research_plan = state["investment_plan"]
        portfolio_context = state.get("portfolio_context", "")
        audit_report = state.get("information_audit_report", "")

        prompt = f"""You are the Portfolio Exposure Risk Analyst. Independently assess how the research plan would affect an actual portfolio. Cover current-position dependence, concentration, sector/factor/currency exposure, correlation, diversification, risk budget, horizon mismatch, turnover, and transaction costs. Do not invent portfolio facts when context is absent.

Research plan: {research_plan}
Portfolio context and constraints: {portfolio_context}

{instrument_context}
Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Information Audit: {audit_report}
Other completed risk reviews: {history}

Return: identified exposures, supporting evidence and source, confidence, conditional sizing or mitigation rules, and the portfolio fields still required for a firm decision.""" + get_market_profile_instruction() + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Portfolio Exposure Risk Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "market_liquidity_history": risk_debate_state.get("market_liquidity_history", ""),
            "fundamental_event_history": risk_debate_state.get("fundamental_event_history", ""),
            "portfolio_exposure_history": role_history + "\n" + argument,
            "latest_speaker": "Portfolio Exposure",
            "current_market_liquidity_response": risk_debate_state.get(
                "current_market_liquidity_response", ""
            ),
            "current_fundamental_event_response": risk_debate_state.get("current_fundamental_event_response", ""),
            "current_portfolio_exposure_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return portfolio_exposure_node


def create_neutral_debator(llm):
    """Deprecated compatibility alias."""
    return create_portfolio_exposure_risk_analyst(llm)
