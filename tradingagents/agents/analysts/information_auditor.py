"""Information Auditor: deterministic review of collected analyst information."""

from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.dataflows.config import get_config
from tradingagents.security import build_information_audit_report


def create_information_auditor():
    """Create a graph node that summarizes security and source-quality warnings."""

    def information_auditor_node(state) -> dict:
        # Keep the same localization contract as report-producing LLM agents.
        get_language_instruction()
        reports = {
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
        }
        report = build_information_audit_report(
            reports,
            output_language=get_config().get("output_language", "English"),
        )
        return {"information_audit_report": report}

    return information_auditor_node
