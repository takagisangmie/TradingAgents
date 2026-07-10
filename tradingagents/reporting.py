"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, methodology reviews, research, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.
"""

from datetime import datetime
from pathlib import Path


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 0. Information security and source quality
    if final_state.get("information_audit_report"):
        audit_dir = save_path / "0_information_audit"
        audit_dir.mkdir(exist_ok=True)
        audit_text = final_state["information_audit_report"]
        (audit_dir / "audit.md").write_text(audit_text, encoding="utf-8")
        sections.append(f"## 0. Information Security Audit\n\n{audit_text}")

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Investment-methodology review
    philosophy_reviews = final_state.get("philosophy_reviews", [])
    if philosophy_reviews:
        philosophy_dir = save_path / "2_philosophy"
        philosophy_dir.mkdir(exist_ok=True)
        philosophy_parts = []
        for review in philosophy_reviews:
            key = review.get("key", "reviewer")
            reviewer = review.get("reviewer", "Methodology Reviewer")
            content = review.get("content", "")
            (philosophy_dir / f"{key}.md").write_text(content, encoding="utf-8")
            philosophy_parts.append((reviewer, content))
        content = "\n\n".join(
            f"### {name}\n{text}" for name, text in philosophy_parts
        )
        sections.append(f"## II. Investment Methodology Reviews\n\n{content}")

    # 3. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "3_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## III. Research Team Decision\n\n{content}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("market_liquidity_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "market_liquidity.md").write_text(risk["market_liquidity_history"], encoding="utf-8")
            risk_parts.append(("Market & Liquidity Risk Analyst", risk["market_liquidity_history"]))
        if risk.get("fundamental_event_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "fundamental_event.md").write_text(risk["fundamental_event_history"], encoding="utf-8")
            risk_parts.append(("Fundamental & Event Risk Analyst", risk["fundamental_event_history"]))
        if risk.get("portfolio_exposure_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "portfolio_exposure.md").write_text(risk["portfolio_exposure_history"], encoding="utf-8")
            risk_parts.append(("Portfolio Exposure Risk Analyst", risk["portfolio_exposure_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Review\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
