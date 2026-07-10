"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
two decision-making agents (Research Manager and Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A traceable claim used in an investment decision."""

    claim: str = Field(description="A material factual or inferential claim.")
    evidence: str = Field(description="The concrete observation supporting the claim.")
    source: str = Field(description="The report or debate turn containing the evidence.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence after accounting for audit warnings and missing data."
    )
    counterevidence: str | None = Field(
        default=None,
        description="Material evidence against the claim, or null when none was found.",
    )


class PhilosophyReview(BaseModel):
    """Common contract for independent investment-methodology reviewers."""

    circle_of_competence: Literal["inside", "outside", "uncertain"] = Field(
        description="Whether the available evidence supports understanding the business."
    )
    business_quality: Literal["strong", "acceptable", "weak", "uncertain"] = Field(
        description="Business quality under the assigned methodology."
    )
    management_quality: Literal[
        "strong", "acceptable", "concerning", "unknown"
    ] = Field(description="Management and capital-allocation assessment.")
    valuation_view: Literal[
        "attractive", "fair", "expensive", "insufficient_data"
    ] = Field(description="Valuation assessment without fabricated precision.")
    recommendation: Literal["pass", "watch", "reject", "abstain"] = Field(
        description="Methodology review outcome; this is not a portfolio action."
    )
    thesis: str = Field(description="Concise methodology-specific investment thesis.")
    evidence_ledger: list[EvidenceItem] = Field(
        default_factory=list,
        description="Traceable claims supporting or challenging the thesis.",
    )
    disconfirming_evidence: list[str] = Field(
        default_factory=list,
        description="Strongest evidence against the thesis.",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Information required before confidence can increase.",
    )
    invalidation_conditions: list[str] = Field(
        default_factory=list,
        description="Observable conditions that would invalidate the thesis.",
    )


def render_philosophy_review(
    review: PhilosophyReview,
    reviewer_name: str,
    framework_name: str,
) -> str:
    """Render a methodology review with deterministic identity metadata."""
    parts = [
        f"**Reviewer**: {reviewer_name}",
        f"**Framework**: {framework_name}",
        f"**Methodology Outcome**: {review.recommendation.capitalize()}",
        f"**Circle of Competence**: {review.circle_of_competence.capitalize()}",
        f"**Business Quality**: {review.business_quality.capitalize()}",
        f"**Management Quality**: {review.management_quality.capitalize()}",
        f"**Valuation View**: {review.valuation_view.replace('_', ' ').title()}",
        "",
        f"**Thesis**: {review.thesis}",
    ]
    if review.evidence_ledger:
        parts.extend(["", "**Evidence Ledger**:"])
        for item in review.evidence_ledger:
            counter = item.counterevidence or "None identified"
            parts.append(
                f"- {item.claim} | Evidence: {item.evidence} | Source: {item.source} "
                f"| Confidence: {item.confidence} | Counterevidence: {counter}"
            )
    for heading, values in (
        ("Disconfirming Evidence", review.disconfirming_evidence),
        ("Missing Information", review.missing_information),
        ("Invalidation Conditions", review.invalidation_conditions),
    ):
        if values:
            parts.extend(["", f"**{heading}**:"])
            parts.extend(f"- {value}" for value in values)
    return "\n".join(parts)


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    The recommendation pins the directional view, the rationale captures
    which evidence carried the debate, and strategic actions remain
    conditional on the supplied portfolio context.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete, constraint-aware next steps. If portfolio context is "
            "missing, state which inputs are required instead of inventing sizing."
        ),
    )
    evidence_ledger: list[EvidenceItem] = Field(
        default_factory=list,
        description="Key claims with evidence, source, confidence, and counterevidence.",
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and downstream review."""
    parts = [
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ]
    if plan.evidence_ledger:
        parts.extend(["", "**Evidence Ledger**:"])
        for item in plan.evidence_ledger:
            counter = item.counterevidence or "None identified"
            parts.append(
                f"- {item.claim} | Evidence: {item.evidence} | Source: {item.source} "
                f"| Confidence: {item.confidence} | Counterevidence: {counter}"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new decision signal."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
