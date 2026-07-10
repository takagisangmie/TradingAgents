# Architecture

## Boundaries

The project separates four responsibilities:

1. `tradingagents/dataflows`: vendor adapters, symbol normalization, point-in-time filtering, and source-access errors.
2. `tradingagents/security`: deterministic review and redaction of untrusted external text.
3. `tradingagents/agents`: report-producing and decision-making graph nodes.
4. `tradingagents/graph`: workflow wiring, checkpointing, propagation, and persistence.

CLI rendering belongs in `cli`; it should not implement market-data or agent
logic.

## External Content Contract

Every successful vendor result is wrapped in `UNTRUSTED_EXTERNAL_DATA` markers
before an LLM receives it. Suspicious instruction-like lines are replaced with
`REDACTED_BY_INFORMATION_AUDITOR` and an `INFORMATION_SECURITY_WARNING`.

The Information Auditor graph node runs after selected analysts and before the
six parallel investment-methodology reviewers and bull/bear debate. It
summarizes prompt-injection findings, empty reports,
missing data, rate limits, permission failures, and community-site risk
controls. Each methodology reviewer reads the same audited reports and writes
only to the append-reduced `philosophy_reviews` state, so parallel branches
cannot overwrite or anchor one another. LangGraph joins all six branches before
the bull/bear research stage. Downstream researchers must lower confidence for
flagged sources.

## Adding a Data Vendor

1. Add one focused module under `tradingagents/dataflows`.
2. Raise `NoMarketDataError`, `VendorRateLimitError`, or
   `VendorNotConfiguredError` according to behavior.
3. Register functions in `dataflows/interface.py` without hidden fallback.
4. Add mocked unit tests for response parsing, point-in-time filtering,
   permission failures, and rate limits.
5. Keep live tests opt-in and keyed by environment variables.

## Iteration Workflow

Development happens on feature branches based on `origin/main`. Before commit:

```powershell
.\.venv\Scripts\python -m ruff check tradingagents cli tests
.\.venv\Scripts\python -m pytest
git diff --check
```

Credentials and generated reports stay outside Git through `.gitignore`.
