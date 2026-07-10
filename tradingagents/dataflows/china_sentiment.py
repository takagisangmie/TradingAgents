"""Low-frequency public-page adapters for A-share community sentiment.

Xueqiu and TaoGuBa do not expose a stable public community API suitable for
this project. These adapters request public stock pages once, never bypass a
challenge, and return an explicit access warning on login walls, CAPTCHA, 403,
418, or 429 responses. Optional user-provided cookies are read from environment
variables but are never logged or persisted.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from functools import lru_cache

import requests
from parsel import Selector

from tradingagents.security import audit_external_content

from .tushare import normalize_ts_code, to_xueqiu_symbol

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "TradingAgents/0.3.1"
)
_ACCESS_MARKERS = (
    "访问频繁",
    "安全验证",
    "请输入验证码",
    "滑动验证",
    "登录后查看",
    "captcha",
    "verify you are human",
    "_waf_",
)


class SourceAccessWarning(RuntimeError):
    """The source refused automated public-page access."""


def _request_page(url: str, *, cookie: str = "", timeout: float = 8.0) -> str:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    if cookie:
        headers["Cookie"] = cookie
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    if response.status_code in {401, 403, 418, 429}:
        raise SourceAccessWarning(f"HTTP {response.status_code}")
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    lowered = response.text.lower()
    if any(marker.lower() in lowered for marker in _ACCESS_MARKERS):
        raise SourceAccessWarning("login/CAPTCHA/risk-control page detected")
    return response.text


def _clean_texts(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", value).strip()
        if len(text) < 12 or text in seen:
            continue
        if len(text) > 500:
            text = text[:500] + "..."
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _selector_candidates(selector: Selector, *, limit: int) -> list[str]:
    candidates = selector.xpath(
        "//*[self::article or contains(@class,'timeline') or contains(@class,'status') "
        "or contains(@class,'topic') or contains(@class,'feed') or contains(@class,'list')]"
        "//text()[normalize-space()]"
    ).getall()
    return _clean_texts(candidates, limit=limit)


def _fallback_body_candidates(selector: Selector, ticker_code: str, *, limit: int) -> list[str]:
    values = selector.xpath("//body//*[not(self::script) and not(self::style)]/text()[normalize-space()]").getall()
    cleaned = _clean_texts(values, limit=limit * 8)
    ticker_hits = [text for text in cleaned if ticker_code in text]
    return (ticker_hits or cleaned)[:limit]


def _render_source(source: str, url: str, posts: list[str]) -> str:
    if not posts:
        return f"[SOURCE_ACCESS_WARNING] {source}: public page returned no parseable discussion posts ({url})."
    raw = "\n".join(f"- {post}" for post in posts)
    audited = audit_external_content(source, raw, max_chars=20_000)
    return f"## {source} public discussion snapshot\nURL: {url}\n{audited.render_for_model()}"


@lru_cache(maxsize=128)
def _fetch_xueqiu_cached(ticker: str, limit: int, cookie: str) -> str:
    symbol = to_xueqiu_symbol(ticker)
    url = f"https://xueqiu.com/S/{symbol}"
    try:
        html = _request_page(url, cookie=cookie)
    except (SourceAccessWarning, requests.RequestException) as exc:
        return f"[SOURCE_ACCESS_WARNING] Xueqiu blocked or failed public access: {type(exc).__name__}: {exc}"
    selector = Selector(text=html)
    posts = _selector_candidates(selector, limit=limit)
    if not posts:
        posts = _fallback_body_candidates(selector, symbol[-6:], limit=limit)
    return _render_source("Xueqiu", url, posts)


def fetch_xueqiu_discussions(ticker: str, limit: int = 20) -> str:
    """Fetch a single public Xueqiu stock page without bypassing risk controls."""
    return _fetch_xueqiu_cached(ticker, limit, os.getenv("XUEQIU_COOKIE", "").strip())


@lru_cache(maxsize=128)
def _fetch_taoguba_cached(ticker: str, limit: int, cookie: str) -> str:
    code = normalize_ts_code(ticker).split(".")[0]
    url = f"https://www.tgb.cn/quotes/{code}"
    try:
        html = _request_page(url, cookie=cookie)
    except (SourceAccessWarning, requests.RequestException) as exc:
        return f"[SOURCE_ACCESS_WARNING] TaoGuBa blocked or failed public access: {type(exc).__name__}: {exc}"
    selector = Selector(text=html)
    posts = _selector_candidates(selector, limit=limit)
    if not posts:
        posts = _fallback_body_candidates(selector, code, limit=limit)
    return _render_source("TaoGuBa", url, posts)


def fetch_taoguba_discussions(ticker: str, limit: int = 20) -> str:
    """Fetch a single public TaoGuBa stock page without bypassing risk controls."""
    return _fetch_taoguba_cached(ticker, limit, os.getenv("TAOGUBA_COOKIE", "").strip())


def _historical_social_warning(trade_date: str | None) -> str | None:
    if not trade_date:
        return None
    try:
        requested = date.fromisoformat(str(trade_date))
    except ValueError:
        return f"[SOURCE_ACCESS_WARNING] Invalid trade date for community sentiment: {trade_date!r}."
    if requested < date.today() - timedelta(days=1):
        return (
            "[SOURCE_ACCESS_WARNING] Historical Xueqiu/TaoGuBa snapshots are not available "
            f"through the public pages for {trade_date}. Current posts were deliberately not "
            "used, preventing look-ahead bias."
        )
    return None


def fetch_china_community_sentiment(
    ticker: str,
    *,
    trade_date: str | None = None,
    limit_per_source: int = 20,
) -> str:
    """Combine Xueqiu and TaoGuBa public discussions for a current A-share run."""
    historical_warning = _historical_social_warning(trade_date)
    if historical_warning:
        return historical_warning
    return "\n\n".join(
        (
            fetch_xueqiu_discussions(ticker, limit_per_source),
            fetch_taoguba_discussions(ticker, limit_per_source),
        )
    )


def clear_sentiment_cache() -> None:
    _fetch_xueqiu_cached.cache_clear()
    _fetch_taoguba_cached.cache_clear()
