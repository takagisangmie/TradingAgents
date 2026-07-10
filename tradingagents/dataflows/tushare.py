"""Tushare Pro vendor for the A-share market profile.

The adapter keeps Tushare-specific symbols, permissions, and response shapes in
one module. Public tool functions return the same text interface used by the
existing agents, while helpers expose typed frames for deterministic outcome
evaluation and technical-indicator calculation.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
import requests
from stockstats import wrap

from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError

_TS_CODE = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ|BJ)$", re.IGNORECASE)
_PREFIXED_CODE = re.compile(r"^(?P<exchange>SH|SZ|BJ)(?P<code>\d{6})$", re.IGNORECASE)

SUPPORTED_INDICATORS = {
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "macd",
    "macds",
    "macdh",
    "rsi",
    "rsi_14",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr",
    "vwma",
    "mfi",
}


def normalize_ts_code(raw: str) -> str:
    """Normalize common A-share forms to Tushare's ``000001.SZ`` format."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("A-share ticker must be a non-empty string")
    symbol = raw.strip().upper()
    match = _TS_CODE.fullmatch(symbol)
    if match:
        return f"{match.group('code')}.{match.group('exchange')}"
    match = _PREFIXED_CODE.fullmatch(symbol)
    if match:
        return f"{match.group('code')}.{match.group('exchange')}"
    if not symbol.isdigit() or len(symbol) != 6:
        raise ValueError(
            f"Unsupported A-share ticker {raw!r}; use a six-digit code or a .SH/.SZ/.BJ suffix"
        )
    if symbol.startswith(("4", "8", "9")):
        exchange = "BJ"
    elif symbol.startswith(("5", "6", "7")):
        exchange = "SH"
    else:
        exchange = "SZ"
    return f"{symbol}.{exchange}"


def to_xueqiu_symbol(raw: str) -> str:
    code, exchange = normalize_ts_code(raw).split(".")
    return f"{exchange}{code}"


def _compact_date(value: str) -> str:
    return datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y%m%d")


def _display_date(value: object) -> str:
    text = str(value or "")
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _permission_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "权限",
            "积分",
            "token",
            "抱歉，您没有访问该接口的权限",
            "permission",
            "2002",
            "401",
            "403",
        )
    )


def _rate_limit_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in ("每分钟最多访问", "频率", "rate limit", "too many", "429")
    )


@lru_cache(maxsize=1)
def _pro_client():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise VendorNotConfiguredError(
            "Tushare is selected but TUSHARE_TOKEN is not set. Add it to .env before a live run."
        )
    return _TushareHttpClient(
        token=token,
        base_url=os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip(),
    )


class _TushareHttpClient:
    """Minimal client for Tushare's documented POST API."""

    def __init__(self, *, token: str, base_url: str, timeout: float = 30.0):
        self._token = token
        self._base_url = base_url
        self._timeout = timeout

    def query(self, api_name: str, *, fields: str = "", **params) -> pd.DataFrame:
        response = requests.post(
            self._base_url,
            json={
                "api_name": api_name,
                "token": self._token,
                "params": params,
                "fields": fields,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in (0, None):
            raise RuntimeError(
                f"Tushare API error {payload['code']}: "
                f"{payload.get('msg') or 'unknown vendor error'}"
            )
        data = payload.get("data") or {}
        columns = data.get("fields") or []
        rows = data.get("items") or []
        return pd.DataFrame(rows, columns=columns)

    def __getattr__(self, api_name: str):
        def invoke(**kwargs):
            fields = kwargs.pop("fields", "")
            return self.query(api_name, fields=fields, **kwargs)

        return invoke


def _call(method: str, **kwargs) -> pd.DataFrame:
    client = _pro_client()
    try:
        result = getattr(client, method)(**kwargs)
    except Exception as exc:  # Tushare SDK exposes vendor errors as generic exceptions.
        message = str(exc)
        if _rate_limit_message(message):
            raise VendorRateLimitError(f"Tushare rate limit for {method}: {message}") from exc
        if _permission_message(message):
            raise VendorNotConfiguredError(
                f"Tushare permission/token issue for {method}: {message}"
            ) from exc
        raise
    if result is None:
        return pd.DataFrame()
    if not isinstance(result, pd.DataFrame):
        return pd.DataFrame(result)
    return result


def _point_in_time_rows(frame: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    """Keep only rows known by ``curr_date``, preferring announcement dates."""
    if frame.empty or not curr_date:
        return frame
    cutoff = _compact_date(curr_date)
    for column in ("ann_date", "f_ann_date", "trade_date", "end_date"):
        if column in frame.columns:
            values = frame[column].fillna("").astype(str)
            frame = frame[(values != "") & (values <= cutoff)]
            break
    return frame


def _csv_report(title: str, frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return f"{title}\n\nNo rows returned."
    data = frame.head(max_rows).copy()
    for column in data.columns:
        if column.endswith("date") or column == "trade_date":
            data[column] = data[column].map(_display_date)
    return f"{title}\n\n{data.to_csv(index=False)}"


def fetch_daily_frame(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    allow_index: bool = False,
) -> pd.DataFrame:
    ts_code = normalize_ts_code(ticker)
    params = {
        "ts_code": ts_code,
        "start_date": _compact_date(start_date),
        "end_date": _compact_date(end_date),
    }
    frame = _call("daily", **params)
    if frame.empty and allow_index:
        frame = _call("index_daily", **params)
    if frame.empty:
        raise NoMarketDataError(ticker, ts_code, f"Tushare returned no rows from {start_date} to {end_date}")
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame


def get_stock_data(ticker: str, start_date: str, end_date: str) -> str:
    frame = fetch_daily_frame(ticker, start_date, end_date)
    columns = [
        column
        for column in (
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "pct_chg",
            "vol",
            "amount",
        )
        if column in frame.columns
    ]
    return _csv_report(
        f"# Tushare A-share daily data for {normalize_ts_code(ticker)} "
        f"from {start_date} to {end_date} (volume unit: lots; amount unit: CNY thousands)",
        frame[columns],
        max_rows=6000,
    )


def get_indicators(ticker: str, indicator: str, curr_date: str, look_back_days: int) -> str:
    if indicator not in SUPPORTED_INDICATORS:
        raise ValueError(
            f"Unsupported Tushare indicator {indicator!r}; choose from {sorted(SUPPORTED_INDICATORS)}"
        )
    end = datetime.strptime(curr_date, "%Y-%m-%d")
    history_days = max(260, int(look_back_days) * 4)
    start = end - timedelta(days=history_days)
    frame = fetch_daily_frame(
        ticker,
        start.strftime("%Y-%m-%d"),
        curr_date,
    )
    stock_frame = frame.rename(columns={"trade_date": "date"}).copy()
    stock_frame["date"] = pd.to_datetime(stock_frame["date"], format="%Y%m%d")
    stock_frame = wrap(stock_frame)
    stock_frame[indicator]
    output = stock_frame[["date", indicator]].tail(max(1, int(look_back_days))).copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    return _csv_report(
        f"# {indicator} for {normalize_ts_code(ticker)} through {curr_date}",
        output,
        max_rows=max(1, int(look_back_days)),
    )


def _stock_basic(ts_code: str) -> pd.DataFrame:
    return _call(
        "stock_basic",
        ts_code=ts_code,
        fields=(
            "ts_code,symbol,name,area,industry,market,exchange,curr_type,"
            "list_status,list_date,delist_date,is_hs,act_name,act_ent_type"
        ),
    )


def resolve_instrument_identity(ticker: str) -> dict[str, str]:
    """Resolve an A-share identity using Tushare; fail open for preflight use."""
    try:
        ts_code = normalize_ts_code(ticker)
        frame = _stock_basic(ts_code)
    except Exception:
        return {}
    if frame.empty:
        return {}
    row = frame.iloc[0]
    return {
        key: str(value)
        for key, value in {
            "company_name": row.get("name"),
            "industry": row.get("industry"),
            "sector": row.get("market"),
            "exchange": row.get("exchange"),
            "area": row.get("area"),
        }.items()
        if value is not None and str(value) not in ("", "nan", "None")
    }


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    ts_code = normalize_ts_code(ticker)
    basic = _stock_basic(ts_code)
    if basic.empty:
        raise NoMarketDataError(ticker, ts_code, "stock_basic returned no company identity")

    end = curr_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=45)).strftime("%Y-%m-%d")
    daily_basic = _call(
        "daily_basic",
        ts_code=ts_code,
        start_date=_compact_date(start),
        end_date=_compact_date(end),
        fields=(
            "ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps_ttm,"
            "dv_ttm,total_share,float_share,total_mv,circ_mv"
        ),
    ).sort_values("trade_date", ascending=False)

    fina_start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=900)).strftime("%Y-%m-%d")
    indicators = _call(
        "fina_indicator",
        ts_code=ts_code,
        start_date=_compact_date(fina_start),
        end_date=_compact_date(end),
        fields=(
            "ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,capital_rese_ps,"
            "surplus_rese_ps,undist_profit_ps,extra_item,profit_dedt,gross_margin,current_ratio,"
            "quick_ratio,roe,roe_dt,roa,debt_to_assets,netprofit_yoy,or_yoy"
        ),
    )
    indicators = _point_in_time_rows(indicators, curr_date)
    indicator_sort = [
        column for column in ("ann_date", "end_date") if column in indicators.columns
    ]
    if indicator_sort:
        indicators = indicators.sort_values(indicator_sort, ascending=False)

    return "\n\n".join(
        (
            _csv_report(f"# Tushare company profile for {ts_code}", basic, max_rows=1),
            _csv_report("# Latest point-in-time daily valuation", daily_basic, max_rows=1),
            _csv_report("# Published financial indicators", indicators, max_rows=8),
        )
    )


def _financial_statement(
    method: str,
    title: str,
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    ts_code = normalize_ts_code(ticker)
    end = curr_date or datetime.now().strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=1500)).strftime("%Y-%m-%d")
    statement_fields = {
        "balancesheet": (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_share,"
            "money_cap,accounts_receiv,inventories,total_cur_assets,total_assets,"
            "total_cur_liab,total_liab,total_hldr_eqy_exc_min_int"
        ),
        "cashflow": (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,n_cashflow_act,"
            "n_cashflow_inv_act,n_cash_flows_fnc_act,n_incr_cash_cash_equ,free_cashflow"
        ),
        "income": (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_revenue,"
            "revenue,operate_profit,total_profit,n_income,n_income_attr_p,ebit,ebitda,"
            "basic_eps,diluted_eps"
        ),
    }
    frame = _call(
        method,
        ts_code=ts_code,
        start_date=_compact_date(start),
        end_date=_compact_date(end),
        fields=statement_fields[method],
    )
    frame = _point_in_time_rows(frame, curr_date)
    if frame.empty:
        raise NoMarketDataError(ticker, ts_code, f"{method} returned no published rows")
    sort_columns = [column for column in ("ann_date", "end_date") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, ascending=False)
    if freq.lower() == "annual" and "end_date" in frame.columns:
        frame = frame[frame["end_date"].astype(str).str.endswith("1231")]
    return _csv_report(f"# Tushare {title} for {ts_code}", frame, max_rows=8)


def get_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _financial_statement("balancesheet", "balance sheet", ticker, freq, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _financial_statement("cashflow", "cash flow statement", ticker, freq, curr_date)


def get_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _financial_statement("income", "income statement", ticker, freq, curr_date)


def _news_frame(start_date: str, end_date: str) -> pd.DataFrame:
    source = get_config().get("tushare_news_source", "sina")
    start = f"{start_date} 00:00:00"
    end = f"{end_date} 23:59:59"
    return _call(
        "news",
        src=source,
        start_date=start,
        end_date=end,
        fields="datetime,title,content,channels",
    )


def _company_name(ts_code: str) -> str:
    frame = _stock_basic(ts_code)
    if frame.empty:
        return ""
    return str(frame.iloc[0].get("name") or "")


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    ts_code = normalize_ts_code(ticker)
    frame = _news_frame(start_date, end_date)
    if frame.empty:
        raise NoMarketDataError(ticker, ts_code, "Tushare news returned no rows")
    company_name = _company_name(ts_code)
    code = ts_code.split(".")[0]
    text = frame.fillna("").astype(str).agg(" ".join, axis=1)
    matched = frame[text.str.contains(re.escape(company_name or code), case=False, regex=True)]
    if company_name:
        matched = frame[
            text.str.contains(re.escape(company_name), case=False, regex=True)
            | text.str.contains(re.escape(code), case=False, regex=True)
        ]
    if matched.empty:
        return f"No Tushare news matched {company_name or ts_code} between {start_date} and {end_date}."
    limit = int(get_config().get("news_article_limit", 20))
    return _csv_report(
        f"# Tushare news for {company_name or ts_code} from {start_date} to {end_date}",
        matched,
        max_rows=limit,
    )


def get_global_news(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    config = get_config()
    look_back_days = look_back_days or int(config.get("global_news_lookback_days", 7))
    limit = limit or int(config.get("global_news_article_limit", 10))
    end = datetime.strptime(curr_date, "%Y-%m-%d")
    start = end - timedelta(days=look_back_days)
    frame = _news_frame(start.strftime("%Y-%m-%d"), curr_date)
    return _csv_report(
        f"# Tushare China market news through {curr_date}",
        frame,
        max_rows=limit,
    )


def get_insider_transactions(ticker: str, curr_date: str | None = None) -> str:
    ts_code = normalize_ts_code(ticker)
    end = datetime.strptime(curr_date, "%Y-%m-%d") if curr_date else datetime.now()
    start = end - timedelta(days=730)
    frame = _call(
        "stk_holdertrade",
        ts_code=ts_code,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if frame.empty:
        return f"No Tushare shareholder/management trades found for {ts_code}."
    return _csv_report(f"# Tushare shareholder and management trades for {ts_code}", frame)


def get_macro_data(indicator: str, curr_date: str, look_back_days: int = 365) -> str:
    """Fetch a small set of China-focused macro series from Tushare."""
    normalized = indicator.strip().lower()
    end = datetime.strptime(curr_date, "%Y-%m-%d")
    start = end - timedelta(days=max(look_back_days, 90))
    handlers: dict[str, tuple[str, dict]] = {
        "cpi": ("cn_cpi", {"start_m": start.strftime("%Y%m"), "end_m": end.strftime("%Y%m")}),
        "ppi": ("cn_ppi", {"start_m": start.strftime("%Y%m"), "end_m": end.strftime("%Y%m")}),
        "gdp": ("cn_gdp", {"start_q": f"{start.year}Q1", "end_q": f"{end.year}Q4"}),
        "shibor": (
            "shibor",
            {"start_date": start.strftime("%Y%m%d"), "end_date": end.strftime("%Y%m%d")},
        ),
    }
    if normalized not in handlers:
        return (
            f"DATA_UNAVAILABLE: Tushare A-share profile supports macro indicators "
            f"{sorted(handlers)}, not {indicator!r}."
        )
    method, params = handlers[normalized]
    frame = _call(method, **params)
    return _csv_report(f"# Tushare China macro indicator: {normalized}", frame, max_rows=30)


def fetch_close_series(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    allow_index: bool = False,
) -> pd.Series:
    frame = fetch_daily_frame(ticker, start_date, end_date, allow_index=allow_index)
    return frame.set_index("trade_date")["close"].astype(float)


def clear_client_cache() -> None:
    """Testing/configuration hook for changing TUSHARE_TOKEN in-process."""
    _pro_client.cache_clear()
