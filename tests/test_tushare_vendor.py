from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.tushare as tushare_vendor
from tradingagents.dataflows.errors import VendorNotConfiguredError


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("600519", "600519.SH"),
        ("000001", "000001.SZ"),
        ("300750.SZ", "300750.SZ"),
        ("SH600519", "600519.SH"),
        ("830799", "830799.BJ"),
    ],
)
def test_normalize_ts_code(raw, expected):
    assert tushare_vendor.normalize_ts_code(raw) == expected


@pytest.mark.unit
def test_missing_token_has_actionable_error(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    tushare_vendor.clear_client_cache()
    with pytest.raises(VendorNotConfiguredError, match="TUSHARE_TOKEN"):
        tushare_vendor._pro_client()


@pytest.mark.unit
def test_stock_data_is_sorted_and_formatted(monkeypatch):
    frame = pd.DataFrame(
        [
            {"ts_code": "600519.SH", "trade_date": "20260106", "open": 2, "high": 3,
             "low": 1, "close": 2.5, "pre_close": 2, "pct_chg": 25, "vol": 20,
             "amount": 50},
            {"ts_code": "600519.SH", "trade_date": "20260105", "open": 1, "high": 2,
             "low": 1, "close": 2, "pre_close": 1, "pct_chg": 100, "vol": 10,
             "amount": 20},
        ]
    )
    monkeypatch.setattr(tushare_vendor, "_call", lambda method, **kwargs: frame.copy())

    result = tushare_vendor.get_stock_data("600519", "2026-01-01", "2026-01-10")
    assert "600519.SH" in result
    assert result.index("2026-01-05") < result.index("2026-01-06")


@pytest.mark.unit
def test_point_in_time_filter_uses_announcement_date():
    frame = pd.DataFrame(
        [
            {"ann_date": "20250101", "value": 1},
            {"ann_date": "20250301", "value": 2},
            {"ann_date": "", "value": 3},
        ]
    )
    result = tushare_vendor._point_in_time_rows(frame, "2025-02-01")
    assert result["value"].tolist() == [1]


@pytest.mark.unit
def test_rate_limit_is_mapped_to_vendor_error(monkeypatch):
    client = mock.Mock()
    client.daily.side_effect = RuntimeError("每分钟最多访问该接口 50 次")
    monkeypatch.setattr(tushare_vendor, "_pro_client", lambda: client)
    with pytest.raises(tushare_vendor.VendorRateLimitError):
        tushare_vendor._call("daily", ts_code="600519.SH")


@pytest.mark.unit
def test_http_client_converts_tushare_payload_to_frame(monkeypatch):
    response = mock.Mock()
    response.json.return_value = {
        "code": 0,
        "data": {
            "fields": ["ts_code", "close"],
            "items": [["600519.SH", 1500.0]],
        },
    }
    monkeypatch.setattr(tushare_vendor.requests, "post", lambda *args, **kwargs: response)
    client = tushare_vendor._TushareHttpClient(
        token="secret-token",
        base_url="https://api.tushare.pro",
    )
    frame = client.daily(ts_code="600519.SH", fields="ts_code,close")
    assert frame.to_dict("records") == [{"ts_code": "600519.SH", "close": 1500.0}]


@pytest.mark.unit
def test_http_permission_code_is_mapped(monkeypatch):
    response = mock.Mock()
    response.json.return_value = {"code": 2002, "msg": "没有接口访问权限", "data": None}
    monkeypatch.setattr(tushare_vendor.requests, "post", lambda *args, **kwargs: response)
    client = tushare_vendor._TushareHttpClient(
        token="secret-token",
        base_url="https://api.tushare.pro",
    )
    monkeypatch.setattr(tushare_vendor, "_pro_client", lambda: client)
    with pytest.raises(VendorNotConfiguredError, match="permission/token"):
        tushare_vendor._call("daily", ts_code="600519.SH")
