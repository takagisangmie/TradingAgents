# A-share Setup

The default market profile is `a_share`. It uses Tushare for prices,
indicators, company identity, point-in-time financial statements, company/news
feeds, shareholder trades, and China macro series.

## Credentials

Create `.env` from `.env.example` and set:

```dotenv
TUSHARE_TOKEN=your-token
```

The CLI prompts for a missing token and stores it in the ignored `.env` file.
Never commit the token. A compatible Tushare relay can be selected with
`TUSHARE_API_URL`; the default is `https://api.tushare.pro`.

Tushare endpoints have account-specific point and frequency requirements. A
missing token, insufficient permission, or rate limit is converted to a typed
vendor error and shown as an information-audit warning.

## Symbols

- Shanghai: `600519.SH`, `688981.SH`
- Shenzhen/ChiNext: `000001.SZ`, `300750.SZ`
- Beijing Exchange: `830799.BJ`
- CSI 300 benchmark: `000300.SH`

Bare six-digit codes and exchange-prefixed forms such as `SH600519` are
normalized to Tushare format.

## Community Sentiment

Xueqiu and TaoGuBa adapters request one public stock page per process/ticker and
cache the result. Optional `XUEQIU_COOKIE` and `TAOGUBA_COOKIE` values can carry
the user's own authenticated browser session. The adapters do not solve
CAPTCHAs, replay WAF challenges, rotate identities, or bypass 403/418/429
responses.

When a historical analysis date is selected, current community posts are not
used. The report receives an explicit historical-snapshot warning instead,
preventing look-ahead bias.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
tradingagents
```

The default cloud model is `glm-5.2` through BigModel China, which requires
`ZHIPU_CN_API_KEY` in addition to `TUSHARE_TOKEN`. For local LLMs, select Ollama
or an OpenAI-compatible endpoint.
