import os
import re
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Exact watchlist from your Money Flow & Rotation dashboard
WATCHLIST = [
    # Sectors
    ("XLK", "Technology", "Sector"),
    ("XLV", "Health Care", "Sector"),
    ("XLF", "Financials", "Sector"),
    ("XLE", "Energy", "Sector"),
    ("XLI", "Industrials", "Sector"),
    ("XLY", "Discretionary", "Sector"),
    ("XLP", "Staples", "Sector"),
    ("XLU", "Utilities", "Sector"),
    ("XLB", "Materials", "Sector"),
    ("XLRE", "Real Estate", "Sector"),
    ("XLC", "Communication", "Sector"),
    # Indices
    ("SPY", "S&P 500", "Index"),
    ("QQQ", "Nasdaq 100", "Index"),
    ("IWM", "Russell 2000", "Index"),
    ("DIA", "Dow", "Index"),
    ("EFA", "Developed ex-US", "Index"),
    ("EEM", "Emerging Markets", "Index"),
    ("EWU", "UK", "Index"),
    # Bonds
    ("TLT", "20Y+ Treasuries", "Bond"),
    ("IEF", "7-10Y Treasuries", "Bond"),
    ("SHY", "1-3Y Treasuries", "Bond"),
    ("LQD", "IG Credit", "Bond"),
    ("HYG", "High Yield", "Bond"),
    # Commodities
    ("GLD", "Gold", "Commodity"),
    ("SLV", "Silver", "Commodity"),
    ("USO", "Oil", "Commodity"),
    ("DBC", "Broad Commodities", "Commodity"),
    # FX / Crypto
    ("UUP", "US Dollar", "FX/Crypto"),
    ("BTC-USD", "Bitcoin", "FX/Crypto"),
]


def _to_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def compute_cmf(df: pd.DataFrame, period: int = 20) -> float:
    high = _to_series(df, "High")
    low = _to_series(df, "Low")
    close = _to_series(df, "Close")
    vol = _to_series(df, "Volume")
    aligned = pd.concat([high, low, close, vol], axis=1).dropna()
    aligned.columns = ["High", "Low", "Close", "Volume"]
    if len(aligned) < period:
        return 0.0
    hl_range = (aligned["High"] - aligned["Low"]).replace(0, np.nan)
    mfm = ((aligned["Close"] - aligned["Low"]) - (aligned["High"] - aligned["Close"])) / hl_range
    mfm = mfm.fillna(0.0)
    mfv = mfm * aligned["Volume"]
    denom = aligned["Volume"].tail(period).sum()
    if denom <= 0:
        return 0.0
    return float(mfv.tail(period).sum() / denom)


def fetch_headlines(ticker: str) -> list[str]:
    headlines = []
    try:
        if FINNHUB_API_KEY and "-" not in ticker:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            frm = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": frm, "to": today, "token": FINNHUB_API_KEY},
                timeout=8,
            )
            if r.status_code == 200:
                for item in r.json()[:5]:
                    h = (item.get("headline") or "").strip()
                    if h and h not in headlines:
                        headlines.append(h)
        if not headlines:
            for item in (yf.Ticker(ticker).news or [])[:5]:
                h = (item.get("content") or {}).get("title") or item.get("title")
                if h and h.strip() not in headlines:
                    headlines.append(h.strip())
    except Exception as e:
        print(f"News warning {ticker}: {e!r}")
    return headlines[:3]


def call_groq_with_fallback(prompt: str, max_tokens: int = 450) -> str:
    if GROQ_API_KEY:
        try:
            from groq import Groq
            res = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            txt = (res.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception as e:
            print(f"Groq failed: {e!r}")
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            txt = (genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text or "").strip()
            if txt:
                return txt
        except Exception as e:
            print(f"Gemini fallback failed: {e!r}")
    return ""


def post_discord(content: str):
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not configured; printing summary below:\n")
        print(content)
        return
    # Discord hard limit is 2000 chars
    payload = {"username": "Sector Money Flow AI", "content": content[:1950]}
    r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=12)
    if not r.ok:
        raise RuntimeError(f"Discord webhook error HTTP {r.status_code}: {r.text[:250]}")
    print("✅ Sent sector money flow summary to Discord.")


def run_daily_sector_flow():
    print("==================================================")
    print("  💸 DAILY SECTOR MONEY FLOW & GROQ ROTATION SCAN")
    print("==================================================")

    tickers = [t for t, _, _ in WATCHLIST]
    raw = yf.download(tickers, period="2mo", interval="1d", group_by="ticker", progress=False, auto_adjust=False)

    rows = []
    for ticker, label, group in WATCHLIST:
        try:
            df = raw[ticker].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
            close = _to_series(df, "Close")
            vol = _to_series(df, "Volume")
            if len(close) < 7:
                continue

            c_now = float(close.iloc[-1])
            c_1d = float(close.iloc[-2])
            c_5d = float(close.iloc[-6])

            ret_1d = ((c_now - c_1d) / c_1d) * 100.0
            ret_5d = ((c_now - c_5d) / c_5d) * 100.0

            # Volume-weighted flow proxy & 20-day Chaikin Money Flow
            avg_vol_10 = float(vol.tail(11).iloc[:-1].mean()) if len(vol) >= 11 else float(vol.mean())
            rvol_1d = (float(vol.iloc[-1]) / avg_vol_10) if avg_vol_10 > 0 else 1.0
            cmf_20 = compute_cmf(df, 20)

            # Reversal score: strong negative 5D turning positive on 1D with volume support
            is_reversal = (ret_5d < -0.5) and (ret_1d > 0.0)
            reversal_spread = ret_1d - (ret_5d / 5.0)

            rows.append({
                "ticker": ticker,
                "label": label,
                "group": group,
                "price": round(c_now, 2),
                "ret_1d": round(ret_1d, 2),
                "ret_5d": round(ret_5d, 2),
                "rvol_1d": round(rvol_1d, 2),
                "cmf_20": round(cmf_20, 3),
                "is_reversal": is_reversal,
                "reversal_spread": round(reversal_spread, 2),
            })
        except Exception as e:
            print(f"Metric warning {ticker}: {e!r}")

    if not rows:
        print("No ETF data retrieved.")
        return

    reversals = sorted(
        [r for r in rows if r["is_reversal"]],
        key=lambda x: (x["ret_1d"], x["reversal_spread"]),
        reverse=True,
    )
    top_1d = sorted(rows, key=lambda x: x["ret_1d"], reverse=True)[:5]
    bottom_1d = sorted(rows, key=lambda x: x["ret_1d"])[:4]

    focus_tickers = []
    for item in (reversals[:5] + top_1d[:3]):
        if item["ticker"] not in focus_tickers:
            focus_tickers.append(item["ticker"])

    news_map = {t: fetch_headlines(t) for t in focus_tickers[:6]}

    prompt = (
        "You are an institutional ETF flow & macro strategist. Analyze this daily ETF rotation data where "
        "we look for 5-day outflows flipping into 1-day inflows (gradual institutional/market-maker accumulation) "
        "alongside macro/fundamental catalysts.\n\n"
        f"5D-to-1D Inflow Reversals: {json.dumps(reversals[:6])}\n"
        f"Top 1D Inflows Overall: {json.dumps(top_1d)}\n"
        f"Weakest 1D Outflows: {json.dumps(bottom_1d)}\n"
        f"Recent Headlines: {json.dumps(news_map)}\n\n"
        "Write a crisp, practical 3-4 bullet commentary for a Discord message explaining:\n"
        "1) Which 1-2 ETFs show the cleanest institutional rotation / 5D->1D reversal backed by news & fundamentals,\n"
        "2) Whether the tape favors risk-on (cyclicals/tech) or risk-off/defensive (staples/health care/utilities/bonds),\n"
        "3) One key risk/whipsaw warning to watch before rotating capital."
    )

    ai_commentary = call_groq_with_fallback(prompt, max_tokens=420)
    if not ai_commentary:
        ai_commentary = "• Check top 5D→1D reversals above alongside CMF(20) and relative volume before rotating."

    rev_lines = [
        f"• `{r['ticker']}` ({r['label']}): **5D `{r['ret_5d']:+.2f}%` → 1D `{r['ret_1d']:+.2f}%`** (RVOL `{r['rvol_1d']:.1f}x` | CMF `{r['cmf_20']:+.2f}`)"
        for r in reversals[:5]
    ] or ["• None (no negative 5D ETFs flipped positive on 1D today)"]

    lead_lines = [
        f"• `{r['ticker']}` ({r['label']}): **1D `{r['ret_1d']:+.2f}%`** (5D `{r['ret_5d']:+.2f}%`)"
        for r in top_1d[:4]
    ]

    lag_lines = [
        f"• `{r['ticker']}` ({r['label']}): **1D `{r['ret_1d']:+.2f}%`** (5D `{r['ret_5d']:+.2f}%`)"
        for r in bottom_1d[:3]
    ]

    discord_msg = (
        f"💸 **Daily Sector Money Flow & Rotation ({datetime.utcnow().strftime('%Y-%m-%d')})**\n\n"
        f"🔄 **5D Outflow → 1D Inflow Reversals (Accumulation Watch)**\n"
        + "\n".join(rev_lines)
        + "\n\n🟢 **Top 1D Inflow Leaders**\n"
        + "\n".join(lead_lines)
        + "\n\n🔴 **Heaviest 1D Outflows**\n"
        + "\n".join(lag_lines)
        + "\n\n🤖 **Groq AI Flow & Fundamentals Take**\n"
        + ai_commentary
    )

    post_discord(discord_msg)


if __name__ == "__main__":
    run_daily_sector_flow()
