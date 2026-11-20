import os
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

# Complete 29-ticker Money Flow & Rotation watchlist from your dashboard
WATCHLIST = [
    # Sectors (11)
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
    # Indices (7)
    ("SPY", "S&P 500", "Index"),
    ("QQQ", "Nasdaq 100", "Index"),
    ("IWM", "Russell 2000", "Index"),
    ("DIA", "Dow", "Index"),
    ("EFA", "Developed ex-US", "Index"),
    ("EEM", "Emerging Markets", "Index"),
    ("EWU", "UK", "Index"),
    # Bonds (5)
    ("TLT", "20Y+ Treasuries", "Bond"),
    ("IEF", "7-10Y Treasuries", "Bond"),
    ("SHY", "1-3Y Treasuries", "Bond"),
    ("LQD", "IG Credit", "Bond"),
    ("HYG", "High Yield", "Bond"),
    # Commodities (4)
    ("GLD", "Gold", "Commodity"),
    ("SLV", "Silver", "Commodity"),
    ("USO", "Oil", "Commodity"),
    ("DBC", "Broad Commodities", "Commodity"),
    # FX / Crypto (2)
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
    return headlines[:2]


def call_groq_with_fallback(prompt: str, max_tokens: int = 650) -> str:
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
    payload = {"username": "Sector Money Flow AI", "content": content[:1950]}
    r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=12)
    if not r.ok:
        raise RuntimeError(f"Discord webhook error HTTP {r.status_code}: {r.text[:250]}")
    print("✅ Sent full 29-ticker rotation & AI recommendation summary to Discord.")


def run_daily_sector_flow():
    print("==================================================")
    print("  💸 FULL 29-TICKER MONEY FLOW & ROTATION AI")
    print("==================================================")

    tickers = [t for t, _, _ in WATCHLIST]
    raw = yf.download(tickers, period="2mo", interval="1d", group_by="ticker", progress=False, auto_adjust=False)

    spy_df = raw["SPY"].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
    spy_close = _to_series(spy_df, "Close")
    spy_ret_5d = ((float(spy_close.iloc[-1]) - float(spy_close.iloc[-6])) / float(spy_close.iloc[-6])) * 100.0 if len(spy_close) >= 6 else 0.0

    rows = []
    for ticker, label, group in WATCHLIST:
        try:
            df = raw[ticker].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw.dropna(how="all")
            close = _to_series(df, "Close")
            vol = _to_series(df, "Volume")
            if len(close) < 21:
                continue

            c_now = float(close.iloc[-1])
            c_1d = float(close.iloc[-2])
            c_5d = float(close.iloc[-6])
            c_20d = float(close.iloc[-21])

            ret_1d = ((c_now - c_1d) / c_1d) * 100.0
            ret_5d = ((c_now - c_5d) / c_5d) * 100.0
            ret_20d = ((c_now - c_20d) / c_20d) * 100.0
            vs_spy_5d = ret_5d - spy_ret_5d

            avg_vol_10 = float(vol.tail(11).iloc[:-1].mean()) if len(vol) >= 11 else float(vol.mean())
            rvol_1d = (float(vol.iloc[-1]) / avg_vol_10) if avg_vol_10 > 0 else 1.0
            cmf_20 = compute_cmf(df, 20)

            # Composite Flow Score mirroring your dashboard's Flow score / 1D / 5D / 20D / 5D vs SPY / CMF(20)
            flow_score = round((ret_1d * 0.35) + (ret_5d * 0.25) + (vs_spy_5d * 0.20) + (cmf_20 * 10.0 * 0.20), 2)
            is_reversal = (ret_5d < -0.5) and (ret_1d > 0.0)

            rows.append({
                "ticker": ticker,
                "label": label,
                "group": group,
                "ret_1d": round(ret_1d, 2),
                "ret_5d": round(ret_5d, 2),
                "ret_20d": round(ret_20d, 2),
                "vs_spy_5d": round(vs_spy_5d, 2),
                "rvol_1d": round(rvol_1d, 2),
                "cmf_20": round(cmf_20, 3),
                "flow_score": flow_score,
                "is_reversal": is_reversal,
            })
        except Exception as e:
            print(f"Metric warning {ticker}: {e!r}")

    if not rows:
        print("No ETF data retrieved.")
        return

    # Fetch headlines across all 29 tickers so Groq pairs every asset's flow with news
    news_map = {r["ticker"]: fetch_headlines(r["ticker"]) for r in rows}

    reversals = sorted([r for r in rows if r["is_reversal"]], key=lambda x: x["ret_1d"], reverse=True)
    top_1d = sorted(rows, key=lambda x: x["ret_1d"], reverse=True)
    bottom_1d = sorted(rows, key=lambda x: x["ret_1d"])

    prompt = (
        "You are an institutional ETF flow & macro strategist analyzing ALL 29 assets on a Money Flow & Rotation dashboard "
        "(11 US Sectors, 7 Indices, 5 Bonds, 4 Commodities, 2 FX/Crypto).\n"
        "Look at 1D%, 5D%, 20D%, 5D vs SPY, CMF(20), Flow Score, and recent headlines for EVERY ticker.\n"
        "Specifically identify:\n"
        "- Where 5D outflows are flipping into 1D inflows (gradual institutional/market-maker accumulation),\n"
        "- Macro & news drivers behind the rotation,\n"
        "- The BEST sector/ETF to move INTO right now, and the WORST sectors/assets to move OUT OF.\n\n"
        f"All 29 Assets Flow Table:\n{json.dumps(rows)}\n\n"
        f"Recent Headlines Across All 29 Assets:\n{json.dumps(news_map)}\n\n"
        "Respond in clean Discord markdown (under 1100 characters total) with EXACTLY these 4 sections:\n"
        "**🟢 BEST SECTOR / ETF TO MOVE INTO:** `<TICKER>` (<Name>) — 1-2 sentences pairing 5D->1D flow, CMF/RVOL, and news.\n"
        "**🔄 RUNNER-UP ACCUMULATION WATCH:** `<TICKER>` (<Name>) — 1 sentence on secondary rotation candidate.\n"
        "**🔴 SECTORS / ASSETS TO MOVE OUT OF:** `<TICKER1>`, `<TICKER2>`, `<TICKER3>` — 1-2 sentences on heavy outflows & news headwinds.\n"
        "**🧭 MACRO & MARKET-MAKER TAKE:** 1-2 sentences on risk-on vs risk-off regime and one invalidation/stop note."
    )

    ai_verdict = call_groq_with_fallback(prompt, max_tokens=550)
    if not ai_verdict:
        best_rev = reversals[0] if reversals else top_1d[0]
        worst = bottom_1d[0]
        ai_verdict = (
            f"**🟢 BEST SECTOR / ETF TO MOVE INTO:** `{best_rev['ticker']}` ({best_rev['label']}) — "
            f"5D `{best_rev['ret_5d']:+.2f}%` → 1D `{best_rev['ret_1d']:+.2f}%` (CMF `{best_rev['cmf_20']:+.2f}`).\n"
            f"**🔴 SECTORS / ASSETS TO MOVE OUT OF:** `{worst['ticker']}` ({worst['label']}) — "
            f"1D `{worst['ret_1d']:+.2f}%` / 5D `{worst['ret_5d']:+.2f}%`."
        )

    rev_summary = ", ".join(
        [f"`{r['ticker']}` ({r['ret_5d']:+.1f}%→{r['ret_1d']:+.1f}%)" for r in reversals[:5]]
    ) or "None today"

    top3_str = " | ".join([f"`{r['ticker']}` `{r['ret_1d']:+.2f}%`" for r in top_1d[:4]])
    bot3_str = " | ".join([f"`{r['ticker']}` `{r['ret_1d']:+.2f}%`" for r in bottom_1d[:4]])

    discord_msg = (
        f"💸 **Daily 29-Asset Money Flow & Rotation ({datetime.utcnow().strftime('%Y-%m-%d')})**\n"
        f"• **Top 1D Inflows:** {top3_str}\n"
        f"• **Worst 1D Outflows:** {bot3_str}\n"
        f"• **5D→1D Reversals:** {rev_summary}\n\n"
        f"{ai_verdict}"
    )

    post_discord(discord_msg)


if __name__ == "__main__":
    run_daily_sector_flow()
