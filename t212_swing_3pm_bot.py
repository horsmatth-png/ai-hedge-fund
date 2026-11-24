import os
import re
import json
import time
import base64
from datetime import datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Environment Variables & Secrets
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

# Trading 212 Credentials (matches your GitHub Secrets: TRADING_212_API & TRADING_212_SECRET_API)
T212_API_KEY = os.getenv("TRADING_212_API") or os.getenv("T212_API_KEY")
T212_SECRET_KEY = os.getenv("TRADING_212_SECRET_API") or os.getenv("T212_SECRET_KEY")
T212_BASE_URL = os.getenv("T212_BASE_URL", "https://demo.trading212.com/api/v0")
T212_DRY_RUN = os.getenv("T212_DRY_RUN", "false").lower() in ("1", "true", "yes")

# Dedicated Swing Trade State File + Instrument Cache
SWING_STATE_FILE = "swing_trades.json"
INSTRUMENT_CACHE_FILE = "t212_instruments_cache.json"

# £1,000 Account & Daily Trade Cap (up to 3 trades per day max; no lifetime open-position cap)
ACCOUNT_SIZE_GBP = float(os.getenv("ACCOUNT_SIZE_GBP", "1000.0"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))        # £10 max risk per trade on £1k
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "30.0"))          # Max £300 notional per trade
MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "3"))               # Up to 3 trades per day across UK+US
MAX_OPEN_EXPOSURE_GBP = float(os.getenv("MAX_OPEN_EXPOSURE_GBP", "950.0"))
REQUIRE_STOP_LOSS = True

# Foolproof IG Stake Floors (if IG automation shares this config module)
IG_MIN_STAKE_UK = float(os.getenv("IG_MIN_STAKE_UK", "1.0"))
IG_MIN_STAKE_US = float(os.getenv("IG_MIN_STAKE_US", "0.24"))


# AI & Scanner Config
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "35"))
AI_CHERRYPICK_COUNT = int(os.getenv("AI_CHERRYPICK_COUNT", "5"))
MAX_PER_SECTOR = 1
MEGA_CAP_CAP = 2

UA = {"User-Agent": "Mozilla/5.0 (compatible; quant-t212-swing-bot/1.0)"}
MEGA_CAPS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO"}

REGION_CFG = {
    "US": {
        "tz": "America/New_York",
        "ccy": "$",
        "scale": 1.0,
        "min_price": 5.0,
        "min_value": 25e6,
        "bot": "T212 Swing+Day AI (£1k)",
    },
    "UK": {
        "tz": "Europe/London",
        "ccy": "£",
        "scale": 0.01,   # Yahoo LSE prices are typically in pence (GBp)
        "min_price": 100.0,
        "min_value": 3e6,
        "bot": "T212 Swing+Day AI (£1k)",
    },
}

FALLBACK_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO", "JPM", "BAC",
    "GS", "V", "MA", "UNH", "LLY", "JNJ", "ABBV", "CAT", "GE", "RTX", "LMT", "COST"
]
FALLBACK_UK = [
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L",
    "REL.L", "LLOY.L", "BARC.L", "VOD.L", "GLEN.L", "NG.L", "RR.L", "BAE.L"
]

SHOCK_RE = re.compile(
    r"\b(resigns?|resignation|investigation|fraud|subpoena|lawsuit|sec probe|"
    r"step(s|ping)? down|recall|bankruptcy|halted|restat\w+|short report|profit warning)\b",
    re.I,
)


def load_json(path: str, default: dict) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return dict(default)
    return dict(default)


def save_json(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def load_swing_state() -> dict:
    return load_json(SWING_STATE_FILE, {})


def save_swing_state(state: dict):
    save_json(SWING_STATE_FILE, state)


def post_discord(username: str, content: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"username": username, "content": content},
            timeout=10,
        )
    except Exception as e:
        print(f"Discord webhook warning: {e!r}")


def _to_float(val) -> float:
    if isinstance(val, pd.Series):
        val = val.dropna().iloc[-1] if not val.dropna().empty else np.nan
    elif isinstance(val, np.ndarray):
        val = val.ravel()[-1] if val.size > 0 else np.nan
    return float(val)


def get_fx_to_gbp() -> float:
    """Returns USD-to-GBP multiplier."""
    try:
        df = yf.download("GBPUSD=X", period="5d", interval="1d", progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            close_col = df["Close"]
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            gbp_usd = _to_float(close_col.dropna())
            if gbp_usd > 0:
                return 1.0 / gbp_usd
    except Exception as e:
        print(f"FX lookup warning (GBPUSD=X): {e!r}")
    return 0.78


def _wiki_constituents(url: str, suffix: str = "") -> dict:
    try:
        html = requests.get(url, headers=UA, timeout=20).text
        for tbl in pd.read_html(StringIO(html)):
            cols = {str(c).lower(): c for c in tbl.columns}
            sym_col = next((cols[k] for k in cols if k in ("symbol", "ticker", "epic")), None)
            sec_col = next((cols[k] for k in cols if "sector" in k or "industry" in k), None)
            if sym_col is None or len(tbl) < 30:
                continue
            out = {}
            for _, row in tbl.iterrows():
                sym = str(row[sym_col]).strip().rstrip(".").replace(".", "-")
                if not sym or sym.lower() == "nan":
                    continue
                out[sym + suffix] = str(row[sec_col]) if sec_col is not None else "Unknown"
            return out
    except Exception as e:
        print(f"Wiki fetch warning: {e!r}")
    return {}


def get_universe(region: str) -> dict:
    try:
        if region == "US":
            uni = {}
            uni.update(_wiki_constituents("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"))
            uni.update(_wiki_constituents("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"))
        else:
            uni = _wiki_constituents("https://en.wikipedia.org/wiki/FTSE_100_Index", suffix=".L")
        if len(uni) >= 50:
            print(f"Universe ({region}): {len(uni)} tickers loaded")
            return uni
    except Exception as e:
        print(f"Universe fetch failed: {e!r}")
    fb = FALLBACK_US if region == "US" else FALLBACK_UK
    print(f"Using fallback universe ({len(fb)} tickers)")
    return {t: "Unknown" for t in fb}


def _split(raw, chunk):
    out = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        top = set(raw.columns.get_level_values(0))
        for t in chunk:
            if t in top:
                d = raw[t].dropna(how="all")
                if not d.empty:
                    out[t] = d
    else:
        out[chunk[0]] = raw.dropna(how="all")
    return out


def batch_download(tickers, chunk_size=100, **kw):
    result = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(chunk, group_by="ticker", progress=False, threads=True, auto_adjust=False, **kw)
            result.update(_split(raw, chunk))
        except Exception as e:
            print(f"Batch download error: {e!r}")
    return result


def wilder_rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    val = (100 - 100 / (1 + rs)).iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def atr(df: pd.DataFrame, n: int = 14) -> float:
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def analyze_swing_and_day_fundamentals(ticker: str, df_daily: pd.DataFrame) -> tuple[bool, dict]:
    """
    Collects both long-term swing fundamentals and daily swing/momentum fundamentals
    for AI evaluation and deterministic baseline scoring.
    """
    info = {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
    except Exception as e:
        print(f"Fundamental info warning {ticker}: {e!r}")

    close = df_daily["Close"].dropna()
    price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
    rsi_d = wilder_rsi(close, 14)

    if len(close) >= 50 and price < sma50 * 0.95:
        return False, {"reason": "below 50d swing trend", "rsi_d": round(rsi_d, 1)}

    debt_to_equity = info.get("debtToEquity")
    free_cash_flow = info.get("freeCashflow")
    operating_cash_flow = info.get("operatingCashflow")
    profit_margins = info.get("profitMargins")
    revenue_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")
    return_on_equity = info.get("returnOnEquity")
    forward_pe = info.get("forwardPE")

    if debt_to_equity is not None and float(debt_to_equity) > 260.0:
        return False, {"reason": f"high D/E ({debt_to_equity})", "rsi_d": round(rsi_d, 1)}

    if free_cash_flow is not None and operating_cash_flow is not None:
        if float(free_cash_flow) < 0 and float(operating_cash_flow) < 0:
            return False, {"reason": "negative FCF & OCF", "rsi_d": round(rsi_d, 1)}

    if profit_margins is not None and float(profit_margins) < -0.15:
        return False, {"reason": "deeply negative margin", "rsi_d": round(rsi_d, 1)}

    f_score = 0
    f_notes = []
    if len(close) >= 200 and price >= sma200:
        f_score += 25
        f_notes.append("above SMA200")
    elif price >= sma50:
        f_score += 15
        f_notes.append("above SMA50")

    if free_cash_flow is not None and float(free_cash_flow) > 0:
        f_score += 25
        f_notes.append("positive FCF")
    if debt_to_equity is None or float(debt_to_equity) <= 120.0:
        f_score += 20
        f_notes.append("moderate leverage")
    if profit_margins is None or float(profit_margins) >= 0.05:
        f_score += 15
        f_notes.append("healthy margins")
    if revenue_growth is not None and float(revenue_growth) > 0.03:
        f_score += 15
        f_notes.append(f"rev growth {float(revenue_growth)*100:+.1f}%")

    return True, {
        "fund_score": f_score,
        "fund_notes": f_notes,
        "sma50": round(sma50, 4),
        "sma200": round(sma200, 4),
        "rsi_d": round(rsi_d, 1),
        "debt_to_equity": debt_to_equity,
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow": operating_cash_flow,
        "profit_margins": profit_margins,
        "revenue_growth": revenue_growth,
        "earnings_growth": earnings_growth,
        "return_on_equity": return_on_equity,
        "forward_pe": forward_pe,
    }


def score_pullback_setup(ticker: str, d5: pd.DataFrame, dd: pd.DataFrame, region: str, sector: str, fund_meta: dict):
    """
    Evaluates intraday EMA + VWAP pullback timing alongside daily swing context
    for a given region's session date.
    """
    d5 = d5.dropna(subset=["Close", "High", "Low", "Volume"])
    if d5.empty:
        return None

    tz = ZoneInfo(REGION_CFG[region]["tz"])
    idx = d5.index
    if getattr(idx, "tzinfo", None) is None:
        idx = idx.tz_localize("UTC").tz_convert(tz)
    else:
        idx = idx.tz_convert(tz)
    d5 = d5.copy()
    d5.index = idx

    session_today = datetime.now(tz).date()
    dates = np.array([i.date() for i in d5.index])
    tdf = d5[dates == session_today]
    if len(tdf) < 3:
        # Fallback to most recent session bar if pre/after-hours or timezone edge
        last_date = dates[-1]
        tdf = d5[dates == last_date]
        session_today = last_date
    if len(tdf) < 3:
        return None

    hist_days = sorted(set(dates) - {session_today})[-9:]
    if len(hist_days) < 2:
        return None

    close_5m = d5["Close"]
    ema20_5m = float(close_5m.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50_5m = float(close_5m.ewm(span=50, adjust=False).mean().iloc[-1])

    price = float(tdf["Close"].iloc[-1])
    prev_bar_close = float(tdf["Close"].iloc[-2])
    low_last = float(tdf["Low"].iloc[-1])
    last_time = tdf.index[-1].time()

    typical = (tdf["High"] + tdf["Low"] + tdf["Close"]) / 3.0
    cum_vol = max(float(tdf["Volume"].cumsum().iloc[-1]), 1.0)
    vwap = float((typical * tdf["Volume"]).cumsum().iloc[-1] / cum_vol)

    cum_today = float(tdf["Volume"].sum())
    cum_hist = []
    for d in hist_days:
        day = d5[dates == d]
        cum_hist.append(float(day[[i.time() <= last_time for i in day.index]]["Volume"].sum()))
    avg_cum = np.mean([c for c in cum_hist if c > 0]) if any(c > 0 for c in cum_hist) else 0.0
    rel_vol = cum_today / avg_cum if avg_cum > 0 else 1.0

    hist_d = dd[[i.date() != session_today for i in dd.index]]
    if len(hist_d) < 50:
        hist_d = dd
    a_daily = atr(hist_d, 14)

    dist_vwap_pct = (price - vwap) / vwap * 100.0 if vwap > 0 else 0.0
    dist_ema20_pct = (price - ema20_5m) / ema20_5m * 100.0 if ema20_5m > 0 else 0.0

    score = float(fund_meta.get("fund_score", 50)) * 0.45
    why = list(fund_meta.get("fund_notes", []))

    if ema20_5m > ema50_5m:
        score += 15
        why.append("5m EMA20>EMA50")
    if price >= vwap * 0.997:
        score += 15
        why.append("above/at VWAP")

    near_vwap_pullback = (-0.9 <= dist_vwap_pct <= 0.65) or (low_last <= vwap * 1.002 and price >= vwap * 0.998)
    near_ema_pullback = (-0.9 <= dist_ema20_pct <= 0.65) or (low_last <= ema20_5m * 1.002 and price >= ema20_5m * 0.998)

    if (near_vwap_pullback or near_ema_pullback) and price >= prev_bar_close:
        score += 25
        why.append("VWAP/EMA pullback bounce")
    elif dist_vwap_pct > 2.2 or dist_ema20_pct > 2.2:
        score -= 15
        why.append("extended above VWAP/EMA")

    if rel_vol >= 0.9:
        score += min(rel_vol, 2.5) / 2.5 * 15
        why.append(f"RVOL {rel_vol:.1f}x")

    rsi_d = float(fund_meta.get("rsi_d", 55.0))
    if 42.0 <= rsi_d <= 68.0:
        score += 10
        why.append(f"dRSI {rsi_d:.0f}")
    elif rsi_d > 76.0:
        score -= 15
        why.append(f"dRSI {rsi_d:.0f} overbought")

    if score < 38:
        return None

    return {
        "ticker": ticker,
        "region": region,
        "sector": sector,
        "side": "LONG",
        "score": round(score, 1),
        "why": why,
        "price": price,
        "vwap": round(vwap, 4),
        "ema20_5m": round(ema20_5m, 4),
        "ema50_5m": round(ema50_5m, 4),
        "rsi_d": rsi_d,
        "rel_vol": round(rel_vol, 2),
        "atr": a_daily,
        "fundamentals": fund_meta,
    }


def fetch_news(ticker: str, region: str) -> list:
    headlines = []
    try:
        if region == "US" and FINNHUB_API_KEY:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            frm = (datetime.utcnow() - timedelta(days=4)).strftime("%Y-%m-%d")
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": frm, "to": today, "token": FINNHUB_API_KEY},
                timeout=8,
            )
            if r.status_code == 200:
                tk = ticker.upper()
                for a in r.json():
                    title = (a.get("headline") or "").strip()
                    related = [x.strip().upper() for x in (a.get("related") or "").split(",") if x.strip()]
                    about_it = (tk in related and len(related) <= 3) or re.search(rf"{re.escape(tk)}", title, re.I)
                    if title and about_it and title not in headlines:
                        headlines.append(title)
                    if len(headlines) >= 5:
                        break
        else:
            for item in (yf.Ticker(ticker).news or [])[:6]:
                title = (item.get("content") or {}).get("title") or item.get("title")
                if title:
                    headlines.append(title.strip())
    except Exception as e:
        print(f"News error {ticker}: {e!r}")
    return headlines[:5]


def _call_ai(prompt: str, max_tokens: int = 400) -> str:
    # Try Groq first (avoids Gemini free-tier daily quota exhaustion)
    if GROQ_API_KEY:
        try:
            from groq import Groq
            out = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            text = (out.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"Groq failed: {e!r}")
    # Fallback to Gemini if Groq is unavailable or fails
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            text = (genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"Gemini failed: {e!r}")
    return ""


def ai_cherrypick_opportunities(candidates: list, max_picks: int = 5) -> list:
    """
    Uses AI to cherry-pick the strongest UK & US opportunities balancing:
    - Short-term day-trade pullback timing (VWAP/EMA20/RVOL)
    - Medium-term swing trend & daily momentum
    - Long-term fundamentals & news catalysts/risks
    Returns ordered list of selected candidate dicts with AI rationale attached.
    """
    if not candidates:
        return []

    compact = []
    for idx, c in enumerate(candidates[:18]):
        f = c.get("fundamentals", {})
        compact.append({
            "idx": idx,
            "ticker": c["ticker"],
            "region": c["region"],
            "sector": c["sector"],
            "price": round(c["price"], 2),
            "vwap": round(c["vwap"], 2),
            "ema20_5m": round(c["ema20_5m"], 2),
            "rvol": c["rel_vol"],
            "rsi_d": c["rsi_d"],
            "fund_score": f.get("fund_score", 0),
            "fcf": f.get("free_cash_flow"),
            "de": f.get("debt_to_equity"),
            "rev_growth": f.get("revenue_growth"),
            "margins": f.get("profit_margins"),
            "news": c.get("_news", [])[:3],
        })

    prompt = (
        "You are an equity portfolio strategist running a £1,000 long-only UK & US strategy at 3pm UK time. "
        "Cherry-pick the top opportunities that combine:\n"
        "1) Short-term day-trade pullback quality (VWAP/EMA pullback bounce + relative volume),\n"
        "2) Swing & long-term fundamentals (cash flow, balance sheet, growth, trend),\n"
        "3) News catalysts & news risk (favor constructive catalysts, penalize legal/regulatory/earnings headwinds).\n\n"
        f"Candidates JSON:\n{json.dumps(compact)}\n\n"
        "Return STRICT JSON only (no markdown fences) with schema:\n"
        '{"picks": [{"idx": <int>, "horizon": "SHORT_AND_LONG" | "SWING_LONG" | "DAY_PULLBACK", "reason": "<1 concise sentence>"}]}\n'
        f"Select at most {max_picks} best names in priority order."
    )

    raw_text = _call_ai(prompt, max_tokens=500)
    selected = []
    used_indices = set()

    if raw_text:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.I | re.M)
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
                for item in parsed.get("picks", []):
                    i = item.get("idx")
                    if isinstance(i, int) and 0 <= i < len(candidates[:18]) and i not in used_indices:
                        c = dict(candidates[i])
                        c["ai_horizon"] = str(item.get("horizon") or "SHORT_AND_LONG")
                        c["ai_reason"] = str(item.get("reason") or "Strong multi-timeframe & fundamental alignment.")
                        selected.append(c)
                        used_indices.add(i)
                        if len(selected) >= max_picks:
                            break
            except Exception as e:
                print(f"AI cherry-pick JSON parse warning: {e!r}")

    if not selected:
        for i, c in enumerate(candidates[:max_picks]):
            item = dict(c)
            item["ai_horizon"] = "SHORT_AND_LONG"
            item["ai_reason"] = "Ranked by fundamental + VWAP/EMA pullback score and clean news."
            selected.append(item)

    return selected


def ai_position_check(ticker: str, qty: float, entry_gbp: float, curr_gbp: float, stop_gbp: float, target_gbp: float, headlines: list) -> tuple[bool, str]:
    pnl_pct = ((curr_gbp - entry_gbp) / entry_gbp * 100.0) if entry_gbp else 0.0
    prompt = (
        f"You are a risk analyst auditing a LONG swing/pullback stock position in {ticker} ({qty:.4f} sh).\n"
        f"Entry: £{entry_gbp:.2f}, Current: £{curr_gbp:.2f} ({pnl_pct:+.2f}%), "
        f"Stop: £{stop_gbp:.2f}, Target: £{target_gbp:.2f}.\n"
        f"Headlines (short & long-term impact): {' | '.join(headlines) or 'No breaking news'}\n\n"
        "Start with 'HEALTHY:' or 'AT_RISK:'. Then write 1 concise sentence on thesis & news risk. No HTML."
    )
    text = _call_ai(prompt, max_tokens=140)
    if not text:
        return False, f"HEALTHY: Position at {pnl_pct:+.2f}% with no shock news."
    return text.upper().startswith("AT_RISK"), text


def pick_diversified(cands: list, n: int) -> list:
    picks, sectors = [], {}
    mega = 0
    for c in cands:
        if sectors.get(c["sector"], 0) >= MAX_PER_SECTOR and c["sector"] != "Unknown":
            continue
        if c["ticker"] in MEGA_CAPS and mega >= MEGA_CAP_CAP:
            continue
        picks.append(c)
        sectors[c["sector"]] = sectors.get(c["sector"], 0) + 1
        mega += c["ticker"] in MEGA_CAPS
        if len(picks) >= n:
            return picks
    for c in cands:
        if c not in picks and not (c["ticker"] in MEGA_CAPS and mega >= MEGA_CAP_CAP):
            picks.append(c)
            mega += c["ticker"] in MEGA_CAPS
        if len(picks) >= n:
            break
    return picks[:n]


def build_position_plan(pick: dict, region: str, fx_to_gbp: float, remaining_budget_gbp: float) -> dict | None:
    """
    Calculates position size within the £1,000 account budget:
    - Risk per trade = 1.0% of £1,000 = £10 max loss
    - Max notional per trade = 30.0% of £1,000 = £300 (also capped by remaining £1k budget)
    - Stop distance = max(1.25 * ATR, 2.5% of price) below entry
    - Target = entry + 2.5 * stop_distance
    """
    if remaining_budget_gbp < 20.0:
        return None

    unit_to_gbp = REGION_CFG[region]["scale"] * (fx_to_gbp if region == "US" else 1.0)
    price_gbp = pick["price"] * unit_to_gbp
    if price_gbp <= 0:
        return None

    stop_dist_native = max(1.25 * pick["atr"], pick["price"] * 0.025)
    stop_dist_gbp = stop_dist_native * unit_to_gbp
    if stop_dist_gbp <= 0:
        return None

    max_risk_gbp = ACCOUNT_SIZE_GBP * (RISK_PER_TRADE_PCT / 100.0)
    max_notional_gbp = min(ACCOUNT_SIZE_GBP * (MAX_POSITION_PCT / 100.0), remaining_budget_gbp)

    qty_by_risk = max_risk_gbp / stop_dist_gbp
    qty_by_cap = max_notional_gbp / price_gbp
    qty = round(min(qty_by_risk, qty_by_cap), 4)

    if qty <= 0 or (qty * price_gbp) < 20.0:
        return None

    # Note: T212 UK equity stopPrice is quoted in GBP (£), whereas Yahoo LSE prices are in pence (GBp)
    entry_native = round(pick["price"], 2)
    stop_native = round(max(0.01, entry_native - stop_dist_native), 2)
    target_native = round(entry_native + 2.5 * stop_dist_native, 2)

    if not (stop_native < entry_native < target_native):
        return None

    return {
        "quantity": qty,
        "entry": entry_native,
        "stop": stop_native,
        "target": target_native,
        "entry_gbp": round(entry_native * unit_to_gbp, 2),
        "stop_gbp": round(stop_native * unit_to_gbp, 2),
        "target_gbp": round(target_native * unit_to_gbp, 2),
        "position_gbp": round(qty * price_gbp, 2),
        "est_risk_gbp": round(qty * stop_dist_gbp, 2),
    }


# ==============================================================================
# Trading 212 Invest API Client (HTTP Basic Auth: API_KEY:SECRET_KEY)
# ==============================================================================

class Trading212Client:
    def __init__(self, api_key: str, secret_key: str, base_url: str, dry_run: bool = False):
        self.api_key = (api_key or "").strip()
        self.secret_key = (secret_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run

        if self.api_key and self.secret_key:
            token = base64.b64encode(f"{self.api_key}:{self.secret_key}".encode("utf-8")).decode("utf-8")
            auth_header = f"Basic {token}"
        else:
            auth_header = self.api_key

        self.headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def _request(self, method: str, path: str, payload: dict | None = None):
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self.headers, json=payload, timeout=15)
        if not resp.ok:
            raise RuntimeError(f"T212 {method} {path} HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.text:
            return resp.json()
        return {}

    def get_account_summary(self) -> dict:
        return self._request("GET", "/equity/account/summary")

    def get_open_positions(self) -> list:
        data = self._request("GET", "/equity/portfolio")
        return data if isinstance(data, list) else []

    def get_pending_orders(self) -> list:
        data = self._request("GET", "/equity/orders")
        return data if isinstance(data, list) else []

    def cancel_order(self, order_id) -> bool:
        if self.dry_run:
            print(f"[DRY_RUN] Would cancel T212 order {order_id}")
            return True
        self._request("DELETE", f"/equity/orders/{order_id}")
        return True

    _instruments_cache_mem = None

    def _get_instruments_list(self) -> list:
        if Trading212Client._instruments_cache_mem is not None:
            return Trading212Client._instruments_cache_mem
        # Rate-limit friendly: fetch once and cache in memory + disk
        all_cache = load_json(INSTRUMENT_CACHE_FILE, {})
        instruments = all_cache.get("_all_instruments")
        if isinstance(instruments, list) and instruments:
            Trading212Client._instruments_cache_mem = instruments
            return instruments
        time.sleep(1.5)
        fetched = self._request("GET", "/equity/metadata/instruments")
        if isinstance(fetched, list) and fetched:
            Trading212Client._instruments_cache_mem = fetched
            all_cache["_all_instruments"] = fetched
            save_json(INSTRUMENT_CACHE_FILE, all_cache)
            return fetched
        return []

    def resolve_t212_ticker(self, yf_ticker: str, region: str) -> str | None:
        cache = load_json(INSTRUMENT_CACHE_FILE, {})
        if yf_ticker in cache:
            return cache[yf_ticker]

        guess = f"{yf_ticker.replace('.L', '')}l_EQ" if region == "UK" else f"{yf_ticker}_US_EQ"
        if not self.is_configured():
            return guess

        try:
            instruments = self._get_instruments_list()
            if instruments:
                clean = yf_ticker.replace(".L", "").upper()
                for inst in instruments:
                    t212_sym = str(inst.get("ticker", ""))
                    short_name = str(inst.get("shortName", "")).upper()
                    if region == "UK" and yf_ticker.endswith(".L"):
                        if (short_name == clean or t212_sym.startswith(clean)) and (
                            "_UK_" in t212_sym or t212_sym.endswith("l_EQ")
                        ):
                            cache[yf_ticker] = t212_sym
                            save_json(INSTRUMENT_CACHE_FILE, cache)
                            return t212_sym
                    elif region == "US":
                        if short_name == clean and "_US_" in t212_sym:
                            cache[yf_ticker] = t212_sym
                            save_json(INSTRUMENT_CACHE_FILE, cache)
                            return t212_sym
        except Exception as e:
            print(f"⚠️ T212 instrument lookup warning for {yf_ticker}: {e!r}")

        # Fallback to standard T212 symbol format if metadata endpoint rate-limits
        return guess

    @staticmethod
    def _clean_qty(quantity: float, decimals: int = 4) -> float:
        return round(abs(float(quantity)), decimals)

    def place_long_market_buy(self, t212_ticker: str, quantity: float) -> dict:
        if not REQUIRE_STOP_LOSS:
            raise ValueError("REQUIRE_STOP_LOSS must remain True")
        qty = self._clean_qty(quantity, 4)
        if self.dry_run or not self.is_configured():
            fake_id = f"SIM-BUY-{int(datetime.utcnow().timestamp())}"
            print(f"[DRY_RUN] T212 BUY {qty} {t212_ticker} (MARKET) -> {fake_id}")
            return {"id": fake_id, "status": "SIMULATED"}
        # Try 4 decimals first, then 2 decimals if T212 rejects quantity precision
        for prec in (4, 2):
            q = self._clean_qty(quantity, prec)
            try:
                return self._request("POST", "/equity/orders/market", {"ticker": t212_ticker, "quantity": q})
            except RuntimeError as err:
                if "quantity-precision-mismatch" in str(err) and prec == 4:
                    continue
                raise

    def place_protective_stop_sell(self, t212_ticker: str, quantity: float, stop_price: float) -> dict:
        """
        Places a broker-side SELL STOP order on Trading 212 immediately after buy fill,
        with retry & verification so no trade is left unprotected.
        """
        if stop_price is None or float(stop_price) <= 0:
            raise ValueError(f"Refusing to place trade without valid stop_price on {t212_ticker}")
        qty = self._clean_qty(quantity, 4)
        if qty <= 0:
            raise ValueError(f"Refusing to place stop order with zero quantity on {t212_ticker}")
        rounded_stop = round(float(stop_price), 2)
        if self.dry_run or not self.is_configured():
            fake_id = f"SIM-STOP-{int(datetime.utcnow().timestamp())}"
            print(f"[DRY_RUN] T212 SELL STOP {-qty} {t212_ticker} @ {rounded_stop} -> {fake_id}")
            return {"id": fake_id, "status": "SIMULATED"}

        last_err = None
        for attempt in range(3):
            try:
                res = self._request(
                    "POST",
                    "/equity/orders/stop",
                    {
                        "ticker": t212_ticker,
                        "quantity": -qty,
                        "stopPrice": rounded_stop,
                        "timeValidity": "GOOD_TILL_CANCEL",
                    },
                )
                return res
            except Exception as err:
                last_err = err
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to place protective SELL STOP on {t212_ticker} @ {rounded_stop}: {last_err}")

    def place_market_sell(self, t212_ticker: str, quantity: float) -> dict:
        qty = self._clean_qty(quantity, 4)
        if self.dry_run or not self.is_configured():
            fake_id = f"SIM-SELL-{int(datetime.utcnow().timestamp())}"
            print(f"[DRY_RUN] T212 SELL {-qty} {t212_ticker} (MARKET) -> {fake_id}")
            return {"id": fake_id, "status": "SIMULATED"}
        return self._request("POST", "/equity/orders/market", {"ticker": t212_ticker, "quantity": -qty})


def _scan_region_candidates(region: str) -> list:
    cfg = REGION_CFG[region]
    universe = get_universe(region)
    daily = batch_download(list(universe), period="8mo", interval="1d")
    if not daily:
        return []

    fund_passed = {}
    shortlist_pool = []
    for t, df in daily.items():
        try:
            if len(df) < 60:
                continue
            price = float(df["Close"].iloc[-1])
            if price < cfg["min_price"]:
                continue
            avg_val = float((df["Close"] * df["Volume"]).tail(20).mean()) * cfg["scale"]
            if avg_val < cfg["min_value"]:
                continue
            ok, meta = analyze_swing_and_day_fundamentals(t, df)
            if ok:
                fund_passed[t] = meta
                shortlist_pool.append((t, meta.get("fund_score", 0)))
        except Exception as e:
            print(f"Fundamental screen error {t} ({region}): {e!r}")

    shortlist_pool.sort(key=lambda x: x[1], reverse=True)
    shortlist = [t for t, _ in shortlist_pool[:SHORTLIST_SIZE]]
    print(f"🏛️ [{region}] Fundamental Screen: {len(shortlist_pool)} passed -> 5m pullback scan on top {len(shortlist)}")
    if not shortlist:
        return []

    intraday_5m = batch_download(shortlist, chunk_size=25, period="10d", interval="5m", prepost=False)
    scored = []
    for t in shortlist:
        if t in intraday_5m and t in daily:
            r = score_pullback_setup(t, intraday_5m[t], daily[t], region, universe.get(t, "Unknown"), fund_passed[t])
            if r:
                scored.append(r)

    scored.sort(key=lambda x: (x["score"], x["rel_vol"]), reverse=True)

    clean = []
    for r in scored[:15]:
        heads = fetch_news(r["ticker"], region)
        if any(SHOCK_RE.search(h) for h in heads):
            print(f"Dropped {r['ticker']} ({region}) due to shock headline")
            continue
        r["_news"] = heads
        clean.append(r)

    return clean


def execute_swing_3pm_uk_us():
    print("==================================================")
    print("  🕒 3PM UK+US SWING & DAY PULLBACK AI CHERRYPICKER")
    print("==================================================")

    t212 = Trading212Client(T212_API_KEY, T212_SECRET_KEY, T212_BASE_URL, dry_run=T212_DRY_RUN)
    print(f"💼 Account Cap: £{ACCOUNT_SIZE_GBP:.0f} | Daily Trade Limit: {MAX_DAILY_TRADES}/day | Dry-Run: {t212.dry_run}")

    if t212.is_configured() and not t212.dry_run:
        try:
            summary = t212.get_account_summary()
            print(f"🔌 T212 Connected | Account Summary: {summary}")
        except Exception as e:
            print(f"❌ T212 connection failed: {e}")
            return

    state = load_swing_state()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if state.get("date") != today_str:
        state = {
            "date": today_str,
            "mode": "T212_SWING_3PM_UK_US_1K",
            "trades_executed_today": 0,
            "open_swing_trades": state.get("open_swing_trades", {}),
            "risk_alerts": state.get("risk_alerts", {}),
        }

    open_trades = state.setdefault("open_swing_trades", {})
    trades_today = int(state.get("trades_executed_today", 0))

    if trades_today >= MAX_DAILY_TRADES:
        print(f"🛑 Daily trade cap ({MAX_DAILY_TRADES}/day) already reached for {today_str}.")
        save_swing_state(state)
        return

    current_exposure_gbp = sum(float(p.get("position_gbp") or 0.0) for p in open_trades.values())
    remaining_budget_gbp = max(0.0, MAX_OPEN_EXPOSURE_GBP - current_exposure_gbp)
    if t212.is_configured() and not t212.dry_run:
        try:
            acct_sum = t212.get_account_summary()
            avail_cash = float(((acct_sum.get("cash") or {}).get("availableToTrade")) or 0.0)
            if avail_cash >= 0:
                remaining_budget_gbp = min(remaining_budget_gbp, avail_cash)
        except Exception:
            pass
    if remaining_budget_gbp < 20.0:
        print(f"🛑 £{ACCOUNT_SIZE_GBP:.0f} swing budget utilized (£{current_exposure_gbp:.2f} open).")
        save_swing_state(state)
        return

    uk_cands = _scan_region_candidates("UK")
    us_cands = _scan_region_candidates("US")

    combined = sorted(uk_cands + us_cands, key=lambda x: (x["score"], x["rel_vol"]), reverse=True)
    diversified = pick_diversified(combined, 18)
    cherrypicked = ai_cherrypick_opportunities(diversified, max_picks=AI_CHERRYPICK_COUNT)

    print(f"🤖 AI Cherry-Picked ({len(cherrypicked)}): {[(c['ticker'], c['region'], c['ai_horizon']) for c in cherrypicked]}")
    if not cherrypicked:
        save_swing_state(state)
        return

    fx_to_gbp = get_fx_to_gbp()

    for pick in cherrypicked:
        if trades_today >= MAX_DAILY_TRADES:
            print(f"🛑 Reached daily limit of {MAX_DAILY_TRADES} trades for {today_str}.")
            break

        ticker = pick["ticker"]
        region = pick["region"]
        cfg = REGION_CFG[region]

        if ticker in open_trades:
            print(f"🔁 Skipping {ticker}: already in {SWING_STATE_FILE}.")
            continue

        plan = build_position_plan(pick, region, fx_to_gbp, remaining_budget_gbp)
        if not plan:
            print(f"⏭️ Skipping {ticker}: outside £{ACCOUNT_SIZE_GBP:.0f} sizing/budget constraints.")
            continue

        if REQUIRE_STOP_LOSS and not (plan["stop"] > 0 and plan["stop"] < plan["entry"]):
            print(f"⛔ Refusing {ticker}: invalid stop loss.")
            continue

        t212_ticker = t212.resolve_t212_ticker(ticker, region)
        if not t212_ticker:
            print(f"⚠️ Could not resolve T212 ticker for {ticker}, skipping.")
            continue

        # T212 UK equities expect stopPrice in GBP (£), while US equities expect USD ($)
        broker_stop_price = plan["stop_gbp"] if region == "UK" else plan["stop"]

        buy_res = None
        filled_qty = plan["quantity"]
        try:
            buy_res = t212.place_long_market_buy(t212_ticker, plan["quantity"])
            if not t212.dry_run and t212.is_configured():
                # Poll portfolio up to ~12s for market order fill so stop order sees owned shares
                owned_qty = 0.0
                for _ in range(4):
                    time.sleep(3.0)
                    try:
                        positions = t212.get_open_positions()
                        for p in positions:
                            if str(p.get("ticker")) == t212_ticker:
                                owned_qty = float(p.get("quantity") or 0.0)
                                break
                    except Exception:
                        pass
                    if owned_qty > 0:
                        break
                if owned_qty <= 0:
                    raise RuntimeError(
                        f"Buy order {buy_res.get('id')} on {t212_ticker} not yet filled/settled in portfolio (market closed or pending)"
                    )
                filled_qty = round(min(plan["quantity"], owned_qty), 4)
            stop_res = t212.place_protective_stop_sell(t212_ticker, filled_qty, broker_stop_price)
            if not stop_res or not stop_res.get("id"):
                raise RuntimeError(f"Stop order response missing ID on {t212_ticker}: {stop_res}")
            if not t212.dry_run and t212.is_configured():
                time.sleep(1.5)
                try:
                    pending = t212.get_pending_orders()
                    stop_id_str = str(stop_res.get("id"))
                    has_stop = any(
                        str(o.get("id")) == stop_id_str or (str(o.get("ticker")) == t212_ticker and float(o.get("quantity") or 0) < 0)
                        for o in pending
                    )
                    if not has_stop:
                        print(f"⚠️ Stop {stop_id_str} not yet visible in /equity/orders for {t212_ticker}; re-checking on surveillance.")
                except Exception as chk_err:
                    print(f"⚠️ Pending order check warning on {t212_ticker}: {chk_err!r}")
        except Exception as e:
            print(f"⚠️ Order error on {ticker} ({t212_ticker}): {e!r}")
            # Only attempt emergency market sell if we actually hold shares in the portfolio
            if not t212.dry_run and t212.is_configured():
                try:
                    positions = t212.get_open_positions()
                    held = next((float(p.get("quantity") or 0.0) for p in positions if str(p.get("ticker")) == t212_ticker), 0.0)
                    if held > 0:
                        t212.place_market_sell(t212_ticker, held)
                except Exception as close_err:
                    print(f"🚨 Emergency close failed on {t212_ticker}: {close_err!r}")
            continue

        open_trades[ticker] = {
            "ticker": ticker,
            "t212_ticker": t212_ticker,
            "region": region,
            "horizon": pick.get("ai_horizon", "SHORT_AND_LONG"),
            "ai_reason": pick.get("ai_reason", ""),
            "side": "BUY",
            "quantity": filled_qty,
            "entry": plan["entry"],
            "stop": plan["stop"],
            "target": plan["target"],
            "entry_gbp": plan["entry_gbp"],
            "stop_gbp": plan["stop_gbp"],
            "target_gbp": plan["target_gbp"],
            "position_gbp": plan["position_gbp"],
            "buy_order_id": buy_res.get("id"),
            "stop_order_id": stop_res.get("id"),
            "opened_at": datetime.utcnow().isoformat(),
        }
        trades_today += 1
        state["trades_executed_today"] = trades_today
        remaining_budget_gbp = max(0.0, remaining_budget_gbp - plan["position_gbp"])

        print(
            f"🚀 SWING BUY [{region}] {ticker} ({t212_ticker}): {plan['quantity']} sh @ {cfg['ccy']}{plan['entry']:.2f} "
            f"(£{plan['position_gbp']:.2f} | Stop {cfg['ccy']}{plan['stop']:.2f} | Target {cfg['ccy']}{plan['target']:.2f})"
        )

        post_discord(
            cfg["bot"],
            (
                f"📈 **3PM Swing+Day AI Buy [{region}]**: `{ticker}` (`{t212_ticker}`) | `{pick.get('ai_horizon')}`\n"
                f"• **Size:** `{plan['quantity']} sh` (`£{plan['position_gbp']:.2f}` / `£{ACCOUNT_SIZE_GBP:.0f}` cap) | **Daily Count:** `{trades_today}/{MAX_DAILY_TRADES}`\n"
                f"• **Entry:** `{cfg['ccy']}{plan['entry']:.2f}` | **Broker Stop:** `{cfg['ccy']}{plan['stop']:.2f}` | **Target:** `{cfg['ccy']}{plan['target']:.2f}`\n"
                f"• **AI Pick:** {pick.get('ai_reason', '')}"
            ),
        )

    save_swing_state(state)


def monitor_swing_trades():
    print("==================================================")
    print("  🔍 SWING TRADES & NEWS/AI RISK SURVEILLANCE")
    print("==================================================")

    state = load_swing_state()
    open_trades = state.get("open_swing_trades", {})
    risk_alerts = state.setdefault("risk_alerts", {})
    now_utc = datetime.utcnow()

    t212 = Trading212Client(T212_API_KEY, T212_SECRET_KEY, T212_BASE_URL, dry_run=T212_DRY_RUN)

    live_pos = []
    live_tickers = None
    live_order_ids = None
    if t212.is_configured() and not t212.dry_run:
        try:
            live_pos = t212.get_open_positions()
            live_tickers = {str(p.get("ticker")) for p in live_pos}
            live_orders = t212.get_pending_orders()
            live_order_ids = {str(o.get("id")) for o in live_orders}
        except Exception as e:
            print(f"⚠️ T212 sync warning: {e!r}")

    # Auto-import any open T212 positions not yet recorded in open_swing_trades
    imported_any = False
    if live_pos:
        tracked_t212_syms = {str(v.get("t212_ticker") or k) for k, v in open_trades.items()}
        for p in live_pos:
            t212_sym = str(p.get("ticker") or "")
            qty_live = float(p.get("quantity") or 0.0)
            avg_price = float(p.get("averagePrice") or 0.0)
            if not t212_sym or qty_live <= 0 or avg_price <= 0:
                continue
            if t212_sym in tracked_t212_syms:
                continue
            # Derive clean display/yfinance ticker & region from T212 ticker suffix
            if t212_sym.endswith("_UK_EQ"):
                region = "UK"
                base_sym = t212_sym[:-6]
                yf_sym = f"{base_sym}.L"
            elif t212_sym.endswith("l_EQ") and not t212_sym.endswith("_US_EQ"):
                region = "UK"
                base_sym = t212_sym[:-4]
                if base_sym.endswith("l"):
                    base_sym = base_sym[:-1]
                yf_sym = f"{base_sym}.L"
            else:
                region = "US"
                base_sym = t212_sym[:-6] if t212_sym.endswith("_US_EQ") else t212_sym.split("_")[0]
                yf_sym = base_sym

            # T212 averagePrice is already in GBP for UK equities (or native currency for US).
            # Use a conservative 3.5% stop below averagePrice and 5.0% target above averagePrice
            # in the instrument's native price units so `curr_price` vs `stop`/`target` are apples-to-apples!
            stop_dist_native = max(avg_price * 0.035, 0.02)
            stop_native_init = round(max(0.01, avg_price - stop_dist_native), 2)
            target_native_init = round(avg_price * 1.05, 2)
            fx_now = get_fx_to_gbp() if region == "US" else 1.0
            entry_gbp_init = round(avg_price * fx_now, 2)
            stop_gbp_init = round(stop_native_init * fx_now, 2)
            target_gbp_init = round(target_native_init * fx_now, 2)
            pos_gbp_init = round(qty_live * entry_gbp_init, 2)

            open_trades[yf_sym] = {
                "ticker": yf_sym,
                "t212_ticker": t212_sym,
                "region": region,
                "horizon": "SYNCED_PORTFOLIO",
                "ai_reason": "Auto-imported from live T212 portfolio for profit/stop surveillance.",
                "side": "BUY",
                "quantity": round(qty_live, 4),
                "entry": round(avg_price, 2),
                "stop": stop_native_init,
                "target": target_native_init,
                "entry_gbp": entry_gbp_init,
                "stop_gbp": stop_gbp_init,
                "target_gbp": target_gbp_init,
                "position_gbp": pos_gbp_init,
                "buy_order_id": None,
                "stop_order_id": None,
                "opened_at": now_utc.isoformat(),
            }
            imported_any = True
            print(f"📥 Imported T212 position {yf_sym} ({t212_sym}): {qty_live} sh @ £{avg_price:.2f} | Target £{target_gbp_init:.2f} | Stop £{stop_gbp_init:.2f}")

    if imported_any:
        state["open_swing_trades"] = open_trades
        save_swing_state(state)

    if not open_trades:
        print(f"📭 No open swing trades in {SWING_STATE_FILE}.")
        return

    tickers = list(open_trades.keys())
    latest = batch_download(tickers, chunk_size=20, period="5d", interval="1d")
    fx_to_gbp = get_fx_to_gbp()

    to_close = []
    dirty = False

    for ticker, pos in open_trades.items():
        region = pos.get("region", "US")
        cfg = REGION_CFG[region]
        qty = float(pos["quantity"])
        entry = float(pos["entry"])
        stop = float(pos["stop"])
        target = float(pos["target"])
        t212_ticker = pos.get("t212_ticker", ticker)
        stop_order_id = pos.get("stop_order_id")

        if REQUIRE_STOP_LOSS and not (stop > 0 and stop < entry):
            print(f"🚨 Missing/invalid stop on {ticker}; closing immediately.")
            to_close.append((ticker, "Missing mandatory stop loss"))
            continue

        if live_tickers is not None and t212_ticker not in live_tickers:
            print(f"ℹ️ {ticker} ({t212_ticker}) no longer in T212 portfolio; reconciling {SWING_STATE_FILE}.")
            df_rec = latest.get(ticker)
            px_rec = _to_float(df_rec["Close"].dropna()) if (df_rec is not None and not df_rec.empty and not df_rec["Close"].dropna().empty) else entry
            u_rec = cfg["scale"] * (fx_to_gbp if region == "US" else 1.0)
            e_gbp_rec = float(pos.get("entry_gbp") or entry * u_rec)
            c_gbp_rec = px_rec * u_rec
            t_gbp_rec = float(pos.get("target_gbp") or target * u_rec)
            if c_gbp_rec >= t_gbp_rec or c_gbp_rec > e_gbp_rec:
                rec_reason = f"Target profit reached / closed in profit (£{c_gbp_rec:.2f} vs entry £{e_gbp_rec:.2f}, target £{t_gbp_rec:.2f})"
            else:
                rec_reason = "Closed on broker (stop filled)"
            pnl_rec = (c_gbp_rec - e_gbp_rec) * qty
            to_close.append((ticker, rec_reason))
            dirty = True
            post_discord(
                "T212 Swing Risk Guard",
                (
                    f"🔒 **Swing Position Closed**: `{ticker}` (`{qty}` sh)\n"
                    f"• **Reason:** {rec_reason}\n"
                    f"• **Est P&L:** `£{pnl_rec:+.2f}`"
                ),
            )
            continue

        if REQUIRE_STOP_LOSS and live_order_ids is not None and stop_order_id and str(stop_order_id) not in live_order_ids:
            try:
                re_stop = t212.place_protective_stop_sell(t212_ticker, qty, broker_stop_price)
                pos["stop_order_id"] = re_stop.get("id")
                stop_order_id = pos["stop_order_id"]
                dirty = True
                print(f"🛡️ Re-attached missing broker stop on {ticker} @ {cfg['ccy']}{stop:.2f}")
            except Exception as re_err:
                print(f"🚨 Failed to re-attach stop on {ticker}: {re_err!r} -> closing position!")
                try:
                    t212.place_market_sell(t212_ticker, qty)
                except Exception:
                    pass
                to_close.append((ticker, "Unprotected position closed"))
                dirty = True
                continue

        df = latest.get(ticker)
        raw_px = _to_float(df["Close"].dropna()) if (df is not None and not df.empty and not df["Close"].dropna().empty) else entry
        # Guard against unit/scale mismatch where raw quote is >20x or <0.05x entry
        if entry > 0 and (raw_px > entry * 20.0 or raw_px < entry * 0.05):
            curr_price = entry
        else:
            curr_price = raw_px

        unit_to_gbp = cfg["scale"] * (fx_to_gbp if region == "US" else 1.0)
        entry_gbp = float(pos.get("entry_gbp") or entry * unit_to_gbp)
        curr_gbp = curr_price * unit_to_gbp
        stop_gbp = float(pos.get("stop_gbp") or stop * unit_to_gbp)
        target_gbp = float(pos.get("target_gbp") or target * unit_to_gbp)
        broker_stop_price = stop_gbp if region == "UK" else stop

        heads = fetch_news(ticker, region)
        exit_reason = None

        pnl_gbp_now = (curr_gbp - entry_gbp) * qty
        pnl_pct_now = ((curr_gbp - entry_gbp) / entry_gbp * 100.0) if entry_gbp else 0.0
        target_gbp_val = float(pos.get("target_gbp") or target * unit_to_gbp)

        # Notify when in profit and target price is reached (keeps stop-loss active unless shock news)
        if curr_gbp >= target_gbp_val and pnl_gbp_now > 0:
            last_tp_alert = (pos.get("profit_alerts") or {}).get("target_hit")
            can_tp_alert = True
            if last_tp_alert:
                can_tp_alert = ((now_utc - datetime.fromisoformat(last_tp_alert)).total_seconds() / 3600.0) >= 2.0
            if can_tp_alert:
                msg = (
                    f"🎯 **PROFIT TARGET REACHED — CONSIDER SELLING**: `{ticker}` (`{qty}` sh)\n"
                    f"• **Entry:** `£{entry_gbp:.2f}` -> **Current:** `£{curr_gbp:.2f}` (`{pnl_pct_now:+.2f}%` | **Est Profit:** `£{pnl_gbp_now:+.2f}`)\n"
                    f"• **Target Price:** `£{target_gbp_val:.2f}` | **Broker Stop Still Active:** `£{stop_gbp:.2f}`"
                )
                print(f"🎯 {ticker} hit profit target (£{curr_gbp:.2f} >= £{target_gbp_val:.2f}, P&L £{pnl_gbp_now:+.2f}) — sending sell notification.")
                post_discord("T212 Profit & Stop Guard", msg)
                pos.setdefault("profit_alerts", {})["target_hit"] = now_utc.isoformat()
                dirty = True

        exit_reason = None
        if any(SHOCK_RE.search(h) for h in heads):
            exit_reason = f"Shock news exit: {heads[0]}"
        elif curr_gbp <= stop_gbp and pnl_gbp_now < 0:
            exit_reason = f"Stop breached (£{curr_gbp:.2f} <= £{stop_gbp:.2f})"

        if exit_reason:
            last_close_alert = (pos.get("close_alert_ts") or "")
            can_close_alert = True
            if last_close_alert:
                can_close_alert = ((now_utc - datetime.fromisoformat(last_close_alert)).total_seconds() / 3600.0) >= 2.0

            sell_succeeded = False
            try:
                if stop_order_id:
                    t212.cancel_order(stop_order_id)
                    time.sleep(1.0)
                t212.place_market_sell(t212_ticker, qty)
                sell_succeeded = True
            except Exception as e:
                print(f"⚠️ Exit execution warning for {ticker}: {e!r}")

            pnl_gbp = (curr_gbp - entry_gbp) * qty
            print(f"🔒 Closed {ticker}: {exit_reason} | Est P&L: £{pnl_gbp:+.2f}")
            if sell_succeeded:
                to_close.append((ticker, exit_reason))
                dirty = True
            if can_close_alert:
                status_note = "Market sell submitted (pending fill if market closed)" if sell_succeeded else "Sell order pending/retry"
                post_discord(
                    "T212 Swing Risk Guard",
                    (
                        f"🔒 **Swing Position Closed**: `{ticker}` (`{qty}` sh)\n"
                        f"• **Reason:** {exit_reason} ({status_note})\n"
                        f"• **Est P&L:** `£{pnl_gbp:+.2f}`"
                    ),
                )
                pos["close_alert_ts"] = now_utc.isoformat()
                dirty = True
            continue

        at_risk, verdict = ai_position_check(ticker, qty, entry_gbp, curr_gbp, stop_gbp, target_gbp, heads)
        icon = "⚠️" if at_risk else "✅"
        print(f"{icon} [{ticker} | {qty} sh @ £{entry_gbp:.2f} -> £{curr_gbp:.2f}] {verdict}")

        if at_risk:
            last_str = risk_alerts.get(ticker)
            can_alert = True
            if last_str:
                can_alert = ((now_utc - datetime.fromisoformat(last_str)).total_seconds() / 3600.0) >= 4.0
            if can_alert:
                post_discord(
                    "T212 Swing Risk Guard",
                    (
                        f"⚠️ **AI Swing & News Warning**: `{ticker}` (`{qty}` sh)\n"
                        f"• **Entry:** `£{entry_gbp:.2f}` | **Current:** `£{curr_gbp:.2f}` | **Stop:** `£{stop_gbp:.2f}`\n"
                        f"• **Audit:** {verdict}"
                    ),
                )
                risk_alerts[ticker] = now_utc.isoformat()
                dirty = True

    for t, _ in to_close:
        open_trades.pop(t, None)

    if dirty:
        save_swing_state(state)


if __name__ == "__main__":
    try:
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
    except Exception:
        pass
    mode = os.getenv("RUN_MODE", "monitor")
    if mode == "execute_3pm":
        execute_swing_3pm_uk_us()
    else:
        monitor_swing_trades()