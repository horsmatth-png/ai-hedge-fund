"""
Intraday Quant Scanner
----------------------
Stage 0: build a wide universe (S&P 500 + S&P 400 for US, FTSE 100 for UK)
Stage 1: one batched daily download -> liquidity filter + "in play" pre-rank
Stage 2: 5-minute data on the shortlist -> VWAP, opening range, time-of-day
         relative volume, trend alignment -> long/short score
Stage 3: news check (shortlist only), sector-diversified picks, ATR-based trade
         plan, LLM writes the commentary only, Discord delivery.
"""
import os
import re
import sys
from datetime import datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

NUM_PICKS = int(os.getenv("NUM_PICKS", "5"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "60"))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", "50000"))   # in account currency
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
MAX_PER_SECTOR = 1          # first pass; relaxed only if not enough sectors
MEGA_CAP_CAP = 2            # max mega-caps in the final list (US)

UA = {"User-Agent": "Mozilla/5.0 (compatible; quant-scanner/1.0)"}

MEGA_CAPS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO"}

REGION_CFG = {
    "US": {"tz": "America/New_York", "ccy": "$", "scale": 1.0, "pts_per_unit": 100,
           "min_price": 5.0, "min_value": 40e6, "bot": "AI Quant Execution Engine"},
    # yfinance quotes LSE stocks in pence (GBp)
    "UK": {"tz": "Europe/London", "ccy": "£", "scale": 0.01, "pts_per_unit": 1,
           "min_price": 100.0, "min_value": 5e6, "bot": "UK Dynamic Quant Engine"},
}

# Only used if Wikipedia can't be reached
FALLBACK_US = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "QCOM", "JPM", "BAC", "GS",
               "V", "MA", "BLK", "UNH", "LLY", "JNJ", "PFE", "ABBV", "CAT", "GE", "RTX", "LMT", "HON",
               "BA", "XOM", "CVX", "COP", "WMT", "COST", "PG", "KO", "PEP"]
FALLBACK_UK = ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L", "REL.L",
               "LLOY.L", "BARC.L", "VOD.L", "GLEN.L", "NG.L", "RR.L"]


# ----------------------------------------------------------------------------
# Stage 0: universe
# ----------------------------------------------------------------------------
def _wiki_constituents(url: str, suffix: str = ""):
    """Return {ticker: sector} from a Wikipedia constituents table."""
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
            print(f"Universe ({region}): {len(uni)} tickers")
            return uni
    except Exception as e:
        print(f"Universe fetch failed: {e!r}")
    fb = FALLBACK_US if region == "US" else FALLBACK_UK
    print(f"Using fallback universe ({len(fb)} tickers) - sector diversification will be weak")
    return {t: "Unknown" for t in fb}


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
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
            raw = yf.download(chunk, group_by="ticker", progress=False, threads=True,
                              auto_adjust=False, **kw)
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


# ----------------------------------------------------------------------------
# Stage 1: broad "in play" pre-filter on daily data
# ----------------------------------------------------------------------------
def stage1_shortlist(universe: dict, cfg: dict, today) -> list:
    daily = batch_download(list(universe), period="4mo", interval="1d")
    rows = []
    for t, df in daily.items():
        try:
            if len(df) < 40:
                continue
            has_today = df.index[-1].date() == today
            hist = df.iloc[:-1] if has_today else df
            prev_close = float(hist["Close"].iloc[-1])
            price = float(df["Close"].iloc[-1])
            if price < cfg["min_price"]:
                continue
            value = float((hist["Close"] * hist["Volume"]).tail(20).mean()) * cfg["scale"]
            if value < cfg["min_value"]:
                continue
            a = atr(hist)
            atr_pct = a / prev_close
            move = (price - prev_close) / prev_close if has_today else 0.0
            # "in play" = moving a meaningful fraction of its normal range, and volatile enough to trade
            rank = abs(move) / max(atr_pct, 1e-4) + atr_pct * 50
            rows.append({"ticker": t, "rank": rank, "has_today": has_today})
        except Exception:
            continue
    rows.sort(key=lambda r: r["rank"], reverse=True)
    print(f"Stage 1: {len(rows)} liquid names -> shortlisting {SHORTLIST_SIZE}")
    return [r["ticker"] for r in rows[:SHORTLIST_SIZE]], daily


# ----------------------------------------------------------------------------
# Stage 2: intraday scoring
# ----------------------------------------------------------------------------
def score_intraday(t: str, d5: pd.DataFrame, dd: pd.DataFrame, today, sector: str):
    d5 = d5.dropna(subset=["Close", "Volume"])
    if d5.empty:
        return None
    dates = np.array([i.date() for i in d5.index])
    tdf = d5[dates == today]
    if len(tdf) < 3:
        return None
    hist_days = sorted(set(dates) - {today})[-9:]
    if len(hist_days) < 3:
        return None

    price = float(tdf["Close"].iloc[-1])
    day_open = float(tdf["Open"].iloc[0])
    last_time = tdf.index[-1].time()

    typical = (tdf["High"] + tdf["Low"] + tdf["Close"]) / 3
    vwap = float((typical * tdf["Volume"]).cumsum().iloc[-1] / max(tdf["Volume"].cumsum().iloc[-1], 1))

    # Time-of-day relative volume: today's volume so far vs same time on prior days
    cum_today = float(tdf["Volume"].sum())
    cum_hist = []
    for d in hist_days:
        day = d5[dates == d]
        cum_hist.append(float(day[[i.time() <= last_time for i in day.index]]["Volume"].sum()))
    avg_cum = np.mean([c for c in cum_hist if c > 0]) if any(c > 0 for c in cum_hist) else 0
    rel_vol = cum_today / avg_cum if avg_cum > 0 else 1.0

    # Opening range (first 30 min = 6 x 5m bars)
    or_bars = tdf.iloc[:6]
    or_high, or_low = float(or_bars["High"].max()), float(or_bars["Low"].min())
    or_done = len(tdf) >= 6

    # Daily context
    hist_d = dd[[i.date() != today for i in dd.index]]
    if len(hist_d) < 50:
        return None
    prev_close = float(hist_d["Close"].iloc[-1])
    a = atr(hist_d)
    sma20 = float(hist_d["Close"].rolling(20).mean().iloc[-1])
    sma50 = float(hist_d["Close"].rolling(50).mean().iloc[-1])
    rsi = wilder_rsi(hist_d["Close"])
    gap_pct = (day_open - prev_close) / prev_close * 100
    move_atr = (price - prev_close) / a  # how much of a normal day's range is already used

    def side_score(side: str) -> tuple:
        s, why = 0.0, []
        sign = 1 if side == "LONG" else -1
        # 1. participation
        if rel_vol >= 0.8:
            pts = min(rel_vol, 3.0) / 3.0 * 25
            s += pts
            why.append(f"RVOL {rel_vol:.1f}x")
        # 2. VWAP side
        if (price - vwap) * sign > 0:
            s += 20
            why.append("above VWAP" if side == "LONG" else "below VWAP")
        # 3. opening-range break
        if or_done:
            if side == "LONG" and price > or_high:
                s += 20
                why.append("ORB break up")
            elif side == "SHORT" and price < or_low:
                s += 20
                why.append("ORB break down")
        # 4. daily trend alignment
        if side == "LONG" and price > sma20 > sma50:
            s += 15
            why.append("daily uptrend")
        elif side == "SHORT" and price < sma20 < sma50:
            s += 15
            why.append("daily downtrend")
        # 5. gap in direction
        if abs(gap_pct) >= 0.5 and gap_pct * sign > 0:
            s += 10
            why.append(f"gap {gap_pct:+.1f}%")
        # 6. room left to run (not already exhausted)
        used = move_atr * sign
        if 0.2 <= used <= 1.2:
            s += 10
        elif used > 1.8:
            s -= 15
            why.append("extended")
        # 7. exhaustion penalty
        if (side == "LONG" and rsi > 78) or (side == "SHORT" and rsi < 22):
            s -= 10
            why.append(f"RSI {rsi:.0f} stretched")
        return s, why

    long_s, long_w = side_score("LONG")
    short_s, short_w = side_score("SHORT")
    side, score, why = ("LONG", long_s, long_w) if long_s >= short_s else ("SHORT", short_s, short_w)

    return {
        "ticker": t, "sector": sector, "side": side, "score": round(score, 1), "why": why,
        "price": price, "vwap": vwap, "rsi": round(rsi, 1), "rel_vol": round(rel_vol, 2),
        "atr": a, "gap_pct": round(gap_pct, 2), "or_high": or_high, "or_low": or_low,
        "support": float(tdf["Low"].min()), "resistance": float(tdf["High"].max()),
    }


# ----------------------------------------------------------------------------
# Stage 3: news, diversification, plan, AI commentary
# ----------------------------------------------------------------------------
SHOCK_RE = re.compile(r"\b(resigns?|resignation|investigation|fraud|subpoena|lawsuit|sec probe|"
                      r"step(s|ping)? down|recall|bankruptcy|halted|restat\w+|short report)\b", re.I)


def fetch_news(ticker: str, region: str):
    headlines, insiders = [], "n/a"
    try:
        if region == "US" and FINNHUB_API_KEY:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            frm = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
            r = requests.get("https://finnhub.io/api/v1/company-news",
                             params={"symbol": ticker, "from": frm, "to": today, "token": FINNHUB_API_KEY}, timeout=8)
            if r.status_code == 200:
                # Keep only stories actually about this company: the ticker is tagged as one of
                # at most 3 related symbols, or it appears in the headline itself.
                tk = ticker.upper()
                for a in r.json():
                    title = (a.get("headline") or "").strip()
                    related = [x.strip().upper() for x in (a.get("related") or "").split(",") if x.strip()]
                    about_it = (tk in related and len(related) <= 3) or re.search(rf"\b{re.escape(tk)}\b", title, re.I)
                    if title and about_it and title not in headlines:
                        headlines.append(title)
                    if len(headlines) >= 4:
                        break
            r = requests.get("https://finnhub.io/api/v1/stock/insider-transactions",
                             params={"symbol": ticker, "token": FINNHUB_API_KEY}, timeout=8)
            if r.status_code == 200:
                txs = r.json().get("data", [])[:3]
                seen, items = set(), []
                for x in txs:
                    line = f"{x.get('name')} ({(x.get('change') or 0):+,} shs)"
                    if x.get("name") and line not in seen:
                        seen.add(line)
                        items.append(line)
                insiders = "; ".join(items) or "none recent"
        else:
            # Finnhub free tier doesn't cover LSE (and matching bare symbols pulls US companies), so use Yahoo
            for item in (yf.Ticker(ticker).news or [])[:6]:
                title = (item.get("content") or {}).get("title") or item.get("title")
                if title:
                    headlines.append(title.strip())
    except Exception as e:
        print(f"News error {ticker}: {e!r}")
    return headlines[:4], insiders


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
    for c in cands:  # backfill if the market is narrow
        if c not in picks and not (c["ticker"] in MEGA_CAPS and mega >= MEGA_CAP_CAP):
            picks.append(c)
            mega += c["ticker"] in MEGA_CAPS
        if len(picks) >= n:
            break
    return picks


def build_plan(c: dict, cfg: dict) -> dict:
    sign = 1 if c["side"] == "LONG" else -1
    risk = 0.4 * c["atr"]
    price = c["price"]
    sl = price - sign * risk
    tp = price + sign * 2 * risk
    s = cfg["scale"]
    risk_ccy_per_unit = risk * s
    budget = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    units = int(budget / risk_ccy_per_unit) if risk_ccy_per_unit > 0 else 0
    return {"entry_lo": price * 0.999, "entry_hi": price * 1.001, "sl": sl, "tp": tp,
            "sl_pts": int(risk * cfg["pts_per_unit"]), "tp_pts": int(2 * risk * cfg["pts_per_unit"]),
            "units": units, "budget": budget}


def ai_commentary(c: dict, headlines: list, insiders: str, region: str) -> tuple:
    prompt = (
        f"You are a concise intraday trading analyst. {c['ticker']} ({c['sector']}) setup: {c['side']}. "
        f"Price {c['price']:.2f}, VWAP {c['vwap']:.2f}, daily RSI {c['rsi']}, relative volume {c['rel_vol']}x, "
        f"gap {c['gap_pct']}%. Signals: {', '.join(c['why']) or 'none'}.\n"
        f"Recent headlines: {' | '.join(headlines) or 'none found'}\nInsider activity: {insiders}\n\n"
        "Write 3 short sentences: why this is (or is not) a clean intraday setup, the main risk, and what would "
        "invalidate it. Use only the facts above, never invent catalysts. No HTML, no tables, no financial-advice disclaimers."
    )
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            text = (genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text or "").strip()
            if text:
                return text, f"Gemini ({GEMINI_MODEL})"
        except Exception as e:
            print(f"Gemini failed: {e!r}")
    if GROQ_API_KEY:
        try:
            from groq import Groq
            out = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3,
                max_completion_tokens=2000, reasoning_effort="low")
            text = (out.choices[0].message.content or "").strip()
            if text:
                return text, f"Groq ({GROQ_MODEL})"
            print("Groq returned empty text")
        except Exception as e:
            print(f"Groq failed: {e!r}")
    return "Rule-based setup: " + ", ".join(c["why"]) + ".", "Quant rules only"


def build_embed(c: dict, plan: dict, headlines: list, insiders: str, region: str) -> dict:
    cfg = REGION_CFG[region]
    ccy, s = cfg["ccy"], cfg["scale"]
    f = lambda v: f"{ccy}{v * s:,.2f}"
    text, engine = ai_commentary(c, headlines, insiders, region)
    text = re.sub(r"<br\s*/?>", " / ", text, flags=re.I)
    mega_tag = " (mega-cap)" if c["ticker"] in MEGA_CAPS else ""

    desc = (
        f"**{c['side']}** | Score **{c['score']}/100** | RVOL **{c['rel_vol']}x** | Gap **{c['gap_pct']:+}%** | RSI **{c['rsi']}**\n"
        f"Signals: {', '.join(c['why']) or 'n/a'}\n\n"
        f"**Levels**\n"
        f"Price {f(c['price'])} | VWAP {f(c['vwap'])} | Opening range {f(c['or_low'])} - {f(c['or_high'])}\n"
        f"Day low {f(c['support'])} | Day high {f(c['resistance'])}\n\n"
        f"**Trade plan** (stop 0.4 ATR, target 2R)\n"
        f"Entry {f(plan['entry_lo'])} - {f(plan['entry_hi'])}\n"
        f"Stop {f(plan['sl'])} ({plan['sl_pts']} pts) | Target {f(plan['tp'])} ({plan['tp_pts']} pts)\n"
        f"Size ~{plan['units']:,} units for {RISK_PER_TRADE_PCT}% account risk ({ccy}{plan['budget']:,.0f})\n\n"
        f"**News:** {' | '.join(headlines) if headlines else 'No recent headlines found.'}\n"
        f"**Insiders:** {insiders}\n\n"
        f"**Read:** {text}"
    )
    return {
        "title": f"{'🟢' if c['side'] == 'LONG' else '🔴'} {c['ticker']}{mega_tag} - {c['sector']}",
        "description": desc[:4000],
        "color": 3066993 if c["side"] == "LONG" else 15158332,
        "footer": {"text": f"{datetime.utcnow():%Y-%m-%d %H:%M UTC} | {engine} | Not financial advice"},
    }


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run_scan(region: str):
    cfg = REGION_CFG[region]
    tz = ZoneInfo(cfg["tz"])
    today = datetime.now(tz).date()
    print("=" * 50, f"\n  {region} INTRADAY SCAN  ({today})\n" + "=" * 50)

    universe = get_universe(region)
    shortlist, daily = stage1_shortlist(universe, cfg, today)
    if not shortlist:
        print("No liquid candidates found.")
        return

    intraday = batch_download(shortlist, chunk_size=30, period="10d", interval="5m", prepost=False)
    scored = []
    for t in shortlist:
        if t in intraday and t in daily:
            try:
                r = score_intraday(t, intraday[t], daily[t], today, universe.get(t, "Unknown"))
                if r:
                    scored.append(r)
            except Exception as e:
                print(f"Score error {t}: {e!r}")
    if not scored:
        print("No intraday data for today (market closed / holiday?).")
        return

    # Tie-break on relative volume so ties never fall back to list order
    scored.sort(key=lambda r: (r["score"], r["rel_vol"]), reverse=True)
    print("Top 15 raw scores:")
    for r in scored[:15]:
        print(f"  {r['ticker']:8} {r['side']:5} {r['score']:5} RVOL {r['rel_vol']:<5} {r['sector']}")

    # News / shock filter only on the top slice (keeps API calls tiny)
    clean = []
    for r in scored[:25]:
        heads, ins = fetch_news(r["ticker"], region)
        if any(SHOCK_RE.search(h) for h in heads):
            print(f"Dropped {r['ticker']} (shock headline)")
            continue
        r["_news"], r["_ins"] = heads, ins
        clean.append(r)

    picks = pick_diversified(clean, NUM_PICKS)
    print(f"Sending {len(picks)} picks: {[p['ticker'] for p in picks]}")
    if not (picks and DISCORD_WEBHOOK_URL):
        return

    embeds = [build_embed(p, build_plan(p, cfg), p["_news"], p["_ins"], region) for p in picks]
    header = (f"**{region} intraday scan** - {len(universe)} names scanned, {len(scored)} scored on live 5m data. "
              f"Top {len(picks)} sector-diversified setups:")
    resp = requests.post(DISCORD_WEBHOOK_URL, json={"username": cfg["bot"], "content": header, "embeds": embeds[:10]}, timeout=15)
    print("Discord:", resp.status_code)


if __name__ == "__main__":
    forced = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("REGION", "")).upper()
    if forced in REGION_CFG:
        region = forced
    else:
        region = "UK" if datetime.now(ZoneInfo("Europe/London")).hour < 12 else "US"
    run_scan(region)
