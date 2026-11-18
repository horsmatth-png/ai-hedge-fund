import os
import re
import sys
import json
from datetime import datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from trading_ig import IGService
from trading_ig.rest import IGException

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
IG_API_KEY = os.getenv("IG_API_KEY")
IG_ACC_TYPE = os.getenv("IG_ACC_TYPE", "DEMO")

STATE_FILE = "active_trades.json"
MAX_TRADES_PER_DAY = 3

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

NUM_PICKS = int(os.getenv("NUM_PICKS", "3"))  # Capped to match your max daily trades
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "60"))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", "50000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
MAX_PER_SECTOR = 1
MEGA_CAP_CAP = 2

UA = {"User-Agent": "Mozilla/5.0 (compatible; quant-scanner/1.0)"}
MEGA_CAPS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO"}

REGION_CFG = {
    "US": {"tz": "America/New_York", "ccy": "$", "scale": 1.0, "pts_per_unit": 100,
           "min_price": 5.0, "min_value": 40e6, "bot": "AI Quant Execution Engine"},
    "UK": {"tz": "Europe/London", "ccy": "£", "scale": 0.01, "pts_per_unit": 1,
           "min_price": 100.0, "min_value": 5e6, "bot": "UK Dynamic Quant Engine"},
}

FALLBACK_US = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "QCOM", "JPM", "BAC", "GS"]
FALLBACK_UK = ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L"]


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


# ----------------------------------------------------------------------------
# Scanner Helpers
# ----------------------------------------------------------------------------
def _wiki_constituents(url: str, suffix: str = ""):
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
    except Exception:
        pass
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
            return uni
    except Exception:
        pass
    fb = FALLBACK_US if region == "US" else FALLBACK_UK
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
        except Exception:
            pass
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


def stage1_shortlist(universe: dict, cfg: dict, today) -> tuple:
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
            rank = abs(move) / max(atr_pct, 1e-4) + atr_pct * 50
            rows.append({"ticker": t, "rank": rank, "has_today": has_today})
        except Exception:
            continue
    rows.sort(key=lambda r: r["rank"], reverse=True)
    return [r["ticker"] for r in rows[:SHORTLIST_SIZE]], daily


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

    cum_today = float(tdf["Volume"].sum())
    cum_hist = []
    for d in hist_days:
        day = d5[dates == d]
        cum_hist.append(float(day[[i.time() <= last_time for i in day.index]]["Volume"].sum()))
    avg_cum = np.mean([c for c in cum_hist if c > 0]) if any(c > 0 for c in cum_hist) else 0
    rel_vol = cum_today / avg_cum if avg_cum > 0 else 1.0

    or_bars = tdf.iloc[:6]
    or_high, or_low = float(or_bars["High"].max()), float(or_bars["Low"].min())
    or_done = len(tdf) >= 6

    hist_d = dd[[i.date() != today for i in dd.index]]
    if len(hist_d) < 50:
        return None
    prev_close = float(hist_d["Close"].iloc[-1])
    a = atr(hist_d)
    sma20 = float(hist_d["Close"].rolling(20).mean().iloc[-1])
    sma50 = float(hist_d["Close"].rolling(50).mean().iloc[-1])
    rsi = wilder_rsi(hist_d["Close"])
    gap_pct = (day_open - prev_close) / prev_close * 100
    move_atr = (price - prev_close) / a

    def side_score(side: str) -> tuple:
        s, why = 0.0, []
        sign = 1 if side == "LONG" else -1
        if rel_vol >= 0.8:
            s += min(rel_vol, 3.0) / 3.0 * 25
            why.append(f"RVOL {rel_vol:.1f}x")
        if (price - vwap) * sign > 0:
            s += 20
        if or_done:
            if side == "LONG" and price > or_high: s += 20
            elif side == "SHORT" and price < or_low: s += 20
        if side == "LONG" and price > sma20 > sma50: s += 15
        elif side == "SHORT" and price < sma20 < sma50: s += 15
        return s, why

    long_s, long_w = side_score("LONG")
    short_s, short_w = side_score("SHORT")
    side, score, why = ("LONG", long_s, long_w) if long_s >= short_s else ("SHORT", short_s, short_w)

    return {
        "ticker": t, "sector": sector, "side": side, "score": round(score, 1), "why": why,
        "price": price, "vwap": vwap, "rsi": round(rsi, 1), "rel_vol": round(rel_vol, 2),
        "atr": a, "gap_pct": round(gap_pct, 2), "or_high": or_high, "or_low": or_low,
    }


def fetch_news(ticker: str, region: str):
    headlines = []
    try:
        if region == "US" and FINNHUB_API_KEY:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            frm = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
            r = requests.get("https://finnhub.io/api/v1/company-news", params={"symbol": ticker, "from": frm, "to": today, "token": FINNHUB_API_KEY}, timeout=5)
            if r.status_code == 200:
                for a in r.json():
                    if a.get("headline"): headlines.append(a["headline"])
        else:
            for item in (yf.Ticker(ticker).news or [])[:4]:
                title = (item.get("content") or {}).get("title") or item.get("title")
                if title: headlines.append(title.strip())
    except Exception:
        pass
    return headlines[:4]


def pick_diversified(cands: list, n: int) -> list:
    picks, sectors = [], {}
    mega = 0
    for c in cands:
        if sectors.get(c["sector"], 0) >= MAX_PER_SECTOR and c["sector"] != "Unknown": continue
        if c["ticker"] in MEGA_CAPS and mega >= MEGA_CAP_CAP: continue
        picks.append(c)
        sectors[c["sector"]] = sectors.get(c["sector"], 0) + 1
        mega += c["ticker"] in MEGA_CAPS
        if len(picks) >= n: return picks
    return picks[:n]


def build_plan(c: dict, cfg: dict) -> dict:
    sign = 1 if c["side"] == "LONG" else -1
    risk = 0.4 * c["atr"]
    price = c["price"]
    sl = price - sign * risk
    tp = price + sign * 2 * risk
    s = cfg["scale"]
    risk_ccy = risk * s
    budget = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    units = int(budget / risk_ccy) if risk_ccy > 0 else 1
    return {"entry_lo": price * 0.999, "entry_hi": price * 1.001, "sl": sl, "tp": tp, "units": max(1, units)}


def map_ticker_to_epic(ticker: str, region: str) -> str:
    clean = ticker.replace(".L", "").upper()
    return f"CS.D.{clean}.CASH.IP" if region == "UK" else f"CS.D.{clean}.CFD.IP"


# ----------------------------------------------------------------------------
# Main Execution & Monitoring Logic
# ----------------------------------------------------------------------------
def execute_daily_trades():
    print("==================================================")
    print("  🔮 RUNNING STOCK SCAN & IG PAPER EXECUTION")
    print("==================================================")

    region = os.getenv("REGION", "US").upper()
    cfg = REGION_CFG.get(region, REGION_CFG["US"])
    tz = ZoneInfo(cfg["tz"])
    today = datetime.now(tz).date()

    universe = get_universe(region)
    shortlist, daily = stage1_shortlist(universe, cfg, today)
    if not shortlist:
        print("⚠️ No liquid candidates found.")
        return

    intraday = batch_download(shortlist, chunk_size=30, period="10d", interval="5m", prepost=False)
    scored = []
    for t in shortlist:
        if t in intraday and t in daily:
            r = score_intraday(t, intraday[t], daily[t], today, universe.get(t, "Unknown"))
            if r: scored.append(r)

    if not scored:
        print("⚠️ No intraday data available today.")
        return

    scored.sort(key=lambda r: (r["score"], r["rel_vol"]), reverse=True)
    picks = pick_diversified(scored[:25], NUM_PICKS)
    print(f"✅ Selected Stock Picks: {[p['ticker'] for p in picks]}")

    if not picks:
        return

    # Connect to IG
    if not IG_USERNAME or not IG_PASSWORD or not IG_API_KEY:
        print("❌ IG API credentials missing.")
        return

    try:
        ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
        ig_service.create_session()
    except Exception as e:
        print(f"❌ Failed to connect to IG API: {e}")
        return

    state = load_state()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if state.get("date") != today_str:
        state = {"date": today_str, "daily_count": 0, "active_orders": {}}

    if state["daily_count"] >= MAX_TRADES_PER_DAY:
        print("🛑 Daily limit of 3 orders already reached.")
        return

    for pick in picks:
        if state["daily_count"] >= MAX_TRADES_PER_DAY:
            break

        ticker = pick["ticker"]
        epic = map_ticker_to_epic(ticker, region)

        if epic in state["active_orders"]:
            print(f"🔁 Skipping {ticker}: Already processed today.")
            continue

        plan = build_plan(pick, cfg)
        direction = "BUY" if pick["side"] == "LONG" else "SELL"
        entry_price = plan["entry_lo"] if direction == "BUY" else plan["entry_hi"]

        try:
            response = ig_service.create_working_order(
                currency_code="USD" if region == "US" else "GBP",
                direction=direction,
                epic=epic,
                expiry="-",
                level=round(entry_price, 2),
                order_type="LIMIT",
                size=plan["units"],
                time_in_force="GOOD_TILL_CANCELLED",
                guaranteed_stop=False,
                stop_level=round(plan["sl"], 2),
                limit_level=round(plan["tp"], 2)
            )

            deal_ref = response.get("dealReference")
            state["active_orders"][epic] = {"dealReference": deal_ref, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
            state["daily_count"] += 1
            print(f"🚀 Placed IG Limit Order for {ticker} | Entry: {entry_price:.2f} | Ref: {deal_ref}")

            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json={
                    "username": "IG Stock Execution Bot",
                    "content": f"🎯 **IG Order Placed**: `{ticker}` ({pick['side']}) at `{entry_price:.2f}` | SL: `{plan['sl']:.2f}` | TP: `{plan['tp']:.2f}`"
                })

        except IGException as e:
            print(f"❌ IG Order Error for {ticker}: {e}")

    save_state(state)


def monitor_trades():
    try:
        ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
        ig_service.create_session()
        working = ig_service.fetch_working_orders().get("workingOrders", [])
        positions = ig_service.fetch_open_positions().get("positions", [])
        print(f"🔍 Monitor Check — Working Orders: {len(working)} | Open Positions: {len(positions)}")
    except IGException as e:
        print(f"⚠️ Error during monitor check: {e}")


if __name__ == "__main__":
    mode = os.getenv("RUN_MODE", "monitor")
    if mode == "execute_daily":
        execute_daily_trades()
    else:
        monitor_trades()
