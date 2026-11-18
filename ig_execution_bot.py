import os
import re
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

# Environment Variables & Secrets
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
IG_API_KEY = os.getenv("IG_API_KEY")
IG_ACC_TYPE = os.getenv("IG_ACC_TYPE", "DEMO")
IG_ACC_NUMBER = os.getenv("IG_ACC_NUMBER")

# State & Limits
STATE_FILE = "active_trades.json"
MAX_TRADES_PER_REGION = 3

# AI & Scanner Config (Updated with active models)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "60"))
ACCOUNT_SIZE_GBP = float(os.getenv("ACCOUNT_SIZE", "50000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
MAX_PER_SECTOR = 1
MEGA_CAP_CAP = 2

UA = {"User-Agent": "Mozilla/5.0 (compatible; quant-scanner/1.0)"}
MEGA_CAPS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO"}

REGION_CFG = {
    "US": {
        "tz": "America/New_York",
        "ccy": "$",
        "scale": 1.0,
        "pts_per_unit": 100.0,
        "min_price": 5.0,
        "min_value": 40e6,
        "bot": "IG Spread Bet US Engine",
    },
    "UK": {
        "tz": "Europe/London",
        "ccy": "£",
        "scale": 0.01,
        "pts_per_unit": 1.0,
        "min_price": 100.0,
        "min_value": 5e6,
        "bot": "IG Spread Bet UK Engine",
    },
}

FALLBACK_US = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "QCOM", "JPM", "BAC", "GS",
               "V", "MA", "BLK", "UNH", "LLY", "JNJ", "PFE", "ABBV", "CAT", "GE", "RTX", "LMT"]
FALLBACK_UK = ["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "RIO.L", "DGE.L", "REL.L",
               "LLOY.L", "BARC.L", "VOD.L", "GLEN.L", "NG.L", "RR.L"]

SHOCK_RE = re.compile(
    r"\b(resigns?|resignation|investigation|fraud|subpoena|lawsuit|sec probe|"
    r"step(s|ping)? down|recall|bankruptcy|halted|restat\w+|short report)\b",
    re.I,
)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


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
    print(f"Stage 1: {len(rows)} liquid names -> shortlisting {SHORTLIST_SIZE}")
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
            why.append("above VWAP" if side == "LONG" else "below VWAP")
        if or_done:
            if side == "LONG" and price > or_high:
                s += 20
                why.append("ORB break up")
            elif side == "SHORT" and price < or_low:
                s += 20
                why.append("ORB break down")
        if side == "LONG" and price > sma20 > sma50:
            s += 15
            why.append("daily uptrend")
        elif side == "SHORT" and price < sma20 < sma50:
            s += 15
            why.append("daily downtrend")
        if abs(gap_pct) >= 0.5 and gap_pct * sign > 0:
            s += 10
            why.append(f"gap {gap_pct:+.1f}%")
        used = move_atr * sign
        if 0.2 <= used <= 1.2:
            s += 10
        elif used > 1.8:
            s -= 15
            why.append("extended")
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
    }


def fetch_news(ticker: str, region: str) -> list:
    headlines = []
    try:
        if region == "US" and FINNHUB_API_KEY:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            frm = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
            r = requests.get("https://finnhub.io/api/v1/company-news",
                             params={"symbol": ticker, "from": frm, "to": today, "token": FINNHUB_API_KEY}, timeout=8)
            if r.status_code == 200:
                tk = ticker.upper()
                for a in r.json():
                    title = (a.get("headline") or "").strip()
                    related = [x.strip().upper() for x in (a.get("related") or "").split(",") if x.strip()]
                    about_it = (tk in related and len(related) <= 3) or re.search(rf"\b{re.escape(tk)}\b", title, re.I)
                    if title and about_it and title not in headlines:
                        headlines.append(title)
                    if len(headlines) >= 4:
                        break
        else:
            for item in (yf.Ticker(ticker).news or [])[:5]:
                title = (item.get("content") or {}).get("title") or item.get("title")
                if title:
                    headlines.append(title.strip())
    except Exception as e:
        print(f"News error {ticker}: {e!r}")
    return headlines[:4]


def ai_commentary(c: dict, headlines: list) -> str:
    prompt = (
        f"You are a concise intraday trading analyst. {c['ticker']} ({c['sector']}) setup: {c['side']}. "
        f"Price {c['price']:.2f}, VWAP {c['vwap']:.2f}, daily RSI {c['rsi']}, relative volume {c['rel_vol']}x, "
        f"gap {c['gap_pct']}%. Signals: {', '.join(c['why']) or 'none'}.\n"
        f"Recent headlines: {' | '.join(headlines) or 'none found'}\n\n"
        "Write 2 short sentences explaining the technical setup and key risk. No HTML, no disclaimers."
    )
    if GROQ_API_KEY:
        try:
            from groq import Groq
            out = Groq(api_key=GROQ_API_KEY).chat.completions.create(
                model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=200)
            text = (out.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"Groq failed: {e!r}")
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            text = (genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"Gemini failed: {e!r}")
    return "Rule-based setup: " + ", ".join(c["why"]) + "."


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


def build_spread_bet_plan(c: dict, cfg: dict, region: str) -> dict:
    sign = 1 if c["side"] == "LONG" else -1
    pts_mult = cfg["pts_per_unit"]

    price_pts = c["price"] * pts_mult
    risk_pts = max(1.0, 0.4 * c["atr"] * pts_mult)

    entry_lo_pts = price_pts * 0.999
    entry_hi_pts = price_pts * 1.001
    sl_pts_level = price_pts - sign * risk_pts
    tp_pts_level = price_pts + sign * 2.0 * risk_pts

    stake = 0.24 if region == "US" else 5.0

    return {
        "entry_lo": round(entry_lo_pts, 1),
        "entry_hi": round(entry_hi_pts, 1),
        "sl_level": round(sl_pts_level, 1),
        "tp_level": round(tp_pts_level, 1),
        "risk_pts": round(risk_pts, 1),
        "stake": stake,
    }


def resolve_spread_bet_epic(ig_service: IGService, ticker: str, region: str) -> dict | None:
    clean = ticker.replace(".L", "").upper()
    try:
        res = ig_service.search_markets(clean)
        markets = res.get("markets", []) if isinstance(res, dict) else res.to_dict("records")

        for m in markets:
            epic = str(m.get("epic", ""))
            expiry = str(m.get("expiry", ""))
            instr_type = str(m.get("instrumentType", ""))
            status = str(m.get("marketStatus", ""))

            if expiry == "DFB" and "SHARES" in instr_type and status == "TRADEABLE":
                if region == "UK" and ".L" in ticker and ("UK" in epic or "KA." in epic or "LSE" in str(m.get("instrumentName", ""))):
                    return {"epic": epic, "name": m.get("instrumentName")}
                if region == "US" and ".L" not in ticker and ("UA." in epic or "UC." in epic or "US" in str(m.get("instrumentName", ""))):
                    return {"epic": epic, "name": m.get("instrumentName")}

        for m in markets:
            epic = str(m.get("epic", ""))
            if str(m.get("expiry", "")) == "DFB" and "SHARES" in str(m.get("instrumentType", "")):
                return {"epic": epic, "name": m.get("instrumentName")}

    except Exception as e:
        print(f"⚠️ Market search failed for {ticker}: {e!r}")

    return {
        "epic": f"KA.D.{clean}.DAILY.IP" if region == "UK" else f"UA.D.{clean}.DAILY.IP",
        "name": clean,
    }


def connect_ig() -> IGService:
    ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
    ig_service.create_session()

    if IG_ACC_NUMBER:
        try:
            ig_service.switch_account(IG_ACC_NUMBER, default_account=True)
            print(f"✅ Successfully switched active IG session to Spread Betting account: {IG_ACC_NUMBER}")
        except Exception as e:
            if "accountId-must-be-different" in str(e):
                print(f"ℹ️ Session is already bound to target account: {IG_ACC_NUMBER}")
            else:
                try:
                    ig_service.switch_account(IG_ACC_NUMBER, default_account=True)
                    print(f"✅ Switched account via fallback: {IG_ACC_NUMBER}")
                except Exception as inner_e:
                    if "accountId-must-be-different" in str(inner_e):
                        print(f"ℹ️ Session is already bound to target account: {IG_ACC_NUMBER}")
                    else:
                        print(f"⚠️ Account switch notice: {inner_e}")
    return ig_service


def execute_daily_trades():
    print("==================================================")
    print("  🔮 RUNNING DUAL-SESSION SPREAD BETTING EXECUTION")
    print("==================================================")

    london_hour = datetime.now(ZoneInfo("Europe/London")).hour
    forced_region = os.getenv("REGION", "").upper()

    if forced_region in REGION_CFG:
        region = forced_region
    else:
        region = "UK" if london_hour < 12 else "US"

    print(f"🌍 Active Session Region: {region} (London Hour: {london_hour}:00)")

    cfg = REGION_CFG[region]
    tz = ZoneInfo(cfg["tz"])
    today = datetime.now(tz).date()

    universe = get_universe(region)
    shortlist, daily = stage1_shortlist(universe, cfg, today)
    if not shortlist:
        print(f"⚠️ No liquid {region} candidates found.")
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
        print(f"⚠️ No intraday data available for {region} today.")
        return

    scored.sort(key=lambda r: (r["score"], r["rel_vol"]), reverse=True)

    clean = []
    for r in scored[:25]:
        heads = fetch_news(r["ticker"], region)
        if any(SHOCK_RE.search(h) for h in heads):
            print(f"Dropped {r['ticker']} (shock headline)")
            continue
        r["_news"] = heads
        clean.append(r)

    candidates = pick_diversified(clean, 8)
    print(f"📋 Ranked {region} Candidates: {[p['ticker'] for p in candidates]}")

    if not candidates:
        return

    if not IG_USERNAME or not IG_PASSWORD or not IG_API_KEY:
        print("❌ IG API credentials missing.")
        return

    try:
        ig_service = connect_ig()
    except Exception as e:
        print(f"❌ Failed to connect to IG API: {e}")
        return

    state = load_state()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    if state.get("date") != today_str:
        state = {"date": today_str, "counts": {"UK": 0, "US": 0}, "active_orders": {}}

    region_count = state.setdefault("counts", {}).get(region, 0)
    if region_count >= MAX_TRADES_PER_REGION:
        print(f"🛑 Daily limit of {MAX_TRADES_PER_REGION} {region} spread bets already reached.")
        return

    for pick in candidates:
        if state["counts"][region] >= MAX_TRADES_PER_REGION:
            break

        ticker = pick["ticker"]
        market_info = resolve_spread_bet_epic(ig_service, ticker, region)
        if not market_info:
            print(f"⚠️ No tradeable DFB spread bet market found on IG for {ticker}, skipping...")
            continue

        epic = market_info["epic"]
        if epic in state["active_orders"]:
            print(f"🔁 Skipping {ticker} ({epic}): Already processed today.")
            continue

        try:
            details = ig_service.fetch_market_by_epic(epic)
            rules = details.get("dealingRules", {})
            min_stop_dist = float(rules.get("minNormalStopOrLimitDistance", {}).get("value", 5.0))
        except Exception as e:
            print(f"⚠️ Could not fetch dealing rules for {epic} ({e}), skipping...")
            continue

        plan = build_spread_bet_plan(pick, cfg, region)
        direction = "BUY" if pick["side"] == "LONG" else "SELL"
        sign = 1 if direction == "BUY" else -1

        entry_pts = plan["entry_lo"] if direction == "BUY" else plan["entry_hi"]
        stake = plan["stake"]

        risk_pts = max(plan["risk_pts"], min_stop_dist * 1.1)
        sl_pts = round(entry_pts - sign * risk_pts, 1)
        tp_pts = round(entry_pts + sign * 2.0 * risk_pts, 1)

        commentary = ai_commentary(pick, pick.get("_news", []))

        try:
            print(f"Attempting IG Spread Bet for {ticker} ({epic}) | {direction} £{stake}/pt @ {entry_pts} pts...")
            response = ig_service.create_working_order(
                currency_code="GBP",
                direction=direction,
                epic=epic,
                expiry="DFB",
                level=round(entry_pts, 1),
                order_type="LIMIT",
                size=round(stake, 2),
                time_in_force="GOOD_TILL_CANCELLED",
                guaranteed_stop=False,
                stop_level=sl_pts,
                limit_level=tp_pts,
            )

            deal_ref = response.get("dealReference")
            state["active_orders"][epic] = {
                "dealReference": deal_ref,
                "ticker": ticker,
                "region": region,
                "side": pick["side"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            state["counts"][region] += 1
            print(f"🚀 Placed IG {region} Spread Bet for {ticker} ({epic}) | Ref: {deal_ref}")

            if DISCORD_WEBHOOK_URL:
                requests.post(
                    DISCORD_WEBHOOK_URL,
                    json={
                        "username": cfg["bot"],
                        "content": (
                            f"🎯 **IG {region} Spread Bet Placed**: `{ticker}` ({pick['side']}) | Score: `{pick['score']}/100`\n"
                            f"• **Stake:** `£{stake:.2f}/pt` | **Entry:** `{entry_pts:.1f} pts` | **SL:** `{sl_pts:.1f}` | **TP:** `{tp_pts:.1f}`\n"
                            f"• **AI Read:** {commentary}"
                        ),
                    },
                    timeout=10,
                )

        except IGException as e:
            print(f"⚠️ IG API rejected spread bet for {ticker} ({epic}): {e} (Trying next candidate)")
        except Exception as e:
            print(f"⚠️ Unexpected error placing spread bet for {ticker}: {e!r}")

    save_state(state)


def monitor_trades():
    print("==================================================")
    print("  🔍 RUNNING 5-MIN IG SPREAD BET SURVEILLANCE")
    print("==================================================")

    if not IG_USERNAME or not IG_PASSWORD or not IG_API_KEY:
        print("❌ IG API credentials missing.")
        return

    try:
        ig_service = connect_ig()
        
        # Explicit position / working orders fetch
        positions = []
        try:
            pos_res = ig_service.fetch_open_positions()
            positions = pos_res.get("positions", []) if isinstance(pos_res, dict) else (pos_res or [])
        except Exception as e:
            print(f"⚠️ Positions fetch warning: {e}")

        working = []
        try:
            work_res = ig_service.fetch_working_orders()
            working = work_res.get("workingOrders", []) if isinstance(work_res, dict) else (work_res or [])
        except Exception as e:
            print(f"⚠️ Working orders fetch warning: {e}")

        print(f"📊 Status — Working Orders: {len(working)} | Open Positions: {len(positions)}")

        state = load_state()
        active_orders = state.get("active_orders", {})
        now_utc = datetime.utcnow()

        for order in working:
            w_data = order.get("workingOrderData", {}) if isinstance(order, dict) else getattr(order, "working_order_data", {})
            deal_id = w_data.get("dealId") if isinstance(w_data, dict) else getattr(w_data, "deal_id", None)
            epic = w_data.get("epic") if isinstance(w_data, dict) else getattr(w_data, "epic", None)

            tracked = active_orders.get(epic, {})
            ts_str = tracked.get("timestamp")
            if ts_str:
                created_dt = datetime.fromisoformat(ts_str)
                age_hours = (now_utc - created_dt).total_seconds() / 3600.0
                if age_hours > 4.0 and deal_id:
                    ig_service.delete_working_order(deal_id)
                    print(f"🗑️ Cancelled stale working order {deal_id} ({epic}) after {age_hours:.1f}h")

        for pos in positions:
            p_data = pos.get("position", {}) if isinstance(pos, dict) else getattr(pos, "position", {})
            m_data = pos.get("market", {}) if isinstance(pos, dict) else getattr(pos, "market", {})
            deal_id = p_data.get("dealId") if isinstance(p_data, dict) else getattr(p_data, "deal_id", None)
            epic = m_data.get("epic") if isinstance(m_data, dict) else getattr(m_data, "epic", None)
            direction = p_data.get("direction") if isinstance(p_data, dict) else getattr(p_data, "direction", None)
            size = p_data.get("size") if isinstance(p_data, dict) else getattr(p_data, "size", None)

            tracked = active_orders.get(epic)
            if not tracked:
                continue

            ticker = tracked.get("ticker")
            region = tracked.get("region", "US")
            heads = fetch_news(ticker, region)

            if any(SHOCK_RE.search(h) for h in heads):
                close_dir = "SELL" if direction == "BUY" else "BUY"
                ig_service.close_open_position(
                    deal_id=deal_id,
                    direction=close_dir,
                    epic=None,
                    expiry=None,
                    order_type="MARKET",
                    size=size,
                )
                print(f"🚨 Shock news detected for {ticker}! Closed spread bet {deal_id} immediately.")
                if DISCORD_WEBHOOK_URL:
                    requests.post(
                        DISCORD_WEBHOOK_URL,
                        json={
                            "username": "IG Risk Guard",
                            "content": f"🚨 **EMERGENCY EXIT**: Closed `{ticker}` ({direction}) due to breaking shock headline: *{heads[0]}*",
                        },
                        timeout=10,
                    )

    except IGException as e:
        print(f"⚠️ IG API Monitor Error: {e}")
    except Exception as e:
        print(f"⚠️ Monitor Error: {e!r}")


if __name__ == "__main__":
    mode = os.getenv("RUN_MODE", "monitor")
    if mode == "execute_daily":
        execute_daily_trades()
    else:
        monitor_trades()
