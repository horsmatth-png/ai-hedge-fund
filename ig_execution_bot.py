import os
import json
from datetime import datetime
import requests
import yfinance as yf
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from trading_ig import IGService
from trading_ig.rest import IGException

load_dotenv()

# Credentials & Config
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

IG_USERNAME = os.getenv("IG_USERNAME")
IG_PASSWORD = os.getenv("IG_PASSWORD")
IG_API_KEY = os.getenv("IG_API_KEY")
IG_ACC_TYPE = os.getenv("IG_ACC_TYPE", "DEMO")

STATE_FILE = "active_trades.json"
MAX_TRADES_PER_DAY = 3
QUANT_UNIVERSE = ["SPY", "QQQ"]


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


def fetch_finnhub_intelligence():
    headlines = []
    if not FINNHUB_API_KEY:
        return "Finnhub API Key missing."
    try:
        news_url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        news_res = requests.get(news_url, timeout=5)
        if news_res.status_code == 200:
            articles = news_res.json()
            for art in articles[:4]:
                if art.get("headline"):
                    headlines.append(art["headline"])
    except Exception as e:
        print(f"⚠️ Finnhub macro fetch error: {repr(e)}")
    return " | ".join(headlines) if headlines else "No major macro headlines in last 72h."


def get_groq_ai_analysis(ticker, current_price, rsi, atr, macro_news):
    """Replicates the quant AI pipeline evaluation to define exact trade parameters."""
    if not GROQ_API_KEY:
        print("⚠️ Groq API key missing. Using technical fallback levels.")
        return None

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    You are an elite quantitative trading AI managing index contracts (SPY/QQQ).
    Ticker: {ticker}
    Current Price: ${current_price}
    RSI (14): {rsi}
    ATR (Volatility Buffer): {atr}
    Macro News: {macro_news}

    Provide a strict JSON response containing:
    - "bias": "BULLISH" or "BEARISH"
    - "entry_limit": float (exact limit entry price)
    - "stop_loss": float (risk stop loss level)
    - "take_profit": float (target profit level)
    - "rationale": short 1-sentence explanation.
    """

    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Groq API evaluation error: {e}")
        return None


def analyze_index_asset(ticker: str):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="3mo", interval="1d")
        if df.empty or len(df) < 30:
            return None

        closes = df["Close"]
        current_price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        daily_change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)

        # Technicals
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)

        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - closes.shift())
        low_close = np.abs(df["Low"] - closes.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_14 = float(true_range.rolling(14).mean().iloc[-1])

        macro_news = fetch_finnhub_intelligence()
        ai_plan = get_groq_ai_analysis(ticker, current_price, rsi, atr_14, macro_news)

        if ai_plan:
            bias = ai_plan.get("bias", "BULLISH")
            entry_price = ai_plan.get("entry_limit", round(current_price, 2))
            sl_price = ai_plan.get("stop_loss", round(current_price * 0.98, 2))
            tp_price = ai_plan.get("take_profit", round(current_price * 1.02, 2))
            rationale = ai_plan.get("rationale", "AI quantitative model projection.")
        else:
            bias = "BULLISH" if current_price >= float(closes.rolling(10).mean().iloc[-1]) else "BEARISH"
            entry_price = round(current_price, 2)
            sl_price = round(current_price * 0.985 if bias == "BULLISH" else current_price * 1.015, 2)
            tp_price = round(current_price * 1.025 if bias == "BULLISH" else current_price * 0.975, 2)
            rationale = "Fallback algorithmic technical momentum."

        return {
            "ticker": ticker,
            "price": current_price,
            "daily_change": daily_change_pct,
            "rsi": rsi,
            "bias": bias,
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "rationale": rationale,
            "news": macro_news
        }
    except Exception as e:
        print(f"⚠️ Error analyzing {ticker}: {repr(e)}")
        return None


def execute_daily_trades():
    print("==================================================")
    print("  🔮 RUNNING 3 PM UK QUANT SCAN & IG PAPER EXECUTION")
    print("==================================================")

    if not IG_USERNAME or not IG_PASSWORD or not IG_API_KEY:
        print("❌ IG API credentials missing.")
        return

    try:
        ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
        ig_service.create_session()
    except Exception as e:
        print(f"❌ Failed to connect to IG: {e}")
        return

    state = load_state()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    if state.get("date") != today_str:
        state = {"date": today_str, "daily_count": 0, "active_orders": {}}

    if state["daily_count"] >= MAX_TRADES_PER_DAY:
        print("🛑 Maximum daily limit of 3 trades already reached. Skipping execution.")
        return

    for ticker in QUANT_UNIVERSE:
        if state["daily_count"] >= MAX_TRADES_PER_DAY:
            break

        epic = "CS.D.SPY.CFD.IP" if ticker == "SPY" else "CS.D.QQQ.CFD.IP"

        # Prevent duplicate orders for the same asset on the same day
        if epic in state["active_orders"]:
            print(f"🔁 Skipping {ticker}: Already processed today.")
            continue

        pick = analyze_index_asset(ticker)
        if not pick:
            continue

        try:
            direction = "BUY" if pick["bias"] == "BULLISH" else "SELL"
            response = ig_service.create_working_order(
                currency_code="USD",
                direction=direction,
                epic=epic,
                expiry="-",
                level=pick["entry_price"],
                order_type="LIMIT",
                size=1,
                time_in_force="GOOD_TILL_CANCELLED",
                guaranteed_stop=False,
                stop_level=pick["stop_loss"],
                limit_level=pick["take_profit"]
            )

            deal_ref = response.get("dealReference")
            state["active_orders"][epic] = {
                "dealReference": deal_ref,
                "ticker": ticker,
                "timestamp": datetime.utcnow().isoformat()
            }
            state["daily_count"] += 1
            print(f"✅ Placed IG Limit Order for {ticker} | Entry: ${pick['entry_price']} | Ref: {deal_ref}")

            # Optional Discord notification
            if DISCORD_WEBHOOK_URL:
                requests.post(DISCORD_WEBHOOK_URL, json={
                    "username": "IG Paper Trade Bot",
                    "content": f"🚀 **IG Limit Order Placed**: `${ticker}` ({pick['bias']}) at `${pick['entry_price']}` | SL: `${pick['stop_loss']}` | TP: `${pick['take_profit']}`"
                })

        except IGException as e:
            print(f"❌ IG Order Error for {ticker}: {e}")

    save_state(state)


def monitor_trades():
    """Runs every 5 minutes to survey working orders and open positions."""
    try:
        ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, IG_ACC_TYPE)
        ig_service.create_session()

        working_resp = ig_service.fetch_working_orders()
        working_orders = working_resp.get("workingOrders", [])

        open_resp = ig_service.fetch_open_positions()
        positions = open_resp.get("positions", [])

        print(f"🔍 5-Min Monitor Check — Working Orders: {len(working_orders)} | Open Positions: {len(positions)}")

    except IGException as e:
        print(f"⚠️ Error during 5-minute trade monitor: {e}")


if __name__ == "__main__":
    mode = os.getenv("RUN_MODE", "monitor")
    if mode == "execute_daily":
        execute_daily_trades()
    else:
        monitor_trades()
