import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import requests
import yfinance as yf
import numpy as np

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY")

# STRICTLY INDEX-LEVEL SCOPE
QUANT_UNIVERSE = ["SPY", "QQQ"]


def fetch_finnhub_intelligence():
    """Pulls broad market and macroeconomic news from Finnhub to gauge general market sentiment."""
    headlines = []
    if not FINNHUB_API_KEY:
        print("⚠️ Finnhub API Key missing.")
        return "Finnhub API Key missing."

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        
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


def analyze_index_asset(ticker: str):
    """Pulls historical data for SPY or QQQ, computes volatility projections (lowest/highest price bounds)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="3mo", interval="1d")
        if df.empty or len(df) < 30:
            print(f"⚠️ Insufficient data returned for {ticker}")
            return None

        closes = df["Close"]
        current_price = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        daily_change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)

        # Technical Indicators (RSI & ATR for low/high projection)
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = round(float((100 - (100 / (1 + rs))).iloc[-1]), 1)

        # Average True Range (ATR) & Volatility boundaries for multi-day/session forecast
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - closes.shift())
        low_close = np.abs(df["Low"] - closes.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr_14 = float(true_range.rolling(14).mean().iloc[-1])

        # Predictive Price Bounds (Statistical projection)
        lowest_price_target = round(current_price - (1.5 * atr_14), 2)
        highest_price_target = round(current_price + (1.8 * atr_14), 2)

        # Volume analysis
        volume = int(df["Volume"].iloc[-1])
        avg_volume = float(df["Volume"].rolling(10).mean().iloc[-1])
        vol_ratio = round(volume / avg_volume, 2) if avg_volume > 0 else 1.0

        # Fixed function call name matching definition above
        macro_news = fetch_finnhub_intelligence()

        # Scoring (0-100)
        score = 70
        if 40 <= rsi <= 65:
            score += 15
        if vol_ratio > 1.0:
            score += 10

        score = max(50, min(98, score))

        # 3 PM UK / US Session Execution Levels
        ma_10 = float(closes.rolling(10).mean().iloc[-1])
        bias = "BULLISH" if current_price >= ma_10 else "BEARISH"
        vwap_approx = round(current_price * 1.0002, 2)
        entry_low = round(current_price * 0.997, 2)
        entry_high = round(current_price * 1.003, 2)
        sl_price = round(current_price * 0.985 if bias == "BULLISH" else current_price * 1.015, 2)
        tp_price = highest_price_target if bias == "BULLISH" else lowest_price_target

        return {
            "ticker": ticker,
            "name": "SPDR S&P 500 ETF Trust" if ticker == "SPY" else "Invesco QQQ Trust",
            "price": current_price,
            "daily_change": daily_change_pct,
            "rsi": rsi,
            "lowest_price": lowest_price_target,
            "highest_price": highest_price_target,
            "score": score,
            "bias": bias,
            "vwap": vwap_approx,
            "entry_zone": f"${entry_low} - ${entry_high}",
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "news": macro_news
        }
    except Exception as e:
        print(f"⚠️ Error analyzing index {ticker}: {repr(e)}")
        return None


def run_quant_predictor_scan():
    print("==================================================")
    print("  🔮 RUNNING SPY & QQQ INDEX QUANT PREDICTOR")
    print("==================================================")

    if not DISCORD_WEBHOOK_URL:
        print("❌ ERROR: DISCORD_WEBHOOK_URL environment variable is missing or empty!")
        return

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_ticker = {executor.submit(analyze_index_asset, ticker): ticker for ticker in QUANT_UNIVERSE}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res:
                print(f"✅ Index Analyzed: ${res['ticker']} ({res['bias']}) — Score: {res['score']}/100 | Low Target: ${res['lowest_price']} | High Target: ${res['highest_price']}")
                embed_payload = generate_quant_embed(res)
                
                # Dispatch to Discord with explicit status check logging
                response = requests.post(DISCORD_WEBHOOK_URL, json=embed_payload)
                print(f"📡 Discord Webhook Response for ${res['ticker']}: Status Code {response.status_code}")
                if response.status_code != 204:
                    print(f"⚠️ Discord Error Response Text: {response.text}")

    print("--> Index quant predictor scan complete. SPY and QQQ reports dispatched to Discord.")


def generate_quant_embed(pick: dict):
    """Formats the quantitative index prediction and 3 PM execution plan into a clean Discord embed."""
    description = (
        f"**SPY & QQQ INDEX QUANT FORECAST DESK**\n\n"
        f"| Metric | Projected Value | Macro Rationale |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Lowest Price Target** | `${pick['lowest_price']}` | Statistical downside trough (ATR buffer) |\n"
        f"| **Highest Price Target** | `${pick['highest_price']}` | Upside expansion target |\n"
        f"| **Current Price** | `${pick['price']}` (`{pick['daily_change']}%`) | Live session feed |\n"
        f"| **RSI & Momentum** | `{pick['rsi']} RSI` | Index momentum regime check |\n"
        f"| **Macro Headlines** | {pick['news'][:120]} | Broad market sentiment scan |\n\n"
        f"**3:00 PM UK Session Execution Plan:**\n"
        f"• **Directional Bias:** `{pick['bias']}`\n"
        f"• **Entry Zone:** `{pick['entry_zone']}` (VWAP: `${pick['vwap']}`)\n"
        f"• **Stop-Loss (SL):** `${pick['stop_loss']}`\n"
        f"• **Take-Profit (TP):** `{pick['take_profit']}`\n\n"
        f"🤖 *Engine: Index Quant Predictor + Finnhub Macro Feed*"
    )

    return {
        "username": "Index Quant Prediction Engine",
        "embeds": [{
            "title": f"📈 Index Forecast: ${pick['ticker']} ({pick['bias']}) — Score: {pick['score']}/100",
            "description": description[:3900],
            "color": 3447003 if pick['bias'] == "BULLISH" else 15158332,
            "footer": {"text": f"3 PM UK Execution Scan | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
        }]
    }


if __name__ == "__main__":
    run_quant_predictor_scan()
